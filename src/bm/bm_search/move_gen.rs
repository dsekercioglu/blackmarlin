use chess::{Board, ChessMove, MoveGen, EMPTY};

use crate::bm::bm_eval::evaluator::StdEvaluator;

use crate::bm::bm_util::h_table::{DoubleMoveHistory, HistoryTable};
use arrayvec::ArrayVec;

use super::move_entry::MoveEntryIterator;

const MAX_MOVES: usize = 218;
const LOSING_CAPTURE: i16 = -(2_i16.pow(10));

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum GenType {
    PvMove,
    CalcCaptures,
    Captures,
    GenQuiet,
    CounterMove,
    Killer,
    ThreatMove,
    Quiet,
}

pub struct OrderedMoveGen<const T: usize, const K: usize> {
    move_gen: MoveGen,
    pv_move: Option<ChessMove>,
    threat_move_entry: MoveEntryIterator<T>,
    killer_entry: MoveEntryIterator<K>,
    counter_move: Option<ChessMove>,
    prev_move: Option<ChessMove>,
    gen_type: GenType,
    board: Board,

    queue: ArrayVec<(ChessMove, i16), MAX_MOVES>,
}

impl<const T: usize, const K: usize> OrderedMoveGen<T, K> {
    pub fn new(
        board: &Board,
        pv_move: Option<ChessMove>,
        counter_move: Option<ChessMove>,
        prev_move: Option<ChessMove>,
        threat_move_entry: MoveEntryIterator<T>,
        killer_entry: MoveEntryIterator<K>,
    ) -> Self {
        Self {
            gen_type: GenType::PvMove,
            move_gen: MoveGen::new_legal(board),
            counter_move,
            prev_move,
            pv_move,
            threat_move_entry,
            killer_entry,
            board: *board,
            queue: ArrayVec::new(),
        }
    }

    pub fn next(&mut self, hist: &HistoryTable, cm_hist: &DoubleMoveHistory) -> Option<ChessMove> {
        if self.gen_type == GenType::PvMove {
            self.gen_type = GenType::CalcCaptures;
            if let Some(pv_move) = self.pv_move {
                if self.board.legal(pv_move) {
                    return Some(pv_move);
                } else {
                    self.pv_move = None;
                }
            }
        }
        if self.gen_type == GenType::CalcCaptures {
            self.move_gen.set_iterator_mask(*self.board.combined());
            for make_move in &mut self.move_gen {
                if Some(make_move) != self.pv_move {
                    let mut expected_gain = StdEvaluator::see(self.board, make_move);
                    if expected_gain < 0 {
                        expected_gain += LOSING_CAPTURE;
                    }
                    self.queue.push((make_move, expected_gain));
                }
            }
            self.gen_type = GenType::Captures;
        }
        if self.gen_type == GenType::Captures {
            let mut max = LOSING_CAPTURE;
            let mut best_index = None;
            for (index, &(_, score)) in self.queue.iter().enumerate() {
                if score >= max {
                    max = score;
                    best_index = Some(index);
                }
            }
            if let Some(index) = best_index {
                return Some(self.queue.swap_remove(index).0);
            } else {
                self.gen_type = GenType::GenQuiet;
            }
        }
        if self.gen_type == GenType::GenQuiet {
            self.move_gen.set_iterator_mask(!EMPTY);
            for make_move in &mut self.move_gen {
                if Some(make_move) == self.pv_move {
                    continue;
                }
                if let Some(piece) = make_move.get_promotion() {
                    match piece {
                        chess::Piece::Queen => {
                            self.queue.push((make_move, i16::MAX));
                        }
                        _ => {
                            self.queue.push((make_move, i16::MIN));
                        }
                    };
                    continue;
                }
                let mut score = 0;
                let piece = self.board.piece_on(make_move.get_source()).unwrap();

                score += hist.get(self.board.side_to_move(), piece, make_move.get_dest());
                if let Some(prev_move) = self.prev_move {
                    let prev_move_piece = self.board.piece_on(prev_move.get_dest()).unwrap();
                    score += cm_hist.get(
                        self.board.side_to_move(),
                        prev_move_piece,
                        prev_move.get_dest(),
                        piece,
                        make_move.get_dest(),
                    );
                }

                self.queue.push((make_move, score));
            }
            self.gen_type = GenType::Killer;
        }
        //Assumes Killer Moves won't repeat
        if self.gen_type == GenType::Killer {
            for make_move in self.killer_entry.clone() {
                let position = self
                    .queue
                    .iter()
                    .position(|(cmp_move, _)| make_move == *cmp_move);
                if let Some(position) = position {
                    self.queue.swap_remove(position);
                    return Some(make_move);
                }
            }
            self.gen_type = GenType::CounterMove;
        }
        if self.gen_type == GenType::CounterMove {
            self.gen_type = GenType::Quiet;
            if let Some(counter_move) = self.counter_move {
                let position = self
                    .queue
                    .iter()
                    .position(|(cmp_move, _)| counter_move == *cmp_move);
                if let Some(position) = position {
                    self.queue.swap_remove(position);
                    return Some(counter_move);
                }
            }
        }
        if self.gen_type == GenType::ThreatMove {
            for make_move in &mut self.threat_move_entry {
                let position = self
                    .queue
                    .iter()
                    .position(|(cmp_move, _)| make_move == *cmp_move);
                if let Some(position) = position {
                    self.queue.swap_remove(position);
                    return Some(make_move);
                }
            }
            self.gen_type = GenType::Quiet;
        }
        if self.gen_type == GenType::Quiet {
            let mut max = 0;
            let mut best_index = None;
            for (index, &(_, score)) in self.queue.iter().enumerate() {
                if best_index.is_none() || score > max {
                    max = score;
                    best_index = Some(index);
                }
            }
            return if let Some(index) = best_index {
                Some(self.queue.swap_remove(index).0)
            } else {
                None
            };
        }
        None
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum QSearchGenType {
    CalcCaptures,
    Captures,
    Quiet,
}

pub struct QuiescenceSearchMoveGen<const SEE_PRUNE: bool> {
    move_gen: MoveGen,
    board: Board,
    gen_type: QSearchGenType,
    queue: ArrayVec<(ChessMove, i16), MAX_MOVES>,
}

impl<const SEE_PRUNE: bool> QuiescenceSearchMoveGen<SEE_PRUNE> {
    pub fn new(board: &Board) -> Self {
        Self {
            board: *board,
            move_gen: MoveGen::new_legal(board),
            gen_type: QSearchGenType::CalcCaptures,
            queue: ArrayVec::new(),
        }
    }
}

impl<const SEE_PRUNE: bool> Iterator for QuiescenceSearchMoveGen<SEE_PRUNE> {
    type Item = ChessMove;

    fn next(&mut self) -> Option<Self::Item> {
        if self.gen_type == QSearchGenType::CalcCaptures {
            self.move_gen.set_iterator_mask(*self.board.combined());
            for make_move in &mut self.move_gen {
                let expected_gain = StdEvaluator::see(self.board, make_move);
                if !SEE_PRUNE || expected_gain > -1 {
                    self.queue.push((make_move, expected_gain));
                }
            }
            self.gen_type = QSearchGenType::Captures;
        }
        if self.gen_type == QSearchGenType::Captures {
            let mut max = 0;
            let mut best_index = None;
            for (index, &(_, score)) in self.queue.iter().enumerate() {
                if best_index.is_none() || score > max {
                    max = score;
                    best_index = Some(index);
                }
            }
            if let Some(index) = best_index {
                return Some(self.queue.swap_remove(index).0);
            }
            self.move_gen.set_iterator_mask(!*self.board.combined());
            self.gen_type = QSearchGenType::Quiet;
        }
        self.move_gen.next()
    }
}
