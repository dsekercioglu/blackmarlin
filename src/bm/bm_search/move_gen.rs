use cozy_chess::{Board, Move, Piece, PieceMoves};

use crate::bm::bm_util::h_table::{DoubleMoveHistory, HistoryTable};
use arrayvec::ArrayVec;

use super::move_entry::MoveEntryIterator;
use super::tactics;

const MAX_MOVES: usize = 218;
const THRESHOLD: i16 = -(2_i16.pow(10));
const LOSING_CAPTURE: i16 = -(2_i16.pow(12));

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum GenType {
    PvMove,
    CalcCaptures,
    Captures,
    GenQuiet,
    CounterMove,
    Killer,
    Quiet,
    BadCaptures,
}

type LazySee = Option<i16>;

pub struct OrderedMoveGen<const K: usize> {
    move_list: ArrayVec<PieceMoves, 18>,
    pv_move: Option<Move>,
    killer_entry: MoveEntryIterator<K>,
    counter_move: Option<Move>,
    prev_move: Option<Move>,
    gen_type: GenType,

    captures: ArrayVec<(Move, i16, LazySee), MAX_MOVES>,
    quiets: ArrayVec<(Move, i16), MAX_MOVES>,
    skip_quiets: bool,
}

impl<const K: usize> OrderedMoveGen<K> {
    pub fn new(
        board: &Board,
        pv_move: Option<Move>,
        counter_move: Option<Move>,
        prev_move: Option<Move>,
        killer_entry: MoveEntryIterator<K>,
    ) -> Self {
        let mut move_list = ArrayVec::new();
        board.generate_moves(|piece_moves| {
            move_list.push(piece_moves);
            false
        });
        Self {
            gen_type: GenType::PvMove,
            move_list,
            counter_move,
            prev_move,
            pv_move,
            killer_entry,
            captures: ArrayVec::new(),
            quiets: ArrayVec::new(),
            skip_quiets: false,
        }
    }

    pub fn set_skip_quiets(&mut self, value: bool) {
        self.skip_quiets = value;
    }

    pub fn skip_quiets(&self) -> bool {
        self.skip_quiets
    }

    fn set_phase(&mut self) {
        if self.skip_quiets {
            match self.gen_type {
                GenType::GenQuiet | GenType::CounterMove | GenType::Killer | GenType::Quiet => {
                    self.gen_type = GenType::BadCaptures
                }
                _ => {}
            }
        }
    }

    pub fn next(
        &mut self,
        board: &Board,
        hist: &HistoryTable,
        c_hist: &HistoryTable,
        cm_hist: &DoubleMoveHistory,
    ) -> Option<Move> {
        self.set_phase();
        if self.gen_type == GenType::PvMove {
            self.gen_type = GenType::CalcCaptures;
            if let Some(pv_move) = self.pv_move {
                for &piece_moves in &self.move_list {
                    if piece_moves.from != pv_move.from {
                        continue;
                    }
                    for mv in piece_moves {
                        if mv == pv_move {
                            return Some(pv_move);
                        }
                    }
                }
                self.pv_move = None;
            }
        }
        if self.gen_type == GenType::CalcCaptures {
            for &piece_moves in &self.move_list {
                let mut piece_moves = piece_moves;
                piece_moves.to &= board.colors(!board.side_to_move());
                for make_move in piece_moves {
                    if Some(make_move) == self.pv_move {
                        continue;
                    }
                    let expected_gain =
                        c_hist.get(board.side_to_move(), piece_moves.piece, make_move.to)
                            + tactics::see::<1>(&board, make_move) * 32;
                    self.captures.push((make_move, expected_gain, None));
                }
            }

            self.gen_type = GenType::Captures;
        }
        if self.gen_type == GenType::Captures {
            let mut max = THRESHOLD;
            let mut best_index = None;
            for (index, (make_move, score, see)) in self.captures.iter_mut().enumerate() {
                if *score > max {
                    let see_score = see.unwrap_or_else(|| tactics::see::<16>(&board, *make_move));
                    *see = Some(see_score);
                    if see_score < 0 {
                        *score += LOSING_CAPTURE;
                        continue;
                    }
                    max = *score;
                    best_index = Some(index);
                }
            }
            if let Some(index) = best_index {
                return Some(self.captures.swap_remove(index).0);
            } else {
                self.gen_type = if self.skip_quiets {
                    GenType::BadCaptures
                } else {
                    GenType::GenQuiet
                }
            }
        }
        if self.gen_type == GenType::GenQuiet {
            for &piece_moves in &self.move_list {
                let mut piece_moves = piece_moves;
                piece_moves.to &= !board.colors(!board.side_to_move());
                for make_move in piece_moves {
                    if Some(make_move) == self.pv_move {
                        continue;
                    }
                    if let Some(piece) = make_move.promotion {
                        match piece {
                            cozy_chess::Piece::Queen => {
                                self.quiets.push((make_move, i16::MAX));
                            }
                            _ => {
                                self.quiets.push((make_move, i16::MIN));
                            }
                        };
                        continue;
                    }
                    let mut score = 0;
                    let piece = board.piece_on(make_move.from).unwrap();

                    if tactics::threatens(board, make_move) {
                        score += 256;
                    }
                    score += hist.get(board.side_to_move(), piece, make_move.to);
                    if let Some(prev_move) = self.prev_move {
                        let prev_move_piece = board.piece_on(prev_move.to).unwrap_or(Piece::King);
                        score += cm_hist.get(
                            board.side_to_move(),
                            prev_move_piece,
                            prev_move.to,
                            piece,
                            make_move.to,
                        );
                    }

                    self.quiets.push((make_move, score));
                }
            }
            self.gen_type = GenType::Killer;
        }
        //Assumes Killer Moves won't repeat
        if self.gen_type == GenType::Killer {
            for make_move in self.killer_entry.clone() {
                let position = self
                    .quiets
                    .iter()
                    .position(|(cmp_move, _)| make_move == *cmp_move);
                if let Some(position) = position {
                    self.quiets.swap_remove(position);
                    return Some(make_move);
                }
            }
            self.gen_type = GenType::CounterMove;
        }
        if self.gen_type == GenType::CounterMove {
            self.gen_type = GenType::Quiet;
            if let Some(counter_move) = self.counter_move {
                let position = self
                    .quiets
                    .iter()
                    .position(|(cmp_move, _)| counter_move == *cmp_move);
                if let Some(position) = position {
                    self.quiets.swap_remove(position);
                    return Some(counter_move);
                }
            }
        }
        if self.gen_type == GenType::Quiet {
            let mut max = 0;
            let mut best_index = None;
            for (index, &(_, score)) in self.quiets.iter().enumerate() {
                if best_index.is_none() || score > max {
                    max = score;
                    best_index = Some(index);
                }
            }
            if let Some(index) = best_index {
                return Some(self.quiets.swap_remove(index).0);
            } else {
                self.gen_type = GenType::BadCaptures;
            };
        }
        let mut max = 0;
        let mut best_index = None;
        for (index, &(_, score, _)) in self.captures.iter().enumerate() {
            if best_index.is_none() || score > max {
                max = score;
                best_index = Some(index);
            }
        }
        if let Some(index) = best_index {
            Some(self.captures.swap_remove(index).0)
        } else {
            None
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum QSearchGenType {
    CalcCaptures,
    Captures,
}

pub struct QuiescenceSearchMoveGen {
    gen_type: QSearchGenType,
    queue: ArrayVec<(Move, i16, LazySee), MAX_MOVES>,
}

impl QuiescenceSearchMoveGen {
    pub fn new() -> Self {
        Self {
            gen_type: QSearchGenType::CalcCaptures,
            queue: ArrayVec::new(),
        }
    }

    pub fn next(&mut self, board: &Board, c_hist: &HistoryTable) -> Option<(Move, i16)> {
        if self.gen_type == QSearchGenType::CalcCaptures {
            board.generate_moves(|mut piece_moves| {
                piece_moves.to &= board.colors(!board.side_to_move());
                for make_move in piece_moves {
                    let expected_gain =
                        c_hist.get(board.side_to_move(), piece_moves.piece, make_move.to)
                            + tactics::see::<1>(&board, make_move) * 32;
                    self.queue.push((make_move, expected_gain, None));
                }
                false
            });
            self.gen_type = QSearchGenType::Captures;
        }
        let mut max = 0;
        let mut best_index = None;
        for (index, (make_move, score, see)) in self.queue.iter_mut().enumerate() {
            if best_index.is_none() || *score > max {
                let see_score = see.unwrap_or_else(|| tactics::see::<16>(&board, *make_move));
                *see = Some(see_score);
                if see_score < 0 {
                    continue;
                }
                max = *score;
                best_index = Some(index);
            }
        }
        if let Some(index) = best_index {
            let out = self.queue.swap_remove(index);
            Some((out.0, out.2.unwrap()))
        } else {
            None
        }
    }
}
