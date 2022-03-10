use cozy_chess::{BitBoard, Board, Move, Piece};

use crate::bm::nnue::Nnue;

use super::eval::Evaluation;

#[derive(Debug, Clone)]
pub struct Position {
    current: Board,
    boards: Vec<Board>,
    evaluator: Nnue,
}

impl Position {
    pub fn new(board: Board) -> Self {
        let mut evaluator = Nnue::new();
        evaluator.reset(&board);
        Self {
            current: board,
            boards: vec![],
            evaluator,
        }
    }

    pub fn reset(&mut self) {
        self.evaluator.reset(&self.current);
    }

    #[inline]
    pub fn forced_draw(&self, ply: u32) -> bool {
        if self.insufficient_material() || self.half_ply() >= 100 {
            return true;
        }
        let hash = self.hash();
        self.boards
            .iter()
            .rev()
            .skip(1)
            .take(ply as usize)
            .any(|board| board.hash() == hash)
            || self
                .boards
                .iter()
                .rev()
                .skip(ply as usize + 1)
                .filter(|board| board.hash() == hash)
                .count()
                >= 2
    }

    #[inline]
    pub fn board(&self) -> &Board {
        &self.current
    }

    #[inline]
    pub fn half_ply(&self) -> u8 {
        self.current.halfmove_clock()
    }

    #[inline]
    pub fn null_move(&mut self) -> bool {
        if let Some(new_board) = self.board().null_move() {
            self.evaluator.null_move();
            self.boards.push(self.current.clone());
            self.current = new_board;
            true
        } else {
            false
        }
    }

    #[inline]
    pub fn make_move(&mut self, make_move: Move) {
        self.evaluator.make_move(&self.current, make_move);
        self.boards.push(self.current.clone());
        self.current.play_unchecked(make_move);
    }

    #[inline]
    pub fn unmake_move(&mut self) {
        self.evaluator.unmake_move();
        let current = self.boards.pop().unwrap();
        self.current = current;
    }

    #[inline]
    pub fn hash(&self) -> u64 {
        self.board().hash()
    }

    pub fn get_eval(&self) -> Evaluation {
        Evaluation::new(self.evaluator.feed_forward(self.board(), 0))
    }

    pub fn get_move_eval(&self, make_move: Move) -> i16 {
        self.evaluator.evaluate_move(self.board(), make_move)
    }

    pub fn insufficient_material(&self) -> bool {
        if self.current.occupied().popcnt() == 2 {
            true
        } else if self.current.occupied().popcnt() == 3 {
            (self.current.pieces(Piece::Rook)
                | self.current.pieces(Piece::Queen)
                | self.current.pieces(Piece::Pawn))
                == BitBoard::EMPTY
        } else {
            false
        }
    }
}
