use cozy_chess::{BitBoard, Board, Color, File, Move, Piece, Rank, Square};

use self::normal::{Dense, Incremental, Psqt};

use super::bm_runner::ab_runner;

mod normal;

include!(concat!(env!("OUT_DIR"), "/nnue_weights.rs"));
include!(concat!(env!("OUT_DIR"), "/policy_weights.rs"));

#[derive(Debug, Clone)]
pub struct Accumulator {
    w_input_layer: Incremental<'static, INPUT, MID>,
    b_input_layer: Incremental<'static, INPUT, MID>,
    w_res_layer: Psqt<'static, INPUT, OUTPUT>,
    b_res_layer: Psqt<'static, INPUT, OUTPUT>,

    w_policy_input: Incremental<'static, INPUT, 256>,
    b_policy_input: Incremental<'static, INPUT, 256>,
}

impl Accumulator {
    pub fn update<const INCR: bool>(&mut self, sq: Square, piece: Piece, color: Color) {
        let w_piece_index = color as usize * 6 + piece as usize;
        let b_piece_index = (!color) as usize * 6 + piece as usize;

        let w_index = sq as usize + w_piece_index * 64;
        let b_index = (sq as usize ^ 56) + b_piece_index * 64;

        if INCR {
            self.w_input_layer.incr_ff::<1>(w_index);
            self.w_res_layer.incr_ff::<1>(w_index);
            self.b_input_layer.incr_ff::<1>(b_index);
            self.b_res_layer.incr_ff::<1>(b_index);
            self.w_policy_input.incr_ff::<1>(b_index);
            self.b_policy_input.incr_ff::<1>(b_index);
        } else {
            self.w_input_layer.incr_ff::<-1>(w_index);
            self.w_res_layer.incr_ff::<-1>(w_index);
            self.b_input_layer.incr_ff::<-1>(b_index);
            self.b_res_layer.incr_ff::<-1>(b_index);
            self.w_policy_input.incr_ff::<-1>(b_index);
            self.b_policy_input.incr_ff::<-1>(b_index);
        }
    }
}

#[derive(Debug, Clone)]
pub struct Nnue {
    accumulator: Vec<Accumulator>,
    head: usize,
    out_layer: Dense<'static, MID, OUTPUT>,
}

impl Nnue {
    pub fn new() -> Self {
        let input_layer = Incremental::new(&INCREMENTAL, INCREMENTAL_BIAS);
        let res_layer = Psqt::new(&PSQT);
        let out_layer = Dense::new(&OUT, OUT_BIAS);

        let policy_input = Incremental::new(&P_WEIGHTS_0, P_BIAS_0);

        Self {
            accumulator: vec![
                Accumulator {
                    w_input_layer: input_layer.clone(),
                    b_input_layer: input_layer,
                    w_res_layer: res_layer.clone(),
                    b_res_layer: res_layer,
                    w_policy_input: policy_input.clone(),
                    b_policy_input: policy_input,
                };
                ab_runner::MAX_PLY as usize + 1
            ],
            head: 0,
            out_layer,
        }
    }

    pub fn reset(&mut self, board: &Board) {
        self.head = 0;
        let accumulator = &mut self.accumulator[0];
        accumulator.w_input_layer.reset(INCREMENTAL_BIAS);
        accumulator.b_input_layer.reset(INCREMENTAL_BIAS);
        accumulator.w_res_layer.reset();
        accumulator.b_res_layer.reset();
        accumulator.w_policy_input.reset(P_BIAS_0);
        accumulator.b_policy_input.reset(P_BIAS_0);
        for sq in board.occupied() {
            let piece = board.piece_on(sq).unwrap();
            let color = board.color_on(sq).unwrap();
            accumulator.update::<true>(sq, piece, color);
        }
    }

    pub fn null_move(&mut self) {
        self.accumulator[self.head + 1] = self.accumulator[self.head].clone();
        self.head += 1;
    }

    pub fn make_move(&mut self, board: &Board, make_move: Move) {
        self.accumulator[self.head + 1] = self.accumulator[self.head].clone();
        self.head += 1;
        let acc = &mut self.accumulator[self.head];

        let from_sq = make_move.from;
        let from_type = board.piece_on(from_sq).unwrap();
        let from_color = board.side_to_move();
        acc.update::<false>(from_sq, from_type, from_color);

        let to_sq = make_move.to;
        if let Some((captured, color)) = board.piece_on(to_sq).zip(board.color_on(to_sq)) {
            acc.update::<false>(to_sq, captured, color);
        }

        if let Some(ep) = board.en_passant() {
            let (stm_fifth, stm_sixth) = match from_color {
                Color::White => (Rank::Fifth, Rank::Sixth),
                Color::Black => (Rank::Fourth, Rank::Third),
            };
            if from_type == Piece::Pawn && to_sq == Square::new(ep, stm_sixth) {
                acc.update::<false>(Square::new(ep, stm_fifth), Piece::Pawn, !from_color);
            }
        }
        if Some(from_color) == board.color_on(to_sq) {
            let stm_first = match from_color {
                Color::White => Rank::First,
                Color::Black => Rank::Eighth,
            };
            if to_sq.file() > from_sq.file() {
                acc.update::<true>(Square::new(File::G, stm_first), Piece::King, from_color);
                acc.update::<true>(Square::new(File::F, stm_first), Piece::Rook, from_color);
            } else {
                acc.update::<true>(Square::new(File::C, stm_first), Piece::King, from_color);
                acc.update::<true>(Square::new(File::D, stm_first), Piece::Rook, from_color);
            }
        } else {
            acc.update::<true>(to_sq, make_move.promotion.unwrap_or(from_type), from_color);
        }
    }

    pub fn unmake_move(&mut self) {
        self.head -= 1;
    }

    #[inline]
    pub fn feed_forward(&self, board: &Board, bucket: usize) -> i16 {
        let acc = &self.accumulator[self.head];
        let (incr_layer, psqt_score) = match board.side_to_move() {
            Color::White => (
                normal::clipped_relu(*acc.w_input_layer.get()),
                acc.w_res_layer.get()[bucket] / 64,
            ),
            Color::Black => (
                normal::clipped_relu(*acc.b_input_layer.get()),
                acc.b_res_layer.get()[bucket] / 64,
            ),
        };

        psqt_score as i16 + normal::out(self.out_layer.ff(&incr_layer, bucket)[bucket])
    }

    #[inline]
    pub fn evaluate_move(&self, board: &Board, make_move: Move) -> i16 {
        let acc = &self.accumulator[self.head];
        let incr_layer = match board.side_to_move() {
            Color::White => (normal::clipped_relu(*acc.w_policy_input.get())),
            Color::Black => (normal::clipped_relu(*acc.b_policy_input.get())),
        };
        let move_piece = board.piece_on(make_move.from).unwrap() as usize;
        let move_sq = match board.side_to_move() {
            Color::White => make_move.to as usize,
            Color::Black => make_move.to as usize ^ 56,
        };
        let move_index = move_piece * 64 + move_sq;

        let mut sum = P_BIAS_1[move_index] as i32;
        for (&weight, &val) in P_WEIGHTS_1[move_index].iter().zip(&incr_layer) {
            sum += weight as i32 * val as i32;
        }
        normal::out(sum)
    }
}
