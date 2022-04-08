use cozy_chess::{BitBoard, Board, Color, Move, Piece, Square};

pub fn threatens(board: &Board, make_move: Move) -> bool {
    let piece = board.piece_on(make_move.from).unwrap();
    let stm = board.side_to_move();
    let opp = board.colors(!stm);

    let opp_minors = opp & (board.pieces(Piece::Knight) | board.pieces(Piece::Bishop));
    let opp_majors = opp & (board.pieces(Piece::Rook) | board.pieces(Piece::Queen));

    let from_attacks = get_piece_attacks(piece, stm, make_move.from, opp_minors, opp_majors);
    let to_attacks = get_piece_attacks(piece, stm, make_move.to, opp_minors, opp_majors);
    !(to_attacks & !from_attacks).is_empty()
}

fn get_piece_attacks(
    piece: Piece,
    color: Color,
    sq: Square,
    opp_minors: BitBoard,
    opp_majors: BitBoard,
) -> BitBoard {
    match piece {
        Piece::Pawn => cozy_chess::get_pawn_attacks(sq, color) & (opp_minors | opp_majors),
        Piece::Knight => cozy_chess::get_knight_moves(sq) & opp_majors,
        Piece::Bishop => cozy_chess::get_bishop_moves(sq, opp_majors),
        _ => BitBoard::EMPTY,
    }
}

pub fn see<const N: usize>(board: &Board, make_move: Move) -> i16 {
    let mut index = 0;
    let mut gains = [0_i16; N];
    let target_square = make_move.to;
    let move_piece = board.piece_on(make_move.from).unwrap();
    gains[0] = if let Some(piece) = board.piece_on(target_square) {
        piece_pts(piece)
    } else {
        if move_piece == Piece::King {
            return 0;
        }
        0
    };
    let mut color = !board.side_to_move();
    let mut blockers = board.occupied() & !make_move.from.bitboard();
    let mut last_piece_pts = piece_pts(move_piece);
    'outer: for i in 1..N {
        gains[i] = last_piece_pts - gains[i - 1];
        let defenders = board.colors(color) & blockers;
        for &piece in &Piece::ALL {
            last_piece_pts = piece_pts(piece);
            let mut potential = match piece {
                Piece::Pawn => {
                    cozy_chess::get_pawn_attacks(target_square, !color)
                        & defenders
                        & board.pieces(Piece::Pawn)
                }
                Piece::Knight => {
                    cozy_chess::get_knight_moves(target_square)
                        & board.pieces(Piece::Knight)
                        & defenders
                }
                Piece::Bishop => {
                    cozy_chess::get_bishop_moves(target_square, blockers)
                        & defenders
                        & board.pieces(Piece::Bishop)
                }
                Piece::Rook => {
                    cozy_chess::get_rook_moves(target_square, blockers)
                        & board.pieces(Piece::Rook)
                        & defenders
                }
                Piece::Queen => {
                    cozy_chess::get_rook_moves(target_square, blockers)
                        & cozy_chess::get_bishop_moves(target_square, blockers)
                        & board.pieces(Piece::Queen)
                        & defenders
                }
                Piece::King => {
                    cozy_chess::get_king_moves(target_square)
                        & board.pieces(Piece::King)
                        & defenders
                }
            };
            if potential != BitBoard::EMPTY {
                let attacker = potential.next().unwrap();
                blockers &= !attacker.bitboard();
                color = !color;
                continue 'outer;
            }
        }
        index = i;
        break;
    }
    for i in (1..index).rev() {
        gains[i - 1] = -i16::max(-gains[i - 1], gains[i]);
    }
    gains[0]
}

fn piece_pts(piece: Piece) -> i16 {
    match piece {
        Piece::Pawn => 100,
        Piece::Knight => 300,
        Piece::Bishop => 300,
        Piece::Rook => 500,
        Piece::Queen => 900,
        Piece::King => 20000,
    }
}
