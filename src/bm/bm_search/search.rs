use arrayvec::ArrayVec;
use cozy_chess::{BitBoard, Board, Move, Piece};

use crate::bm::bm_runner::ab_runner::{LocalContext, SharedContext, MAX_PLY};
use crate::bm::bm_search::move_entry::MoveEntry;
use crate::bm::bm_util::eval::Depth::Next;
use crate::bm::bm_util::eval::Evaluation;
use crate::bm::bm_util::h_table;
use crate::bm::bm_util::position::Position;
use crate::bm::bm_util::t_table::EntryType;
use crate::bm::bm_util::t_table::EntryType::{Exact, LowerBound, UpperBound};

use super::move_gen::OrderedMoveGen;
use super::move_gen::QuiescenceSearchMoveGen;

pub trait SearchType {
    const NM: bool;
    const PV: bool;
    type Zw: SearchType;
}

pub struct Pv;
pub struct Zw;
pub struct NoNm;

impl SearchType for Pv {
    const NM: bool = false;
    const PV: bool = true;
    type Zw = Zw;
}

impl SearchType for Zw {
    const NM: bool = true;
    const PV: bool = false;
    type Zw = Zw;
}

impl SearchType for NoNm {
    const NM: bool = false;
    const PV: bool = false;
    type Zw = NoNm;
}

#[inline]
const fn do_rev_fp(depth: u32) -> bool {
    depth < 7
}

#[inline]
const fn rev_fp(depth: u32, improving: bool) -> i16 {
    (depth as i16 - improving as i16) * 50
}

#[inline]
fn do_nmp<Search: SearchType>(board: &Board, depth: u32, eval: i16, beta: i16) -> bool {
    Search::NM
        && depth > 4
        && eval >= beta
        && (board.pieces(Piece::Pawn) | board.pieces(Piece::King)) != board.occupied()
}

#[inline]
fn nmp_depth(depth: u32, eval: i16, beta: i16) -> u32 {
    assert!(eval >= beta);
    let r = 3 + depth / 4 + ((eval - beta) / 200) as u32;
    depth.saturating_sub(r).max(1)
}

#[inline]
const fn iir(depth: u32) -> u32 {
    if depth >= 4 {
        1
    } else {
        0
    }
}

#[inline]
const fn fp(depth: u32) -> i16 {
    depth as i16 * 100
}

#[inline]
const fn see_fp(depth: u32) -> i16 {
    depth as i16 * 100
}

#[inline]
const fn hp(depth: u32) -> i32 {
    -h_table::MAX_VALUE * ((depth * depth) as i32) / 64
}

#[inline]
const fn history_lmr(history: i16) -> i16 {
    history / 80
}

#[inline]
const fn q_see_threshold() -> i16 {
    200
}

pub fn search<Search: SearchType>(
    pos: &mut Position,
    local_context: &mut LocalContext,
    shared_context: &SharedContext,
    ply: u32,
    mut depth: u32,
    mut alpha: Evaluation,
    beta: Evaluation,
) -> Evaluation {
    local_context.search_stack_mut()[ply as usize].pv_len = 0;

    if ply != 0 && shared_context.abort_search() {
        local_context.trigger_abort();
        return Evaluation::min();
    }

    local_context.update_sel_depth(ply);
    if ply != 0 && pos.forced_draw(ply) {
        local_context.increment_nodes();
        return Evaluation::new(0);
    }

    /*
    At depth 0, we run Quiescence Search
    */
    if depth == 0 || ply >= MAX_PLY {
        return q_search(pos, local_context, shared_context, ply, alpha, beta);
    }

    let skip_move = local_context.search_stack()[ply as usize].skip_move;
    let tt_entry = if skip_move.is_some() {
        None
    } else {
        shared_context.get_t_table().get(pos.board())
    };

    local_context.increment_nodes();

    let mut best_move = None;

    let initial_alpha = alpha;

    /*
    Transposition Table
    If we get a TT hit and the search is deep enough,
    we can use the score from TT to cause an early cutoff
    We also use the best move from the transposition table
    to help with move ordering
    */
    if let Some(entry) = tt_entry {
        *local_context.tt_hits() += 1;
        best_move = Some(entry.table_move());
        if !Search::PV && entry.depth() >= depth {
            let score = entry.score();
            match entry.entry_type() {
                Exact => {
                    return score;
                }
                LowerBound => {
                    if score >= beta {
                        return score;
                    }
                }
                UpperBound => {
                    if score <= alpha {
                        return score;
                    }
                }
            }
        }
    } else {
        *local_context.tt_misses() += 1;
    }

    let in_check = pos.board().checkers() != BitBoard::EMPTY;

    let eval = if skip_move.is_none() {
        pos.get_eval(local_context.stm(), local_context.eval())
    } else {
        local_context.search_stack()[ply as usize].eval
    };

    local_context.search_stack_mut()[ply as usize].eval = eval;
    let improving = if ply < 2 || in_check {
        false
    } else {
        eval > local_context.search_stack()[ply as usize - 2].eval
    };

    if !Search::PV && !in_check && skip_move.is_none() {
        /*
        Reverse Futility Pruning:
        If in a non PV node and evaluation is higher than beta + a depth dependent margin
        we assume we can at least achieve beta
        */
        if do_rev_fp(depth) && eval - rev_fp(depth, improving) >= beta {
            return eval;
        }

        /*
        Null Move Pruning:
        If in a non PV node and we can still achieve beta at a reduced depth after
        giving the opponent the side to move we can prune this node and return the evaluation
        While doing null move pruning, we also get the "best move" for the opponent in case
        This is seen as the major threat in the current position and can be used in
        move ordering for the next ply
        */
        if do_nmp::<Search>(pos.board(), depth, eval.raw(), beta.raw()) && pos.null_move() {
            local_context.search_stack_mut()[ply as usize].move_played = None;

            let nmp_depth = nmp_depth(depth, eval.raw(), beta.raw());
            let zw = beta >> Next;
            let search_score = search::<NoNm>(
                pos,
                local_context,
                shared_context,
                ply + 1,
                nmp_depth,
                zw,
                zw + 1,
            );
            pos.unmake_move();
            let score = search_score << Next;
            if score >= beta {
                let mut verified = depth < 10;
                if !verified {
                    let verification = search::<NoNm>(
                        pos,
                        local_context,
                        shared_context,
                        ply + 1,
                        nmp_depth,
                        alpha,
                        beta,
                    );
                    verified = verification >= beta;
                }
                if verified {
                    return score;
                }
            }
        }
        let prob_beta = beta + 50;
        if depth > 4
            && !tt_entry.map_or(false, |entry| {
                entry.depth() >= depth - 3 && entry.score() < prob_beta
            })
        {
            let mut move_gen = QuiescenceSearchMoveGen::new();
            while let Some((make_move, _)) =
                move_gen.next(pos.board(), local_context.get_ch_table())
            {
                local_context.search_stack_mut()[ply as usize].move_played = Some(make_move);
                pos.make_move(make_move);

                let zw = prob_beta >> Next;
                let mut score =
                    q_search(pos, local_context, shared_context, ply + 1, zw, zw + 1) << Next;
                if score >= prob_beta {
                    score = search::<Search::Zw>(
                        pos,
                        local_context,
                        shared_context,
                        ply + 1,
                        depth - 4,
                        zw,
                        zw + 1,
                    ) << Next;
                }
                pos.unmake_move();
                if score >= prob_beta {
                    return score;
                }
            }
        }
    }

    if tt_entry.is_none() {
        depth -= iir(depth)
    }

    while local_context.get_k_table().len() <= ply as usize {
        local_context.get_k_table().push(MoveEntry::new());
    }

    if let Some(entry) = local_context.get_k_table().get_mut(ply as usize + 1) {
        entry.clear();
    }

    let mut highest_score = None;

    let prev_move = if ply != 0 {
        Some(local_context.search_stack()[ply as usize - 1].move_played)
    } else {
        None
    };

    let counter_move = if let Some(Some(prev_move)) = prev_move {
        local_context.get_cm_table().get(
            pos.board().side_to_move(),
            pos.board().piece_on(prev_move.to).unwrap_or(Piece::King),
            prev_move.to,
        )
    } else {
        None
    };

    let killers = local_context.get_k_table()[ply as usize];
    let mut move_gen = OrderedMoveGen::new(
        pos.board(),
        best_move,
        counter_move,
        prev_move.unwrap_or(None),
        killers.into_iter(),
    );

    let mut moves_seen = 0;
    let mut move_exists = false;

    let mut quiets = ArrayVec::<Move, 64>::new();
    let mut captures = ArrayVec::<Move, 64>::new();

    while let Some(make_move) = move_gen.next(
        pos.board(),
        local_context.get_h_table(),
        local_context.get_ch_table(),
        local_context.get_cm_hist(),
    ) {
        if Some(make_move) == skip_move {
            continue;
        }
        local_context.search_stack_mut()[ply as usize + 1].pv_len = 0;

        move_exists = true;
        let is_capture = pos
            .board()
            .colors(!pos.board().side_to_move())
            .has(make_move.to);

        let h_score = if is_capture {
            local_context.get_ch_table().get(
                pos.board().side_to_move(),
                make_move.from,
                make_move.to,
            )
        } else {
            local_context.get_h_table().get(
                pos.board().side_to_move(),
                make_move.from,
                make_move.to,
            )
        };

        let mut extension = 0;
        let mut score;

        /*
        Singular Extensions:
        If a move can't be beaten by any other move, we assume the move
        is singular (only solution) and extend in order to get a more accurate
        estimation of best move/eval
        */
        if let Some(entry) = tt_entry {
            if moves_seen == 0
                && entry.table_move() == make_move
                && ply != 0
                && !entry.score().is_mate()
                && entry.depth() + 2 >= depth
                && matches!(entry.entry_type(), EntryType::LowerBound | EntryType::Exact)
            {
                let s_beta = entry.score() - depth as i16 * 3;
                local_context.search_stack_mut()[ply as usize].skip_move = Some(make_move);

                let multi_cut = depth >= 7;
                let s_score = if multi_cut {
                    search::<Search::Zw>(
                        pos,
                        local_context,
                        shared_context,
                        ply,
                        depth / 2 - 1,
                        s_beta - 1,
                        s_beta,
                    )
                } else {
                    eval
                };

                local_context.search_stack_mut()[ply as usize].skip_move = None;
                if s_score < s_beta {
                    if s_beta + 250 <= alpha {
                        return alpha;
                    }
                    extension = 1;
                } else if multi_cut && s_beta >= beta {
                    /*
                    Multi-cut:
                    If a move isn't singular and the move that disproves the singularity
                    our singular beta is above beta, we assume the move is good enough to beat beta
                    */
                    return s_beta;
                }
            }
        }

        let non_mate_line = highest_score.map_or(false, |s: Evaluation| !s.is_mate());
        /*
        In non-PV nodes If a move isn't good enough to beat alpha - a static margin
        we assume it's safe to prune this move
        */
        let do_fp = !Search::PV && non_mate_line && moves_seen > 0 && !is_capture && depth <= 7;

        if do_fp && eval + fp(depth) <= alpha {
            move_gen.set_skip_quiets(true);
            continue;
        }

        /*
        If a move is placed late in move ordering, we can safely prune it based on a depth related margin
        */
        if !move_gen.skip_quiets()
            && non_mate_line
            && !is_capture
            && quiets.len()
                >= shared_context
                    .get_lmp_lookup()
                    .get(depth as usize, improving as usize)
        {
            move_gen.set_skip_quiets(true);
            continue;
        }

        /*
        In low depth, non-PV nodes, we assume it's safe to prune a move
        if it has very low history
        */
        let do_hp = !Search::PV && non_mate_line && moves_seen > 0 && depth <= 8 && eval <= alpha;

        if do_hp && (h_score as i32) < hp(depth) {
            continue;
        }

        /*
        In non-PV nodes If a move evaluated by SEE isn't good enough to beat alpha - a static margin
        we assume it's safe to prune this move
        */
        let do_see_prune = !Search::PV && non_mate_line && moves_seen > 0 && depth <= 7;
        if do_see_prune && eval + see::<16>(pos.board(), make_move) + see_fp(depth) <= alpha {
            continue;
        }

        pos.make_move(make_move);
        shared_context.get_t_table().prefetch(pos.board());
        local_context.search_stack_mut()[ply as usize].move_played = Some(make_move);
        let gives_check = pos.board().checkers() != BitBoard::EMPTY;
        if gives_check {
            extension = 1;
        }

        /*
        LMR
        We try to prove a move is worse than alpha at a reduced depth
        If the move proves to be worse than alpha, we don't have to do a
        full depth search
        */
        let mut reduction = shared_context
            .get_lmr_lookup()
            .get(depth as usize, moves_seen) as i16;

        if moves_seen > 0 {
            /*
            If a move is quiet, we already have information on this move
            in the history table. If history score is high, we reduce
            less and if history score is low we reduce more.
            */

            reduction -= history_lmr(h_score);
            if Search::PV {
                reduction -= 1;
            };
            if improving {
                reduction -= 1;
            }
            if Some(make_move) == counter_move
                || killers.into_iter().any(|killer| killer == make_move)
            {
                reduction -= 1;
            }
            reduction = reduction.min(depth as i16 - 2).max(0);
        }

        let lmr_depth = (depth as i16 - reduction) as u32;

        if moves_seen == 0 {
            let search_score = search::<Search>(
                pos,
                local_context,
                shared_context,
                ply + 1,
                depth - 1 + extension,
                beta >> Next,
                alpha >> Next,
            );
            score = search_score << Next;
        } else {
            //Reduced Search/Zero Window if no reduction
            let zw = alpha >> Next;

            let lmr_score = search::<Search::Zw>(
                pos,
                local_context,
                shared_context,
                ply + 1,
                lmr_depth - 1 + extension,
                zw - 1,
                zw,
            );
            score = lmr_score << Next;

            /*
            If no reductions occured in LMR we don't waste time re-searching
            otherwise, we run a full depth search to attempt a fail low
            */
            if lmr_depth < depth && score > alpha {
                let zw_score = search::<Search::Zw>(
                    pos,
                    local_context,
                    shared_context,
                    ply + 1,
                    depth - 1 + extension,
                    zw - 1,
                    zw,
                );
                score = zw_score << Next;
            }
            /*
            If we don't get a fail low, this means the move has to be searched fully
            */
            if Search::PV && score > alpha {
                let search_score = search::<Search>(
                    pos,
                    local_context,
                    shared_context,
                    ply + 1,
                    depth - 1 + extension,
                    beta >> Next,
                    alpha >> Next,
                );
                score = search_score << Next;
            }
        }

        pos.unmake_move();
        moves_seen += 1;

        if highest_score.is_none() || score > highest_score.unwrap() {
            highest_score = Some(score);
            best_move = Some(make_move);
            if score > alpha {
                if Search::PV || (ply == 0 && moves_seen == 1) {
                    let (child_pv, len) = {
                        let child = &local_context.search_stack()[ply as usize + 1];
                        (child.pv, child.pv_len)
                    };
                    local_context.search_stack_mut()[ply as usize]
                        .update_pv(make_move, &child_pv[..len]);
                }
                if score >= beta {
                    if !local_context.abort() {
                        let amt = depth + extension;
                        if !is_capture {
                            let killer_table = local_context.get_k_table();
                            killer_table[ply as usize].push(make_move);
                            local_context.get_h_table_mut().cutoff(
                                pos.board(),
                                make_move,
                                &quiets,
                                amt,
                            );
                            if let Some(Some(prev_move)) = prev_move {
                                local_context.get_cm_table_mut().cutoff(
                                    pos.board(),
                                    prev_move,
                                    make_move,
                                    amt,
                                );
                                local_context.get_cm_hist_mut().cutoff(
                                    pos.board(),
                                    prev_move,
                                    make_move,
                                    &quiets,
                                    amt,
                                );
                            }
                        } else {
                            local_context.get_ch_table_mut().cutoff(
                                pos.board(),
                                make_move,
                                &captures,
                                amt,
                            );
                        }
                    }
                    break;
                }
                alpha = score;
            }
        }
        if is_capture {
            if !captures.is_full() {
                captures.push(make_move);
            }
        } else if !quiets.is_full() {
            quiets.push(make_move);
        }
    }
    if !move_exists {
        return if pos.board().checkers() == BitBoard::EMPTY {
            Evaluation::new(0)
        } else {
            Evaluation::new_checkmate(-1)
        };
    }
    let highest_score = highest_score.unwrap();

    if skip_move.is_none() && !local_context.abort() {
        if let Some(final_move) = &best_move {
            let entry_type = if highest_score > initial_alpha {
                if highest_score >= beta {
                    LowerBound
                } else {
                    Exact
                }
            } else {
                UpperBound
            };
            shared_context.get_t_table().set(
                pos.board(),
                depth,
                entry_type,
                highest_score,
                *final_move,
            );
        }
    }
    highest_score
}

/*
Quiescence Search is a form of search that only searches tactical moves to achieve a quiet position.
This is done as the static evaluation function isn't suited to detecting tactical aspects of the position.
*/

pub fn q_search(
    pos: &mut Position,
    local_context: &mut LocalContext,
    shared_context: &SharedContext,
    ply: u32,
    mut alpha: Evaluation,
    beta: Evaluation,
) -> Evaluation {
    local_context.increment_nodes();

    local_context.update_sel_depth(ply);
    if ply >= MAX_PLY {
        return pos.get_eval(local_context.stm(), local_context.eval());
    }

    let initial_alpha = alpha;
    let tt_entry = shared_context.get_t_table().get(pos.board());
    if let Some(entry) = tt_entry {
        match entry.entry_type() {
            LowerBound => {
                if entry.score() >= beta {
                    return entry.score();
                }
            }
            Exact => return entry.score(),
            UpperBound => {
                if entry.score() <= alpha {
                    return entry.score();
                }
            }
        }
    }

    let mut highest_score = None;
    let mut best_move = None;
    let in_check = pos.board().checkers() != BitBoard::EMPTY;

    let stand_pat = pos.get_eval(local_context.stm(), local_context.eval());
    /*
    If not in check, we have a stand pat score which is the static eval of the current position.
    This is done as captures aren't necessarily the best moves.
    */
    if !in_check && stand_pat > alpha {
        alpha = stand_pat;
        highest_score = Some(stand_pat);
        if stand_pat >= beta {
            return stand_pat;
        }
    }

    let mut move_gen = QuiescenceSearchMoveGen::new();
    while let Some((make_move, see)) = move_gen.next(pos.board(), local_context.get_ch_table()) {
        let is_capture = pos
            .board()
            .colors(!pos.board().side_to_move())
            .has(make_move.to);
        if in_check || is_capture {
            /*
            SEE beta cutoff: (Koivisto)
            If SEE considerably improves evaluation above beta, we can return beta early
            */
            if stand_pat + see - q_see_threshold() >= beta {
                return beta;
            }
            if stand_pat + see + q_see_threshold() <= alpha {
                continue;
            }
            pos.make_move(make_move);
            let search_score = q_search(
                pos,
                local_context,
                shared_context,
                ply + 1,
                beta >> Next,
                alpha >> Next,
            );
            let score = search_score << Next;
            if highest_score.is_none() || score > highest_score.unwrap() {
                highest_score = Some(score);
                best_move = Some(make_move);
            }
            if score > alpha {
                alpha = score;
                if score >= beta {
                    pos.unmake_move();
                    break;
                }
            }
            pos.unmake_move();
        }
    }
    if let (Some(best_move), Some(highest_score)) = (best_move, highest_score) {
        let entry_type = if highest_score > initial_alpha {
            if highest_score >= beta {
                LowerBound
            } else {
                Exact
            }
        } else {
            UpperBound
        };

        shared_context
            .get_t_table()
            .set(pos.board(), 0, entry_type, highest_score, best_move);
    }
    highest_score.unwrap_or(alpha)
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
                Piece::Pawn => cozy_chess::get_pawn_attacks(target_square, !color),
                Piece::Knight => cozy_chess::get_knight_moves(target_square),
                Piece::Bishop => cozy_chess::get_bishop_moves(target_square, blockers),
                Piece::Rook => cozy_chess::get_rook_moves(target_square, blockers),
                Piece::Queen => {
                    cozy_chess::get_rook_moves(target_square, blockers)
                        | cozy_chess::get_bishop_moves(target_square, blockers)
                }
                Piece::King => cozy_chess::get_king_moves(target_square),
            } & board.pieces(piece)
                & defenders;
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
