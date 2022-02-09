use arrayvec::ArrayVec;
use cozy_chess::{BitBoard, Move, Piece};

use crate::bm::bm_eval::eval::Depth::Next;
use crate::bm::bm_eval::eval::Evaluation;
use crate::bm::bm_eval::evaluator::StdEvaluator;
use crate::bm::bm_runner::ab_runner::{LocalContext, SharedContext, MAX_PLY, SEARCH_PARAMS};
use crate::bm::bm_search::move_entry::MoveEntry;
use crate::bm::bm_util::h_table;
use crate::bm::bm_util::position::Position;
use crate::bm::bm_util::t_table::EntryType::{Exact, LowerBound, UpperBound};
use crate::bm::bm_util::t_table::{Analysis, EntryType};

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

const MIN_PIECE_CNT: u32 = 2;

pub fn search<Search: SearchType>(
    position: &mut Position,
    local_context: &mut LocalContext,
    shared_context: &SharedContext,
    ply: u32,
    mut target_ply: u32,
    mut alpha: Evaluation,
    beta: Evaluation,
) -> (Option<Move>, Evaluation) {
    if ply != 0 && shared_context.abort_search() {
        local_context.trigger_abort();
        return (None, Evaluation::min());
    }

    local_context.update_sel_depth(ply);
    if ply != 0 && position.forced_draw(ply) {
        local_context.increment_nodes();
        return (None, Evaluation::new(0));
    }

    /*
    At depth 0, we run Quiescence Search
    */
    if ply >= target_ply || ply >= MAX_PLY {
        return (
            None,
            q_search(
                position,
                local_context,
                shared_context,
                ply,
                ply + SEARCH_PARAMS.get_q_search_depth(),
                alpha,
                beta,
            ),
        );
    }

    let skip_move = local_context.search_stack()[ply as usize].skip_move;
    let tt_entry = if skip_move.is_some() {
        None
    } else {
        shared_context.get_t_table().get(position.board())
    };

    local_context.increment_nodes();

    let mut best_move = None;

    let board = position.board().clone();

    let initial_alpha = alpha;

    let mut depth = target_ply - ply;

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
        if !Search::PV && ply + entry.depth() >= target_ply {
            let score = entry.score();
            match entry.entry_type() {
                Exact => {
                    return (best_move, score);
                }
                LowerBound => {
                    if score >= beta {
                        return (best_move, score);
                    }
                }
                UpperBound => {
                    if score <= alpha {
                        return (best_move, score);
                    }
                }
            }
        }
    } else {
        *local_context.tt_misses() += 1;
    }

    let in_check = position.board().checkers() != BitBoard::EMPTY;

    let eval = if skip_move.is_none() {
        position.get_eval()
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
        let do_rev_f_prune = SEARCH_PARAMS.do_rev_fp() && SEARCH_PARAMS.do_rev_f_prune(depth);
        if do_rev_f_prune {
            let f_margin = SEARCH_PARAMS.get_rev_fp().threshold(depth);
            if eval - f_margin + (improving as i16) * 50 >= beta {
                return (None, eval);
            }
        }

        /*
        Null Move Pruning:
        If in a non PV node and we can still achieve beta at a reduced depth after
        giving the opponent the side to move we can prune this node and return the evaluation
        While doing null move pruning, we also get the "best move" for the opponent in case
        This is seen as the major threat in the current position and can be used in
        move ordering for the next ply
        */

        let only_pawns =
            MIN_PIECE_CNT + board.pieces(Piece::Pawn).popcnt() == board.occupied().popcnt();
        let do_null_move = SEARCH_PARAMS.do_nmp(depth) && Search::NM && !only_pawns;

        if do_null_move && eval >= beta && position.null_move() {
            {
                let threat_table = local_context.get_threat_table();
                while threat_table.len() <= ply as usize + 1 {
                    threat_table.push(MoveEntry::new());
                }
            }
            local_context.search_stack_mut()[ply as usize].move_played = None;

            let zw = beta >> Next;
            let reduction =
                SEARCH_PARAMS.get_nmp().reduction(depth) + ((eval - beta).raw() / 200) as u32;
            let r_target_ply = target_ply.saturating_sub(reduction).max(ply + 2);
            let (threat_move, search_score) = search::<NoNm>(
                position,
                local_context,
                shared_context,
                ply + 1,
                r_target_ply,
                zw,
                zw + 1,
            );
            if let Some(threat_move) = threat_move {
                let threat_table = local_context.get_threat_table();
                threat_table[ply as usize + 1].push(threat_move)
            }
            position.unmake_move();
            let score = search_score << Next;
            if score >= beta {
                return (None, score);
            }
        }
    }

    if tt_entry.is_none() && depth >= 4 {
        depth -= 1;
        target_ply -= 1;
    }

    /*
    Internal Iterative Deepening
    In PV nodes, if we don't have a move from the transposition table, we do a reduced
    depth search to get a good estimation on what the best move is
    This is currently disabled
    */
    let do_iid = SEARCH_PARAMS.do_iid(depth) && Search::PV && !in_check;
    if do_iid && best_move.is_none() {
        let reduction = SEARCH_PARAMS.get_iid().reduction(depth);
        let target_ply = target_ply.max(reduction) - reduction;
        let (iid_move, _) = search::<Search>(
            position,
            local_context,
            shared_context,
            ply,
            target_ply,
            alpha,
            beta,
        );
        best_move = iid_move;
    }

    while local_context.get_k_table().len() <= ply as usize {
        local_context.get_k_table().push(MoveEntry::new());
        local_context.get_threat_table().push(MoveEntry::new());
    }

    if let Some(entry) = local_context.get_k_table().get_mut(ply as usize + 2) {
        entry.clear();
    }
    if let Some(entry) = local_context.get_threat_table().get_mut(ply as usize + 2) {
        entry.clear();
    }

    let mut highest_score = None;

    let threat_move_entry = if ply > 1 {
        local_context.get_threat_table()[ply as usize]
    } else {
        MoveEntry::new()
    };

    let prev_move = if ply != 0 {
        Some(local_context.search_stack()[ply as usize - 1].move_played)
    } else {
        None
    };

    let counter_move = if let Some(Some(prev_move)) = prev_move {
        local_context.get_cm_table().get(
            board.side_to_move(),
            board.piece_on(prev_move.to).unwrap_or(Piece::King),
            prev_move.to,
        )
    } else {
        None
    };

    let mut move_gen = OrderedMoveGen::new(
        position.board(),
        best_move,
        counter_move,
        prev_move.unwrap_or(None),
        threat_move_entry.into_iter(),
        local_context.get_k_table()[ply as usize].into_iter(),
    );

    let mut moves_seen = 0;
    let mut move_exists = false;

    let mut quiets = ArrayVec::<Move, 64>::new();
    let mut captures = ArrayVec::<Move, 64>::new();

    while let Some(make_move) = move_gen.next(
        local_context.get_h_table(),
        local_context.get_ch_table(),
        local_context.get_cm_hist(),
    ) {
        if Some(make_move) == skip_move {
            continue;
        }
        move_exists = true;
        let is_capture = board.colors(!board.side_to_move()).has(make_move.to);

        let h_score = if is_capture {
            local_context.get_ch_table().get(
                board.side_to_move(),
                board.piece_on(make_move.from).unwrap(),
                make_move.to,
            )
        } else {
            local_context.get_h_table().get(
                board.side_to_move(),
                board.piece_on(make_move.from).unwrap(),
                make_move.to,
            )
        };
        let cmh_score = if let Some(Some(prev_move)) = prev_move {
            let piece = board.piece_on(make_move.from).unwrap();
            let prev_move_piece = board.piece_on(prev_move.to).unwrap_or(Piece::King);
            Some(local_context.get_cm_hist().get(
                board.side_to_move(),
                prev_move_piece,
                prev_move.to,
                piece,
                make_move.to,
            ))
        } else {
            None
        };

        let mut extension = 0;
        let mut score;
        if moves_seen == 0 {
            /*
            Singular Extensions:
            If a move can't be beaten by any other move, we assume the move
            is singular (only solution) and extend in order to get a more accurate
            estimation of best move/eval
            */
            if let Some(entry) = tt_entry {
                if entry.table_move() == make_move
                    && ply != 0
                    && depth >= 7
                    && !entry.score().is_mate()
                    && entry.depth() >= depth - 2
                    && (entry.entry_type() == EntryType::LowerBound
                        || entry.entry_type() == EntryType::Exact)
                {
                    let reduced_plies = ply + depth / 2 - 1;
                    let s_beta = entry.score() - depth as i16 * 3;
                    local_context.search_stack_mut()[ply as usize].skip_move = Some(make_move);
                    let (_, s_score) = search::<Search::Zw>(
                        position,
                        local_context,
                        shared_context,
                        ply,
                        reduced_plies,
                        s_beta - 1,
                        s_beta,
                    );
                    local_context.search_stack_mut()[ply as usize].skip_move = None;
                    if s_score < s_beta {
                        extension += 1;
                    } else if s_beta >= beta {
                        /*
                        Multi-cut:
                        If a move isn't singular and the move that disproves the singularity
                        our singular beta is above beta, we assume the move is good enough to beat beta
                        */
                        return (Some(make_move), s_beta);
                    }
                }
            }
            position.make_move(make_move);
            local_context.search_stack_mut()[ply as usize].move_played = Some(make_move);

            let gives_check = position.board().checkers() != BitBoard::EMPTY;
            if gives_check {
                extension += 1;
            }

            /*
            First moves don't get reduced
            */
            let (_, search_score) = search::<Search>(
                position,
                local_context,
                shared_context,
                ply + 1,
                target_ply + extension,
                beta >> Next,
                alpha >> Next,
            );
            score = search_score << Next;
        } else {
            /*
            In non-PV nodes If a move isn't good enough to beat alpha - a static margin
            we assume it's safe to prune this move
            */
            let do_fp = !Search::PV && !is_capture && SEARCH_PARAMS.do_fp() && depth <= 7;

            if do_fp && eval + SEARCH_PARAMS.get_fp() * (depth as i16) < alpha {
                move_gen.set_skip_quiets(true);
                continue;
            }

            let history_prune_margin = -h_table::MAX_VALUE * ((depth * depth) as i32) / 64;
            /*
            In low depth, non-PV nodes, we assume it's safe to prune a move
            if it has very low history
            */
            let do_hp = !Search::PV && depth <= 8 && eval <= alpha;

            if do_hp && (h_score as i32) < history_prune_margin {
                continue;
            }

            let do_cmh_p = !Search::PV && depth <= 2 && eval <= alpha;
            if let Some(cmh_score) = cmh_score {
                if do_cmh_p && !is_capture && (cmh_score as i32) < history_prune_margin {
                    continue;
                }
            }

            /*
            If a move is placed late in move ordering, we can safely prune it based on a depth related margin
            */
            if SEARCH_PARAMS.do_lmp()
                && !move_gen.skip_quiets()
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
            In non-PV nodes If a move evaluated by SEE isn't good enough to beat alpha - a static margin
            we assume it's safe to prune this move
            */
            let do_see_prune = !Search::PV && !in_check && depth <= 7;
            if do_see_prune
                && eval
                    + StdEvaluator::see::<16>(&board, make_move)
                    + SEARCH_PARAMS.get_fp() * (depth as i16)
                    < alpha
            {
                continue;
            }

            position.make_move(make_move);
            local_context.search_stack_mut()[ply as usize].move_played = Some(make_move);
            let gives_check = position.board().checkers() != BitBoard::EMPTY;
            if gives_check {
                extension += 1;
            }
            /*
            LMR
            We try to prove a move is worse than alpha at a reduced depth
            If the move proves to be worse than alpha, we don't have to do a
            full depth search
            */
            let mut reduction = 0_i16;
            let do_lmr = SEARCH_PARAMS.do_lmr(depth);

            if do_lmr {
                reduction = shared_context
                    .get_lmr_lookup()
                    .get(depth as usize, moves_seen) as i16;

                /*
                If a move is quiet, we already have information on this move
                in the history table. If history score is high, we reduce
                less and if history score is low we reduce more.
                */

                reduction -= h_score / SEARCH_PARAMS.get_h_reduce_div();
                if Search::PV {
                    reduction -= 1;
                };
                if improving {
                    reduction -= 1;
                }
                reduction = reduction.min(depth as i16 - 1).max(0);
            }

            let lmr_ply = (target_ply as i16 - reduction).max(0) as u32;
            //Reduced Search/Zero Window if no reduction
            let zw = alpha >> Next;

            let (_, lmr_score) = search::<Search::Zw>(
                position,
                local_context,
                shared_context,
                ply + 1,
                lmr_ply + extension,
                zw - 1,
                zw,
            );
            score = lmr_score << Next;

            /*
            If no reductions occured in LMR we don't waste time re-searching
            otherwise, we run a full depth search to attempt a fail low
            */
            if lmr_ply < target_ply && score > alpha {
                let (_, zw_score) = search::<Search::Zw>(
                    position,
                    local_context,
                    shared_context,
                    ply + 1,
                    target_ply + extension,
                    zw - 1,
                    zw,
                );
                score = zw_score << Next;
            }
            /*
            If we don't get a fail low, this means the move has to be searched fully
            */
            if Search::PV && score > alpha {
                let (_, search_score) = search::<Search>(
                    position,
                    local_context,
                    shared_context,
                    ply + 1,
                    target_ply + extension,
                    beta >> Next,
                    alpha >> Next,
                );
                score = search_score << Next;
            }
        }

        position.unmake_move();
        moves_seen += 1;

        if highest_score.is_none() || score > highest_score.unwrap() {
            highest_score = Some(score);
            best_move = Some(make_move);
        }
        if score > alpha {
            if score >= beta {
                if skip_move.is_none() && !local_context.abort() {
                    if !is_capture {
                        let killer_table = local_context.get_k_table();
                        killer_table[ply as usize].push(make_move);
                        local_context
                            .get_h_table_mut()
                            .cutoff(&board, make_move, &quiets, depth);
                        if let Some(Some(prev_move)) = prev_move {
                            local_context
                                .get_cm_table_mut()
                                .cutoff(&board, prev_move, make_move, depth);
                            local_context
                                .get_cm_hist_mut()
                                .cutoff(&board, prev_move, make_move, &quiets, depth);
                        }
                    } else {
                        local_context
                            .get_ch_table_mut()
                            .cutoff(&board, make_move, &captures, depth);
                    }

                    let analysis = Analysis::new(depth, LowerBound, score, make_move);
                    shared_context.get_t_table().set(position.board(), analysis);
                }
                return (Some(make_move), score);
            }
            alpha = score;
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
        return if board.checkers() == BitBoard::EMPTY {
            (None, Evaluation::new(0))
        } else {
            (None, Evaluation::new_checkmate(-1))
        };
    }
    let highest_score = highest_score.unwrap();

    if skip_move.is_none() && !local_context.abort() {
        if let Some(final_move) = &best_move {
            let entry_type = if highest_score > initial_alpha {
                Exact
            } else {
                UpperBound
            };

            let analysis = Analysis::new(depth, entry_type, highest_score, *final_move);
            shared_context.get_t_table().set(position.board(), analysis);
        }
    }
    (best_move, highest_score)
}

/*
Quiescence Search is a form of search that only searches tactical moves to achieve a quiet position.
This is done as the static evaluation function isn't suited to detecting tactical aspects of the position.
*/

pub fn q_search(
    position: &mut Position,
    local_context: &mut LocalContext,
    shared_context: &SharedContext,
    ply: u32,
    target_ply: u32,
    mut alpha: Evaluation,
    beta: Evaluation,
) -> Evaluation {
    local_context.increment_nodes();

    local_context.update_sel_depth(ply);
    if ply >= target_ply || ply >= MAX_PLY {
        return position.get_eval();
    }

    let initial_alpha = alpha;
    let tt_entry = shared_context.get_t_table().get(position.board());
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

    let board = position.board().clone();
    let mut highest_score = None;
    let mut best_move = None;
    let in_check = board.checkers() != BitBoard::EMPTY;

    let stand_pat = position.get_eval();
    /*
    If not in check, we have a stand pat score which is the static eval of the current position.
    This is done as captures aren't necessarily the best moves.
    */
    if !in_check {
        /*
        If stand pat is way below alpha, assume it can't be beaten.
        */
        let do_dp = SEARCH_PARAMS.do_dp();
        if do_dp && stand_pat + SEARCH_PARAMS.get_delta() < alpha {
            return stand_pat;
        }
        if stand_pat > alpha {
            alpha = stand_pat;
            highest_score = Some(stand_pat);
            if stand_pat >= beta {
                return stand_pat;
            }
        }
    }

    let mut move_gen = QuiescenceSearchMoveGen::new(&board);
    while let Some((make_move, see)) = move_gen.next(local_context.get_ch_table()) {
        let is_capture = board.colors(!board.side_to_move()).has(make_move.to);
        if in_check || is_capture {
            /*
            SEE beta cutoff: (Koivisto)
            If SEE considerably improves evaluation above beta, we can return beta early
            */
            if stand_pat + see - 200 > beta {
                return beta;
            }
            position.make_move(make_move);
            let search_score = q_search(
                position,
                local_context,
                shared_context,
                ply + 1,
                target_ply,
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
                    position.unmake_move();

                    let analysis = Analysis::new(0, LowerBound, score, make_move);
                    shared_context.get_t_table().set(position.board(), analysis);
                    return score;
                }
            }
            position.unmake_move();
        }
    }
    if let (Some(best_move), Some(highest_score)) = (best_move, highest_score) {
        let entry_type = if highest_score > initial_alpha {
            Exact
        } else {
            UpperBound
        };

        let analysis = Analysis::new(0, entry_type, highest_score, best_move);
        shared_context.get_t_table().set(position.board(), analysis);
    }
    highest_score.unwrap_or(alpha)
}
