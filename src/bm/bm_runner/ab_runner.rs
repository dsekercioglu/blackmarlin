use std::sync::Arc;
use std::time::Instant;

use chess::{Board, ChessMove, Color, Piece, Square};

use crate::bm::bm_eval::eval::Evaluation;
use crate::bm::bm_runner::ab_consts::*;
use crate::bm::bm_runner::config::{GuiInfo, NoInfo, SearchMode, SearchStats};
use crate::bm::bm_search::move_entry::MoveEntry;
use crate::bm::bm_search::reduction::Reduction;
use crate::bm::bm_search::search;
use crate::bm::bm_search::search::Pv;
use crate::bm::bm_search::threshold::Threshold;
use crate::bm::bm_util::h_table::{CounterMoveTable, DoubleMoveHistory, HistoryTable};
use crate::bm::bm_util::lookup::LookUp2d;
use crate::bm::bm_util::position::Position;
use crate::bm::bm_util::t_table::TranspositionTable;
use crate::bm::bm_util::window::Window;

use super::time::TimeManager;

pub const SEARCH_PARAMS: SearchParams = SearchParams {
    killer_move_cnt: KILLER_MOVE_CNT,
    threat_move_cnt: THREAT_MOVE_CNT,
    fail_cnt: FAIL_CNT,
    iid_depth: IID_DEPTH,
    rev_f_prune_depth: REV_F_PRUNE_DEPTH,
    fp: F_PRUNE_THRESHOLD,
    do_fp: DO_F_PRUNE,
    rev_fp: Threshold::new(REV_F_PRUNE_THRESHOLD_BASE, REV_F_PRUNE_THRESHOLD_FACTOR),
    do_rev_fp: DO_REV_F_PRUNE,
    nmp: Reduction::new(
        NULL_MOVE_REDUCTION_BASE,
        NULL_MOVE_REDUCTION_FACTOR,
        NULL_MOVE_REDUCTION_DIVISOR,
    ),
    nmp_depth: NULL_MOVE_PRUNE_DEPTH,
    do_nmp: DO_NULL_MOVE_REDUCTION,
    iid: Reduction::new(IID_BASE, IID_FACTOR, IID_DIVISOR),
    do_iid: DO_IID,
    lmr_depth: LMR_DEPTH,
    do_lmr: DO_LMR,
    do_lmp: DO_LMP,
    q_search_depth: QUIESCENCE_SEARCH_DEPTH,
    delta_margin: DELTA_MARGIN,
    do_dp: DO_DELTA_PRUNE,
    do_see_prune: DO_SEE_PRUNE,
    h_reduce_divisor: HISTORY_REDUCTION_DIVISOR,
};

#[derive(Debug, Clone)]
pub struct SearchParams {
    killer_move_cnt: usize,
    threat_move_cnt: usize,
    fail_cnt: u8,
    iid_depth: u32,
    fp: i16,
    do_fp: bool,
    rev_f_prune_depth: u32,
    rev_fp: Threshold,
    do_rev_fp: bool,
    nmp: Reduction,
    nmp_depth: u32,
    do_nmp: bool,
    iid: Reduction,
    do_iid: bool,
    lmr_depth: u32,
    do_lmr: bool,
    do_lmp: bool,
    q_search_depth: u32,
    delta_margin: i16,
    do_dp: bool,
    do_see_prune: bool,
    h_reduce_divisor: i16,
}

impl SearchParams {
    #[inline]
    pub const fn get_k_move_cnt(&self) -> usize {
        self.killer_move_cnt
    }

    #[inline]
    pub const fn get_threat_move_cnt(&self) -> usize {
        self.threat_move_cnt
    }

    #[inline]
    pub const fn get_q_search_depth(&self) -> u32 {
        self.q_search_depth
    }

    #[inline]
    pub const fn get_delta(&self) -> i16 {
        self.delta_margin
    }

    #[inline]
    pub const fn do_dp(&self) -> bool {
        self.do_dp
    }

    #[inline]
    pub const fn do_see_prune(&self) -> bool {
        self.do_see_prune
    }

    #[inline]
    pub const fn do_rev_f_prune(&self, depth: u32) -> bool {
        depth < self.rev_f_prune_depth
    }

    #[inline]
    pub const fn get_rev_fp(&self) -> &Threshold {
        &self.rev_fp
    }

    #[inline]
    pub const fn do_rev_fp(&self) -> bool {
        self.do_rev_fp
    }

    #[inline]
    pub const fn get_fp(&self) -> i16 {
        self.fp
    }

    #[inline]
    pub const fn do_fp(&self) -> bool {
        self.do_fp
    }

    #[inline]
    pub const fn get_nmp(&self) -> &Reduction {
        &self.nmp
    }

    #[inline]
    pub const fn do_nmp(&self, depth: u32) -> bool {
        self.do_nmp && depth >= self.nmp_depth
    }

    #[inline]
    pub const fn get_iid(&self) -> &Reduction {
        &self.iid
    }

    #[inline]
    pub const fn do_iid(&self, depth: u32) -> bool {
        self.do_iid && depth > self.iid_depth
    }

    #[inline]
    pub const fn do_lmr(&self, depth: u32) -> bool {
        self.do_lmr && depth >= self.lmr_depth
    }

    #[inline]
    pub const fn do_lmp(&self) -> bool {
        self.do_lmp
    }

    pub fn get_h_reduce_div(&self) -> i16 {
        self.h_reduce_divisor
    }
}

type LmrLookup = LookUp2d<u32, 32, 64>;
type LmpLookup = LookUp2d<usize, { LMP_DEPTH as usize }, 2>;

#[derive(Debug, Clone)]
pub struct SharedContext {
    start: Instant,
    time_manager: Arc<TimeManager>,

    t_table: Arc<TranspositionTable>,
    lmr_lookup: Arc<LmrLookup>,
    lmp_lookup: Arc<LmpLookup>,
}

#[derive(Debug, Clone)]
pub struct LocalContext {
    window: Window,
    tt_hits: u32,
    tt_misses: u32,
    eval: Evaluation,
    eval_stack: Vec<Evaluation>,
    skip_moves: Vec<Option<ChessMove>>,
    variation: Vec<Option<(Color, Piece, Square)>>,
    sel_depth: u32,
    h_table: HistoryTable,
    ch_table: HistoryTable,
    cm_table: CounterMoveTable,
    cm_hist: DoubleMoveHistory,
    fu_hist: DoubleMoveHistory,
    killer_moves: Vec<MoveEntry<{ SEARCH_PARAMS.get_k_move_cnt() }>>,
    threat_moves: Vec<MoveEntry<{ SEARCH_PARAMS.get_threat_move_cnt() }>>,
    nodes: u32,
    abort: bool,
}

impl SharedContext {
    #[inline]
    pub fn abort_absolute(&self, depth: u32, nodes: u32) -> bool {
        self.time_manager.abort(self.start, depth, nodes)
    }

    #[inline]
    pub fn get_t_table(&self) -> &Arc<TranspositionTable> {
        &self.t_table
    }

    #[inline]
    pub fn get_lmr_lookup(&self) -> &Arc<LmrLookup> {
        &self.lmr_lookup
    }

    #[inline]
    pub fn get_lmp_lookup(&self) -> &Arc<LmpLookup> {
        &self.lmp_lookup
    }
}

impl LocalContext {
    #[inline]
    pub fn get_threat_table(&mut self) -> &mut Vec<MoveEntry<THREAT_MOVE_CNT>> {
        &mut self.threat_moves
    }

    #[inline]
    pub fn get_h_table(&self) -> &HistoryTable {
        &self.h_table
    }

    #[inline]
    pub fn get_ch_table(&self) -> &HistoryTable {
        &self.ch_table
    }

    #[inline]
    pub fn get_cm_table(&self) -> &CounterMoveTable {
        &self.cm_table
    }

    #[inline]
    pub fn get_cm_hist(&self) -> &DoubleMoveHistory {
        &self.cm_hist
    }

    #[inline]
    pub fn get_fu_hist(&self) -> &DoubleMoveHistory {
        &self.fu_hist
    }

    #[inline]
    pub fn get_h_table_mut(&mut self) -> &mut HistoryTable {
        &mut self.h_table
    }

    #[inline]
    pub fn get_ch_table_mut(&mut self) -> &mut HistoryTable {
        &mut self.ch_table
    }

    #[inline]
    pub fn get_cm_table_mut(&mut self) -> &mut CounterMoveTable {
        &mut self.cm_table
    }

    #[inline]
    pub fn get_cm_hist_mut(&mut self) -> &mut DoubleMoveHistory {
        &mut self.cm_hist
    }

    #[inline]
    pub fn get_fu_hist_mut(&mut self) -> &mut DoubleMoveHistory {
        &mut self.fu_hist
    }

    #[inline]
    pub fn get_k_table(&mut self) -> &mut Vec<MoveEntry<KILLER_MOVE_CNT>> {
        &mut self.killer_moves
    }

    #[inline]
    pub fn tt_hits(&mut self) -> &mut u32 {
        &mut self.tt_hits
    }

    #[inline]
    pub fn tt_misses(&mut self) -> &mut u32 {
        &mut self.tt_misses
    }

    #[inline]
    pub fn get_eval(&self, ply: u32) -> Option<Evaluation> {
        self.eval_stack.get(ply as usize).copied()
    }

    #[inline]
    pub fn push_move(&mut self, make_move: Option<(Color, Piece, Square)>, ply: u32) {
        if ply as usize >= self.variation.len() {
            self.variation.push(make_move);
        } else {
            self.variation[ply as usize] = make_move;
        }
    }

    #[inline]
    pub fn get_move(&self, ply: u32) -> Option<Option<(Color, Piece, Square)>> {
        self.variation.get(ply as usize).copied()
    }

    #[inline]
    pub fn push_eval(&mut self, eval: Evaluation, ply: u32) {
        if ply as usize >= self.eval_stack.len() {
            self.eval_stack.push(eval);
        } else {
            self.eval_stack[ply as usize] = eval;
        }
    }

    #[inline]
    pub fn set_skip_move(&mut self, ply: u32, skip_move: ChessMove) {
        while ply as usize >= self.skip_moves.len() {
            self.skip_moves.push(None);
        }
        self.skip_moves[ply as usize] = Some(skip_move);
    }

    #[inline]
    pub fn reset_skip_move(&mut self, ply: u32) {
        self.skip_moves[ply as usize] = None;
    }

    #[inline]
    pub fn get_skip_move(&mut self, ply: u32) -> Option<ChessMove> {
        self.skip_moves.get(ply as usize).copied().unwrap_or(None)
    }

    #[inline]
    pub fn update_sel_depth(&mut self, ply: u32) {
        self.sel_depth = self.sel_depth.max(ply);
    }

    pub fn nodes(&mut self) -> &mut u32 {
        &mut self.nodes
    }

    pub fn trigger_abort(&mut self) {
        self.abort = true;
    }

    pub fn abort(&self) -> bool {
        self.abort
    }
}

pub struct AbRunner {
    shared_context: SharedContext,
    local_context: LocalContext,
    position: Position,
}

impl AbRunner {
    fn launch_searcher<SM: 'static + SearchMode + Send, Info: 'static + GuiInfo + Send>(
        &self,
        search_start: Instant,
        thread: u8,
    ) -> impl FnMut() -> (Option<ChessMove>, Evaluation, u32, u32) {
        let mut nodes = 0;

        let shared_context = self.shared_context.clone();
        let mut local_context = self.local_context.clone();
        let mut position = self.position.clone();
        let mut debugger = SM::new(self.position.board());
        let gui_info = Info::new();
        move || {
            let start_time = Instant::now();
            let mut best_move = None;
            let mut eval: Option<Evaluation> = None;
            let mut depth = 1_u32;
            'outer: loop {
                let mut fail_cnt = 0;
                local_context.window.reset();
                loop {
                    let (alpha, beta) = if eval.is_some()
                        && eval.unwrap().raw().abs() < 1000
                        && depth > 4
                        && fail_cnt < SEARCH_PARAMS.fail_cnt
                    {
                        local_context.window.get()
                    } else {
                        (Evaluation::min(), Evaluation::max())
                    };
                    local_context.nodes = 0;
                    let (make_move, score) = search::search::<Pv>(
                        &mut position,
                        &mut local_context,
                        &shared_context,
                        0,
                        depth,
                        alpha,
                        beta,
                    );
                    nodes += local_context.nodes;
                    if depth > 1 && shared_context.abort_absolute(depth, nodes) {
                        break 'outer;
                    }
                    local_context.window.set(score);
                    local_context.eval = score;

                    shared_context.time_manager.deepen(
                        thread,
                        depth,
                        nodes,
                        local_context.eval,
                        best_move.unwrap_or(make_move.unwrap()),
                        search_start.elapsed(),
                    );
                    if (score > alpha && score < beta) || score.is_mate() {
                        best_move = make_move;
                        eval = Some(score);
                        break;
                    } else {
                        fail_cnt += 1;
                        if score <= alpha {
                            local_context.window.fail_low();
                        } else {
                            local_context.window.fail_high();
                        }
                    }
                }
                debugger.push(SearchStats::new(
                    start_time.elapsed().as_millis(),
                    depth,
                    eval,
                    best_move,
                ));
                if let Some(eval) = eval {
                    if let Some(best_move) = best_move {
                        let best_move = best_move;
                        let mut pv = vec![best_move];
                        position.make_move(best_move);
                        while let Some(analysis) = shared_context.t_table.get(position.board()) {
                            pv.push(analysis.table_move());
                            position.make_move(analysis.table_move());
                            if pv.len() > depth as usize {
                                break;
                            }
                        }
                        for _ in 0..pv.len() {
                            position.unmake_move()
                        }
                        gui_info.print_info(
                            local_context.sel_depth,
                            depth,
                            eval,
                            start_time.elapsed(),
                            nodes,
                            &pv,
                        );
                    }
                }
                depth += 1;
            }
            if let Some(evaluation) = eval {
                debugger.complete();
                (best_move, evaluation, depth, nodes)
            } else {
                panic!("# Search function has failed to evaluate the position");
            }
        }
    }

    pub fn new(board: Board, time_manager: Arc<TimeManager>) -> Self {
        let mut position = Position::new(board);
        Self {
            shared_context: SharedContext {
                time_manager,
                t_table: Arc::new(TranspositionTable::new(2_usize.pow(20))),
                lmr_lookup: Arc::new(LookUp2d::new(|depth, mv| {
                    if depth == 0 || mv == 0 {
                        0
                    } else {
                        (LMR_BASE + (depth as f32).ln() * (mv as f32).ln() / LMR_DIV) as u32
                    }
                })),
                lmp_lookup: Arc::new(LookUp2d::new(|depth, improving| {
                    let mut x = LMP_OFFSET + depth as f32 * depth as f32 * LMP_FACTOR;
                    if improving == 0 {
                        x /= IMPROVING_DIVISOR;
                    }
                    x as usize
                })),
                start: Instant::now(),
            },
            local_context: LocalContext {
                window: Window::new(WINDOW_START, WINDOW_FACTOR, WINDOW_DIVISOR, WINDOW_ADD),
                h_table: HistoryTable::new(),
                ch_table: HistoryTable::new(),
                cm_table: CounterMoveTable::new(),
                cm_hist: DoubleMoveHistory::new(),
                fu_hist: DoubleMoveHistory::new(),
                killer_moves: vec![],
                threat_moves: vec![],
                tt_hits: 0,
                tt_misses: 0,
                eval: position.get_eval(),
                eval_stack: vec![],
                skip_moves: vec![],
                variation: vec![],
                sel_depth: 0,
                nodes: 0,
                abort: false,
            },
            position,
        }
    }

    pub fn search<SM: 'static + SearchMode + Send, Info: 'static + GuiInfo + Send>(
        &mut self,
        threads: u8,
    ) -> (ChessMove, Evaluation, u32, u32) {
        let mut join_handlers = vec![];
        let search_start = Instant::now();
        self.shared_context.start = Instant::now();
        //TODO: Research the effects of different depths
        for i in 1..threads {
            join_handlers.push(std::thread::spawn(
                self.launch_searcher::<SM, NoInfo>(search_start, i),
            ));
        }
        let (final_move, final_eval, max_depth, mut node_count) =
            self.launch_searcher::<SM, Info>(search_start, 0)();
        for join_handler in join_handlers {
            let (_, _, _, nodes) = join_handler.join().unwrap();
            node_count += nodes;
        }
        if final_move.is_none() {
            panic!("# All move generation has failed");
        }
        (final_move.unwrap(), final_eval, max_depth, node_count)
    }

    pub fn hash(&mut self, hash_mb: usize) {
        let entry_count = hash_mb * 65536;
        self.shared_context.t_table = Arc::new(TranspositionTable::new(entry_count));
    }

    pub fn raw_eval(&mut self) -> Evaluation {
        self.position.get_eval()
    }

    pub fn new_game(&self) {
        self.shared_context.t_table.clean();
    }

    pub fn set_board(&mut self, board: Board) {
        self.position = Position::new(board);
    }

    pub fn make_move(&mut self, make_move: ChessMove) {
        self.position.make_move(make_move);
    }

    #[cfg(feature = "data")]
    pub fn get_position(&self) -> &Position {
        &self.position
    }

    pub fn get_board(&self) -> &Board {
        self.position.board()
    }
}
