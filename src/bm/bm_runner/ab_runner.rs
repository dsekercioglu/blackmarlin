use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use cozy_chess::{Board, Move};

use crate::bm::bm_eval::eval::Evaluation;
use crate::bm::bm_runner::config::{GuiInfo, NoInfo, SearchMode, SearchStats};
use crate::bm::bm_search::move_entry::MoveEntry;
use crate::bm::bm_search::search;
use crate::bm::bm_search::search::Pv;
use crate::bm::bm_util::h_table::{CounterMoveTable, DoubleMoveHistory, HistoryTable};
use crate::bm::bm_util::lookup::LookUp2d;
use crate::bm::bm_util::position::Position;
use crate::bm::bm_util::t_table::TranspositionTable;
use crate::bm::bm_util::window::Window;
use crate::bm::uci;

use super::time::TimeManager;

pub const MAX_PLY: u32 = 128;

#[derive(Debug, Clone)]
pub struct NodeCounter {
    node_counters: Vec<Option<Arc<AtomicU64>>>,
}

impl NodeCounter {
    fn initialize_node_counters(&mut self, threads: usize) {
        self.node_counters = vec![None; threads];
    }

    fn add_node_counter(&mut self, thread: usize, node_counter: Arc<AtomicU64>) {
        self.node_counters[thread] = Some(node_counter);
    }

    fn get_node_count(&self) -> u64 {
        let mut total_nodes = 0;
        for nodes in self.node_counters.iter() {
            if let Some(nodes) = nodes {
                total_nodes += nodes.load(Ordering::Relaxed);
            }
        }
        total_nodes
    }
}

#[derive(Debug)]
pub struct Nodes(Arc<AtomicU64>);

impl Clone for Nodes {
    fn clone(&self) -> Self {
        Self(Arc::new(AtomicU64::new(0)))
    }
}

type LmrLookup = LookUp2d<u32, 32, 64>;
type LmpLookup = LookUp2d<usize, 16, 2>;

#[derive(Debug, Clone)]
pub struct SharedContext {
    start: Instant,
    time_manager: Arc<TimeManager>,

    t_table: Arc<TranspositionTable>,
    lmr_lookup: Arc<LmrLookup>,
    lmp_lookup: Arc<LmpLookup>,
}

#[derive(Debug, Clone)]
pub struct SearchStack {
    pub eval: Evaluation,
    pub skip_move: Option<Move>,
    pub move_played: Option<Move>,
    pub pv: [Option<Move>; MAX_PLY as usize + 1],
    pub pv_len: usize,
}

impl SearchStack {
    pub fn update_pv(&mut self, best_move: Move, child_pv: &[Option<Move>]) {
        self.pv[0] = Some(best_move);
        for (pv, &child) in self.pv[1..].iter_mut().zip(child_pv) {
            *pv = child;
        }
        self.pv_len = child_pv.len() + 1;
    }
}

#[derive(Debug, Clone)]
pub struct LocalContext {
    window: Window,
    tt_hits: u32,
    tt_misses: u32,
    eval: Evaluation,
    search_stack: Vec<SearchStack>,
    sel_depth: u32,
    h_table: HistoryTable,
    ch_table: HistoryTable,
    cm_table: CounterMoveTable,
    cm_hist: DoubleMoveHistory,
    killer_moves: Vec<MoveEntry<2>>,
    nodes: Nodes,
    abort: bool,
}

impl SharedContext {
    #[inline]
    pub fn abort_search(&self) -> bool {
        self.time_manager.abort_search(self.start)
    }

    #[inline]
    pub fn abort_deepening(&self, depth: u32, nodes: u64) -> bool {
        self.time_manager.abort_deepening(self.start, depth, nodes)
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
    pub fn get_k_table(&mut self) -> &mut Vec<MoveEntry<2>> {
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
    pub fn search_stack(&self) -> &[SearchStack] {
        &self.search_stack
    }

    #[inline]
    pub fn search_stack_mut(&mut self) -> &mut [SearchStack] {
        &mut self.search_stack
    }

    #[inline]
    pub fn update_sel_depth(&mut self, ply: u32) {
        self.sel_depth = self.sel_depth.max(ply);
    }

    pub fn reset_nodes(&self) {
        self.nodes.0.store(0, Ordering::Relaxed);
    }

    pub fn increment_nodes(&self) {
        self.nodes.0.fetch_add(1, Ordering::Relaxed);
    }

    pub fn nodes(&self) -> u64 {
        self.nodes.0.load(Ordering::Relaxed)
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
    node_counter: NodeCounter,
    position: Position,
    chess960: bool,
}

impl AbRunner {
    fn launch_searcher<SM: 'static + SearchMode + Send, Info: 'static + GuiInfo + Send>(
        &mut self,
        search_start: Instant,
        thread: u8,
        chess960: bool,
    ) -> impl FnMut() -> (Option<Move>, Evaluation, u32, u64) {
        let main_thread = thread == 0;
        let shared_context = self.shared_context.clone();
        let mut local_context = self.local_context.clone();
        self.node_counter
            .add_node_counter(thread as usize, local_context.nodes.0.clone());
        let node_counter = if main_thread {
            Some(self.node_counter.clone())
        } else {
            None
        };
        let mut position = self.position.clone();
        let mut debugger = SM::new(self.position.board());
        let gui_info = Info::new();
        move || {
            let mut nodes = 0;
            local_context.reset_nodes();
            let start_time = Instant::now();
            let mut best_move = None;
            let mut eval: Option<Evaluation> = None;
            let mut depth = 1_u32;
            let mut abort = false;
            'outer: loop {
                let mut fail_cnt = 0;
                local_context.window.reset();
                loop {
                    if abort {
                        break 'outer;
                    }
                    let (alpha, beta) = if eval.is_some()
                        && eval.unwrap().raw().abs() < 1000
                        && depth > 4
                        && fail_cnt < 10
                    {
                        local_context.window.get()
                    } else {
                        (Evaluation::min(), Evaluation::max())
                    };
                    local_context.sel_depth = 0;
                    let score = search::search::<Pv>(
                        &mut position,
                        &mut local_context,
                        &shared_context,
                        0,
                        depth,
                        alpha,
                        beta,
                    );
                    nodes = local_context.nodes();
                    if depth > 1 && local_context.abort() {
                        break 'outer;
                    }
                    local_context.window.set(score);
                    local_context.eval = score;

                    shared_context.time_manager.deepen(
                        thread,
                        depth,
                        nodes,
                        local_context.eval,
                        local_context.search_stack[0].pv[0].unwrap(),
                        search_start.elapsed(),
                    );
                    abort = shared_context.abort_deepening(depth, nodes);
                    if (score > alpha && score < beta) || score.is_mate() {
                        best_move = local_context.search_stack[0].pv[0];
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
                if main_thread {
                    debugger.push(SearchStats::new(
                        start_time.elapsed().as_millis(),
                        depth,
                        eval,
                        best_move,
                    ));

                    let mut pv = vec![];
                    let root_stack = &local_context.search_stack[0];
                    for make_move in &root_stack.pv[..root_stack.pv_len] {
                        if let Some(make_move) = *make_move {
                            let mut uci_move = make_move;
                            uci::convert_move_to_uci(&mut uci_move, position.board(), chess960);
                            position.make_move(make_move);
                            pv.push(uci_move);
                            if pv.len() > depth as usize {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                    for _ in 0..pv.len() {
                        position.unmake_move()
                    }
                    let total_nodes = node_counter.as_ref().unwrap().get_node_count();
                    gui_info.print_info(
                        local_context.sel_depth,
                        depth,
                        eval.unwrap(),
                        start_time.elapsed(),
                        total_nodes,
                        &pv,
                    );
                }

                depth += 1;
                if depth > 1 && shared_context.abort_deepening(depth, nodes) {
                    break 'outer;
                }
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
            node_counter: NodeCounter {
                node_counters: vec![],
            },
            shared_context: SharedContext {
                time_manager,
                t_table: Arc::new(TranspositionTable::new(2_usize.pow(20))),
                lmr_lookup: Arc::new(LookUp2d::new(|depth, mv| {
                    if depth == 0 || mv == 0 {
                        0
                    } else {
                        (0.75 + (depth as f32).ln() * (mv as f32).ln() / 1.25) as u32
                    }
                })),
                lmp_lookup: Arc::new(LookUp2d::new(|depth, improving| {
                    let mut x = 3.0 + depth as f32 * depth as f32;
                    if improving == 0 {
                        x /= 1.5;
                    }
                    x as usize
                })),
                start: Instant::now(),
            },
            local_context: LocalContext {
                window: Window::new(25, 1, 4, 5),
                tt_hits: 0,
                tt_misses: 0,
                eval: position.get_eval(),
                search_stack: vec![
                    SearchStack {
                        eval: Evaluation::new(0),
                        skip_move: None,
                        move_played: None,
                        pv: [None; MAX_PLY as usize + 1],
                        pv_len: 0,
                    };
                    MAX_PLY as usize + 1
                ],
                sel_depth: 0,
                h_table: HistoryTable::new(),
                ch_table: HistoryTable::new(),
                cm_table: CounterMoveTable::new(),
                cm_hist: DoubleMoveHistory::new(),
                killer_moves: vec![],
                nodes: Nodes(Arc::new(AtomicU64::new(0))),
                abort: false,
            },
            position,
            chess960: false,
        }
    }

    pub fn search<SM: 'static + SearchMode + Send, Info: 'static + GuiInfo + Send>(
        &mut self,
        threads: u8,
    ) -> (Move, Evaluation, u32, u64) {
        let mut join_handlers = vec![];
        let search_start = Instant::now();
        self.shared_context.start = Instant::now();
        self.node_counter.initialize_node_counters(threads as usize);
        //TODO: Research the effects of different depths
        for i in 1..threads {
            join_handlers.push(std::thread::spawn(self.launch_searcher::<SM, NoInfo>(
                search_start,
                i,
                self.chess960,
            )));
        }
        let (final_move, final_eval, max_depth, mut node_count) =
            self.launch_searcher::<SM, Info>(search_start, 0, self.chess960)();
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

    pub fn make_move(&mut self, make_move: Move) {
        self.position.make_move(make_move);
    }

    #[cfg(feature = "data")]
    pub fn get_position(&self) -> &Position {
        &self.position
    }

    pub fn get_board(&self) -> &Board {
        self.position.board()
    }

    pub fn set_chess960(&mut self, chess960: bool) {
        self.chess960 = chess960;
    }
}
