use crate::bm::bm_eval::eval::Evaluation;
use crate::bm::bm_eval::evaluator::StdEvaluator;
use chess::{Board, ChessMove, MoveGen};
use std::fmt::Debug;
use std::sync::atomic::{AtomicBool, AtomicI16, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub trait TimeManager: Debug + Send + Sync {
    fn deepen(
        &self,
        thread: u8,
        depth: u32,
        nodes: u32,
        eval: Evaluation,
        best_move: ChessMove,
        delta_time: Duration,
    );

    fn initiate(&self, time_left: Duration, board: &Board);

    fn abort(&self, start: Instant, depth: u32, nodes: u32) -> bool;

    fn clear(&self);
}

#[derive(Debug, Copy, Clone)]
pub struct Percentage {
    numerator: u32,
    denominator: u32,
}

#[derive(Debug)]
pub struct ConstDepth {
    depth: AtomicU32,
}

impl ConstDepth {
    pub fn new(depth: u32) -> Self {
        Self {
            depth: AtomicU32::new(depth),
        }
    }

    pub fn set_depth(&self, depth: u32) {
        self.depth.store(depth, Ordering::SeqCst);
    }
}

impl TimeManager for ConstDepth {
    fn deepen(&self, _: u8, _: u32, _: u32, _: Evaluation, _: ChessMove, _: Duration) {}

    fn initiate(&self, _: Duration, _: &Board) {}

    fn abort(&self, _: Instant, depth: u32, _: u32) -> bool {
        depth >= self.depth.load(Ordering::SeqCst)
    }

    fn clear(&self) {}
}

#[derive(Debug)]
pub struct ConstTime {
    start: Instant,
    target_duration: AtomicU32,
    std_target_duration: AtomicU32,
}

impl ConstTime {
    pub fn new(target_duration: Duration) -> Self {
        Self {
            start: Instant::now(),
            target_duration: AtomicU32::new(target_duration.as_millis() as u32),
            std_target_duration: AtomicU32::new(target_duration.as_millis() as u32),
        }
    }

    pub fn set_duration(&self, duration: Duration) {
        self.target_duration
            .store(duration.as_millis() as u32, Ordering::SeqCst);
    }
}

impl TimeManager for ConstTime {
    fn deepen(&self, _: u8, _: u32, _: u32, _: Evaluation, _: ChessMove, _: Duration) {}

    fn initiate(&self, _: Duration, _: &Board) {}

    fn abort(&self, start: Instant, _: u32, _: u32) -> bool {
        self.target_duration.load(Ordering::SeqCst) < start.elapsed().as_millis() as u32
    }

    fn clear(&self) {
        self.target_duration.store(
            self.std_target_duration.load(Ordering::SeqCst),
            Ordering::SeqCst,
        );
    }
}

const EXPECTED_MOVES: u32 = 40;
const MIN_MOVES: u32 = 40;

#[derive(Debug)]
pub struct MainTimeManager {
    start: Instant,
    expected_moves: AtomicU32,
    last_eval: AtomicI16,
    max_duration: AtomicU32,
    normal_duration: AtomicU32,
    target_duration: AtomicU32,
    prev_move: Mutex<Option<ChessMove>>,
    board: Mutex<Board>,
}

impl MainTimeManager {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            expected_moves: AtomicU32::new(EXPECTED_MOVES),
            last_eval: AtomicI16::new(0),
            max_duration: AtomicU32::new(0),
            normal_duration: AtomicU32::new(0),
            target_duration: AtomicU32::new(0),
            prev_move: Mutex::new(None),
            board: Mutex::new(Board::default()),
        }
    }
}

impl TimeManager for MainTimeManager {
    fn deepen(
        &self,
        _: u8,
        depth: u32,
        _: u32,
        eval: Evaluation,
        current_move: ChessMove,
        _: Duration,
    ) {
        if depth <= 4 {
            return;
        }

        let board = *self.board.lock().unwrap();
        let see = StdEvaluator::see(board, current_move) as f32;
        let time_multiplier = 2_f32.powf(see.min(0.0) / -600.0);

        let current_eval = eval.raw();
        let last_eval = self.last_eval.load(Ordering::SeqCst);
        let mut time = (self.normal_duration.load(Ordering::SeqCst) * 1000) as f32;

        let mut move_changed = false;
        if let Some(prev_move) = &mut *self.prev_move.lock().unwrap() {
            if *prev_move != current_move {
                move_changed = true;
            }
            *prev_move = current_move;
        }

        let bias = if move_changed {
            0.5
        } else {
            -0.2
        };
        time *= 1.25_f32.powf((current_eval - last_eval).abs().min(150) as f32 / 50.0 + bias);

        let time = time.min(self.max_duration.load(Ordering::SeqCst) as f32 * 1000.0);
        self.normal_duration
            .store((time * 0.001) as u32, Ordering::SeqCst);
        self.target_duration
            .store((time * time_multiplier * 0.001) as u32, Ordering::SeqCst);
        self.last_eval.store(current_eval, Ordering::SeqCst);
    }

    fn initiate(&self, time_left: Duration, board: &Board) {
        *self.board.lock().unwrap() = *board;
        let move_cnt = MoveGen::new_legal(board).into_iter().count();
        if move_cnt == 0 {
            self.target_duration.store(0, Ordering::SeqCst);
        } else {
            let default = time_left.as_millis() as u32 / self.expected_moves.load(Ordering::SeqCst);
            self.normal_duration.store(default, Ordering::SeqCst);
            self.target_duration.store(default, Ordering::SeqCst);
            self.max_duration
                .store(time_left.as_millis() as u32 * 1 / 3, Ordering::SeqCst);
        };
    }

    fn abort(&self, start: Instant, _: u32, _: u32) -> bool {
        self.target_duration.load(Ordering::SeqCst) < start.elapsed().as_millis() as u32
    }

    fn clear(&self) {
        self.expected_moves.fetch_sub(1, Ordering::SeqCst);
        self.expected_moves.fetch_max(MIN_MOVES, Ordering::SeqCst);
    }
}

#[derive(Debug)]
pub struct ManualAbort {
    abort: AtomicBool,
}

impl ManualAbort {
    pub fn new() -> Self {
        Self {
            abort: AtomicBool::new(false),
        }
    }

    pub fn quick_abort(&self) {
        self.abort.store(true, Ordering::SeqCst);
    }
}

impl TimeManager for ManualAbort {
    fn deepen(&self, _: u8, _: u32, _: u32, _: Evaluation, _: ChessMove, _: Duration) {}

    fn initiate(&self, _: Duration, _: &Board) {
        self.abort.store(false, Ordering::SeqCst);
    }

    fn abort(&self, _: Instant, _: u32, _: u32) -> bool {
        self.abort.load(Ordering::SeqCst)
    }

    fn clear(&self) {}
}

#[derive(Debug)]
pub struct CompoundTimeManager {
    managers: Box<[Arc<dyn TimeManager>]>,
    mode: AtomicUsize,
}

impl CompoundTimeManager {
    pub fn new(managers: Box<[Arc<dyn TimeManager>]>, initial_mode: usize) -> Self {
        Self {
            managers,
            mode: AtomicUsize::new(initial_mode),
        }
    }

    pub fn set_mode(&self, mode: usize) {
        self.mode.store(mode, Ordering::SeqCst);
    }
}

impl TimeManager for CompoundTimeManager {
    fn deepen(
        &self,
        thread: u8,
        depth: u32,
        nodes: u32,
        eval: Evaluation,
        best_move: ChessMove,
        delta_time: Duration,
    ) {
        self.managers[self.mode.load(Ordering::SeqCst)]
            .deepen(thread, depth, nodes, eval, best_move, delta_time);
    }

    fn initiate(&self, time_left: Duration, board: &Board) {
        self.managers[self.mode.load(Ordering::SeqCst)].initiate(time_left, board);
    }

    fn abort(&self, start: Instant, depth: u32, nodes: u32) -> bool {
        self.managers[self.mode.load(Ordering::SeqCst)].abort(start, depth, nodes)
    }

    fn clear(&self) {
        self.managers.iter().for_each(|manager| manager.clear());
    }
}

#[derive(Debug)]
pub struct Diagnostics<Inner: TimeManager> {
    manager: Arc<Inner>,
    data: Mutex<Vec<(u32, u32)>>,
}

impl<Inner: TimeManager> Diagnostics<Inner> {
    pub fn new(manager: Arc<Inner>) -> Diagnostics<Inner> {
        Self {
            manager,
            data: Mutex::new(vec![]),
        }
    }

    pub fn get_data(&self) -> &Mutex<Vec<(u32, u32)>> {
        &self.data
    }
}

impl<Inner: TimeManager> TimeManager for Diagnostics<Inner> {
    fn deepen(
        &self,
        thread: u8,
        depth: u32,
        nodes: u32,
        eval: Evaluation,
        best_move: ChessMove,
        delta_time: Duration,
    ) {
        self.manager
            .deepen(thread, depth, nodes, eval, best_move, delta_time);
        self.data.lock().unwrap().push((nodes, depth));
    }

    fn initiate(&self, time_left: Duration, board: &Board) {
        self.manager.initiate(time_left, board);
    }

    fn abort(&self, start: Instant, depth: u32, nodes: u32) -> bool {
        self.manager.abort(start, depth, nodes)
    }

    fn clear(&self) {
        self.manager.clear();
    }
}
