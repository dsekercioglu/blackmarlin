use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use cozy_chess::{Board, File, Move, Piece, Square};

use crate::bm::bm_runner::ab_runner::AbRunner;
use crate::bm::bm_runner::config::{NoInfo, Run, UciInfo};

use crate::bm::bm_runner::time::{TimeManagementInfo, TimeManager};

use super::bm_search::search::{FP, HLMR, HP, QA, QB, RFP, RMP, SEEFP, SEEFPC};

const VERSION: &str = "6.0";

const POSITIONS: &[&str] = &[
    "r3k2r/2pb1ppp/2pp1q2/p7/1nP1B3/1P2P3/P2N1PPP/R2QK2R w KQkq a6 0 14",
    "4rrk1/2p1b1p1/p1p3q1/4p3/2P2n1p/1P1NR2P/PB3PP1/3R1QK1 b - - 2 24",
    "r3qbrk/6p1/2b2pPp/p3pP1Q/PpPpP2P/3P1B2/2PB3K/R5R1 w - - 16 42",
    "6k1/1R3p2/6p1/2Bp3p/3P2q1/P7/1P2rQ1K/5R2 b - - 4 44",
    "8/8/1p2k1p1/3p3p/1p1P1P1P/1P2PK2/8/8 w - - 3 54",
    "7r/2p3k1/1p1p1qp1/1P1Bp3/p1P2r1P/P7/4R3/Q4RK1 w - - 0 36",
    "r1bq1rk1/pp2b1pp/n1pp1n2/3P1p2/2P1p3/2N1P2N/PP2BPPP/R1BQ1RK1 b - - 2 10",
    "3r3k/2r4p/1p1b3q/p4P2/P2Pp3/1B2P3/3BQ1RP/6K1 w - - 3 87",
    "2r4r/1p4k1/1Pnp4/3Qb1pq/8/4BpPp/5P2/2RR1BK1 w - - 0 42",
    "4q1bk/6b1/7p/p1p4p/PNPpP2P/KN4P1/3Q4/4R3 b - - 0 37",
    "2q3r1/1r2pk2/pp3pp1/2pP3p/P1Pb1BbP/1P4Q1/R3NPP1/4R1K1 w - - 2 34",
    "1r2r2k/1b4q1/pp5p/2pPp1p1/P3Pn2/1P1B1Q1P/2R3P1/4BR1K b - - 1 37",
    "r3kbbr/pp1n1p1P/3ppnp1/q5N1/1P1pP3/P1N1B3/2P1QP2/R3KB1R b KQkq b3 0 17",
    "8/6pk/2b1Rp2/3r4/1R1B2PP/P5K1/8/2r5 b - - 16 42",
    "1r4k1/4ppb1/2n1b1qp/pB4p1/1n1BP1P1/7P/2PNQPK1/3RN3 w - - 8 29",
    "8/p2B4/PkP5/4p1pK/4Pb1p/5P2/8/8 w - - 29 68",
    "3r4/ppq1ppkp/4bnp1/2pN4/2P1P3/1P4P1/PQ3PBP/R4K2 b - - 2 20",
    "5rr1/4n2k/4q2P/P1P2n2/3B1p2/4pP2/2N1P3/1RR1K2Q w - - 1 49",
    "1r5k/2pq2p1/3p3p/p1pP4/4QP2/PP1R3P/6PK/8 w - - 1 51",
    "q5k1/5ppp/1r3bn1/1B6/P1N2P2/BQ2P1P1/5K1P/8 b - - 2 34",
    "r1b2k1r/5n2/p4q2/1ppn1Pp1/3pp1p1/NP2P3/P1PPBK2/1RQN2R1 w - - 0 22",
    "r1bqk2r/pppp1ppp/5n2/4b3/4P3/P1N5/1PP2PPP/R1BQKB1R w KQkq - 0 5",
    "r1bqr1k1/pp1p1ppp/2p5/8/3N1Q2/P2BB3/1PP2PPP/R3K2n b Q - 1 12",
    "r1bq2k1/p4r1p/1pp2pp1/3p4/1P1B3Q/P2B1N2/2P3PP/4R1K1 b - - 2 19",
    "r4qk1/6r1/1p4p1/2ppBbN1/1p5Q/P7/2P3PP/5RK1 w - - 2 25",
    "r7/6k1/1p6/2pp1p2/7Q/8/p1P2K1P/8 w - - 0 32",
    "r3k2r/ppp1pp1p/2nqb1pn/3p4/4P3/2PP4/PP1NBPPP/R2QK1NR w KQkq - 1 5",
    "3r1rk1/1pp1pn1p/p1n1q1p1/3p4/Q3P3/2P5/PP1NBPPP/4RRK1 w - - 0 12",
    "5rk1/1pp1pn1p/p3Brp1/8/1n6/5N2/PP3PPP/2R2RK1 w - - 2 20",
    "8/1p2pk1p/p1p1r1p1/3n4/8/5R2/PP3PPP/4R1K1 b - - 3 27",
    "8/4pk2/1p1r2p1/p1p4p/Pn5P/3R4/1P3PP1/4RK2 w - - 1 33",
    "8/5k2/1pnrp1p1/p1p4p/P6P/4R1PK/1P3P2/4R3 b - - 1 38",
    "8/8/1p1kp1p1/p1pr1n1p/P6P/1R4P1/1P3PK1/1R6 b - - 15 45",
    "8/8/1p1k2p1/p1prp2p/P2n3P/6P1/1P1R1PK1/4R3 b - - 5 49",
    "8/8/1p4p1/p1p2k1p/P2npP1P/4K1P1/1P6/3R4 w - - 6 54",
    "8/8/1p4p1/p1p2k1p/P2n1P1P/4K1P1/1P6/6R1 b - - 6 59",
    "8/5k2/1p4p1/p1pK3p/P2n1P1P/6P1/1P6/4R3 b - - 14 63",
    "8/1R6/1p1K1kp1/p6p/P1p2P1P/6P1/1Pn5/8 w - - 0 67",
    "1rb1rn1k/p3q1bp/2p3p1/2p1p3/2P1P2N/PP1RQNP1/1B3P2/4R1K1 b - - 4 23",
    "4rrk1/pp1n1pp1/q5p1/P1pP4/2n3P1/7P/1P3PB1/R1BQ1RK1 w - - 3 22",
    "r2qr1k1/pb1nbppp/1pn1p3/2ppP3/3P4/2PB1NN1/PP3PPP/R1BQR1K1 w - - 4 12",
    "2r2k2/8/4P1R1/1p6/8/P4K1N/7b/2B5 b - - 0 55",
    "6k1/5pp1/8/2bKP2P/2P5/p4PNb/B7/8 b - - 1 44",
    "2rqr1k1/1p3p1p/p2p2p1/P1nPb3/2B1P3/5P2/1PQ2NPP/R1R4K w - - 3 25",
    "r1b2rk1/p1q1ppbp/6p1/2Q5/8/4BP2/PPP3PP/2KR1B1R b - - 2 14",
    "6r1/5k2/p1b1r2p/1pB1p1p1/1Pp3PP/2P1R1K1/2P2P2/3R4 w - - 1 36",
    "rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2",
    "2rr2k1/1p4bp/p1q1p1p1/4Pp1n/2PB4/1PN3P1/P3Q2P/2RR2K1 w - f6 0 20",
    "3br1k1/p1pn3p/1p3n2/5pNq/2P1p3/1PN3PP/P2Q1PB1/4R1K1 w - - 0 23",
    "2r2b2/5p2/5k2/p1r1pP2/P2pB3/1P3P2/K1P3R1/7R w - - 23 93",
];

pub struct UciAdapter {
    bm_runner: Arc<Mutex<AbRunner>>,
    time_manager: Arc<TimeManager>,
    analysis: Option<JoinHandle<()>>,
    forced: bool,
    threads: u8,
    chess960: bool,
}

impl UciAdapter {
    pub fn new() -> Self {
        let time_manager = Arc::new(TimeManager::new());
        let bm_runner = Arc::new(Mutex::new(AbRunner::new(
            Board::default(),
            time_manager.clone(),
        )));
        Self {
            bm_runner,
            threads: 1,
            forced: false,
            analysis: None,
            time_manager,
            chess960: false,
        }
    }

    pub fn input(&mut self, input: String) -> bool {
        let name = "Black Marlin".to_string();
        let command = UciCommand::new(&input, self.chess960);
        match command {
            UciCommand::Uci => {
                println!("id name {} {}", name, VERSION);
                println!("id author Doruk S.");
                println!("option name Hash type spin default 16 min 1 max 65536");
                println!("option name Threads type spin default 1 min 1 max 255");
                println!("option name UCI_Chess960 type check default false");
                macro_rules! param {
                    ($name: expr, $default: expr, $min: expr, $max: expr) => {
                        println!(
                            "option name {} type spin default {} min {} max {}",
                            $name, $default, $min, $max
                        );
                    };
                }
                param!("RFP", 50, 40, 100);
                param!("FP", 100, 50, 200);
                param!("SEEFP", 100, 50, 200);
                param!("SEEFPC", 100, 50, 200);

                param!("RMP", 250, 50, 500);
                param!("QB", 200, 100, 300);
                param!("QA", 200, 100, 300);

                param!("HP", 64, 32, 96);
                param!("HLMR", 80, 40, 120);

                println!("uciok");
            }
            UciCommand::IsReady => println!("readyok"),
            UciCommand::Move(make_move) => {
                let runner = &mut *self.bm_runner.lock().unwrap();
                runner.make_move(make_move);
            }
            UciCommand::Empty => {}
            UciCommand::Stop => {
                self.time_manager.abort_now();
                self.exit();
            }
            UciCommand::Quit => {
                return false;
            }
            UciCommand::Eval => {
                let runner = &mut *self.bm_runner.lock().unwrap();

                println!("eval    : {}", runner.raw_eval().raw());
            }
            UciCommand::Go(commands) => self.go(commands),
            UciCommand::NewGame => {
                let runner = &mut *self.bm_runner.lock().unwrap();
                runner.new_game();
                runner.set_board(Board::default());
            }
            UciCommand::Position(position, moves) => {
                let runner = &mut *self.bm_runner.lock().unwrap();
                runner.set_board(position);
                for mut make_move in moves {
                    convert_move(&mut make_move, runner.get_board(), self.chess960);
                    runner.make_move(make_move);
                }
            }
            UciCommand::SetOption(name, value) => {
                let name: &str = &name;
                self.time_manager.abort_now();
                match name {
                    "Hash" => {
                        self.bm_runner
                            .lock()
                            .unwrap()
                            .hash(value.parse::<usize>().unwrap());
                    }
                    "Threads" => {
                        self.threads = value.parse::<u8>().unwrap();
                    }
                    "UCI_Chess960" => {
                        self.chess960 = value.to_lowercase().parse::<bool>().unwrap();
                        self.bm_runner.lock().unwrap().set_chess960(self.chess960);
                    }
                    /*
                    pub static mut RFP: i16 = 50;
                    pub static mut FP: i16 = 100;
                    pub static mut SEEFP: i16 = 100;
                    pub static mut SEEFPC: i16 = 100;

                    pub static mut RMP: i16 = 250;
                    pub static mut QB: i16 = 200;
                    pub static mut QA: i16 = 200;

                    pub static mut HP: i32 = 64;
                    pub static mut HLMR: i16 = 80;
                    */
                    "RFP" => unsafe { RFP = value.parse::<i16>().unwrap() },
                    "FP" => unsafe { FP = value.parse::<i16>().unwrap() },
                    "SEEFP" => unsafe { SEEFP = value.parse::<i16>().unwrap() },
                    "SEEFPC" => unsafe { SEEFPC = value.parse::<i16>().unwrap() },
                    "RMP" => unsafe { RMP = value.parse::<i16>().unwrap() },
                    "QB" => unsafe { QB = value.parse::<i16>().unwrap() },
                    "QA" => unsafe { QA = value.parse::<i16>().unwrap() },
                    "HP" => unsafe { HP = value.parse::<i32>().unwrap() },
                    "HLMR" => unsafe { HLMR = value.parse::<i16>().unwrap() },
                    _ => {}
                }
            }
            UciCommand::Bench => {
                self.exit();

                let mut bench_data = vec![];

                let bm_runner = &mut *self.bm_runner.lock().unwrap();
                let mut sum_node_cnt = 0;
                let mut sum_time = Duration::from_nanos(0);
                for position in POSITIONS {
                    let board = cozy_chess::Board::from_str(position).unwrap();
                    bm_runner.new_game();
                    bm_runner.set_board(board.clone());
                    let options = [TimeManagementInfo::MaxDepth(12)];
                    let start = Instant::now();

                    self.time_manager.initiate(&board, &options);
                    let (make_move, eval, _, node_cnt) =
                        bm_runner.search::<Run, NoInfo>(self.threads);
                    self.time_manager.clear();
                    let elapsed = start.elapsed();
                    bench_data.push((
                        eval.raw(),
                        make_move,
                        node_cnt,
                        (node_cnt as f32 / elapsed.as_secs_f32()) as u32,
                    ));
                    sum_time += elapsed;
                    sum_node_cnt += node_cnt;
                }
                let mut buffer = String::new();
                let mut line_len = 0;
                for (index, (cp, mv, nodes, nps)) in bench_data.into_iter().enumerate() {
                    let line = &format!(
                        "[#{:>3}]{:>8} cp  Best: {:>8} {:>8} nodes {:>8} nps\n",
                        index + 1,
                        cp,
                        mv,
                        nodes,
                        nps
                    );
                    buffer += line;
                    line_len = line.len();
                }
                buffer += &("=".repeat(line_len) + "\n");
                buffer += &format!(
                    "OVERALL {:>30} nodes {:>8} nps",
                    sum_node_cnt,
                    (sum_node_cnt as f32 / sum_time.as_secs_f32()) as u32
                );
                println!("{}", buffer);
            }
            UciCommand::Static => {
                let runner = &mut *self.bm_runner.lock().unwrap();
                println!("{}", runner.raw_eval().raw());
            }
        }
        true
    }

    fn go(&mut self, commands: Vec<TimeManagementInfo>) {
        self.exit();
        self.forced = false;
        self.time_manager
            .initiate(self.bm_runner.lock().unwrap().get_board(), &commands);
        let bm_runner = self.bm_runner.clone();
        let threads = self.threads;
        let chess960 = self.chess960;
        self.analysis = Some(std::thread::spawn(move || {
            let mut bm_runner = bm_runner.lock().unwrap();
            let (mut best_move, _, _, _) = bm_runner.search::<Run, UciInfo>(threads);
            convert_move_to_uci(&mut best_move, bm_runner.get_board(), chess960);
            println!("bestmove {}", best_move);
        }));
    }

    fn exit(&mut self) {
        if let Some(analysis) = self.analysis.take() {
            analysis.join().unwrap();
        }
    }
}

pub fn convert_move_to_uci(make_move: &mut Move, board: &Board, chess960: bool) {
    if !chess960 && board.color_on(make_move.from) == board.color_on(make_move.to) {
        let rights = board.castle_rights(board.side_to_move());
        let file = if Some(make_move.to.file()) == rights.short {
            File::G
        } else {
            File::C
        };
        make_move.to = Square::new(file, make_move.to.rank());
    }
}

fn convert_move(make_move: &mut Move, board: &Board, chess960: bool) {
    let convert_castle = !chess960
        && board.piece_on(make_move.from) == Some(Piece::King)
        && make_move.from.file() == File::E
        && matches!(make_move.to.file(), File::C | File::G);
    if convert_castle {
        let file = if make_move.to.file() == File::C {
            File::A
        } else {
            File::H
        };
        make_move.to = Square::new(file, make_move.to.rank());
    }
}

enum UciCommand {
    Uci,
    IsReady,
    NewGame,
    Position(Board, Vec<Move>),
    Go(Vec<TimeManagementInfo>),
    SetOption(String, String),
    Move(Move),
    Bench,
    Empty,
    Stop,
    Quit,
    Eval,
    Static,
}

impl UciCommand {
    fn new(input: &str, chess960: bool) -> Self {
        let input_move = cozy_chess::Move::from_str(input);
        if let Ok(m) = input_move {
            return UciCommand::Move(m);
        }
        let mut split = input.split_ascii_whitespace();
        let token = match split.next() {
            None => {
                return UciCommand::Empty;
            }
            Some(string) => string,
        };
        match token {
            "uci" => UciCommand::Uci,
            "ucinewgame" => UciCommand::NewGame,
            "position" => {
                let mut board = "".to_string();
                let mut chess_board = None;
                let split = split.into_iter().collect::<Vec<_>>();

                let mut board_end = 0;
                for (index, token) in split.iter().enumerate() {
                    let token = token.trim();
                    if token == "startpos" {
                        chess_board = Some(Board::default());
                        board_end = index + 1;
                        break;
                    } else if token != "fen" {
                        if token == "moves" {
                            if let Ok(board) = Board::from_fen(board.trim(), chess960) {
                                chess_board = Some(board);
                                board_end = index;
                                break;
                            }
                        }
                        board += token;
                        board += " ";
                    }
                }
                if chess_board.is_none() {
                    chess_board = Some(Board::from_fen(board.trim(), chess960).unwrap());
                }
                let mut moves = vec![];
                if board_end < split.len() && split[board_end] == "moves" {
                    for token in &split[board_end + 1..] {
                        let make_move = Move::from_str(token).unwrap();
                        moves.push(make_move);
                    }
                }
                UciCommand::Position(chess_board.unwrap(), moves)
            }
            "go" => {
                let mut commands = vec![];
                while let Some(option) = split.next() {
                    commands.push(match option {
                        "wtime" => {
                            let millis = split.next().unwrap().parse::<i64>().unwrap();
                            let millis = millis.max(0) as u64;
                            TimeManagementInfo::WTime(Duration::from_millis(millis))
                        }
                        "btime" => {
                            let millis = split.next().unwrap().parse::<i64>().unwrap();
                            let millis = millis.max(0) as u64;
                            TimeManagementInfo::BTime(Duration::from_millis(millis))
                        }
                        "winc" => {
                            let millis = split.next().unwrap().parse::<u64>().unwrap();
                            TimeManagementInfo::WInc(Duration::from_millis(millis))
                        }
                        "binc" => {
                            let millis = split.next().unwrap().parse::<u64>().unwrap();
                            TimeManagementInfo::BInc(Duration::from_millis(millis))
                        }
                        "movetime" => {
                            let millis = split.next().unwrap().parse::<u64>().unwrap();
                            TimeManagementInfo::MoveTime(Duration::from_millis(millis))
                        }
                        "movestogo" => {
                            let moves_to_go = split.next().unwrap().parse::<u32>().unwrap();
                            TimeManagementInfo::MovesToGo(moves_to_go)
                        }
                        "depth" => {
                            let depth = split.next().unwrap().parse::<u32>().unwrap();
                            TimeManagementInfo::MaxDepth(depth)
                        }
                        "nodes" => {
                            let nodes = split.next().unwrap().parse::<u64>().unwrap();
                            TimeManagementInfo::MaxNodes(nodes)
                        }
                        _ => TimeManagementInfo::Unknown,
                    });
                }
                UciCommand::Go(commands)
            }
            "stop" => UciCommand::Stop,
            "quit" => UciCommand::Quit,
            "eval" => UciCommand::Eval,
            "isready" => UciCommand::IsReady,
            "bench" => UciCommand::Bench,
            "static" => UciCommand::Static,
            "setoption" => {
                split.next();
                let name = split.next().unwrap().to_string();
                split.next();
                let value = split.next().unwrap().to_string();
                UciCommand::SetOption(name, value)
            }
            _ => UciCommand::Empty,
        }
    }
}
