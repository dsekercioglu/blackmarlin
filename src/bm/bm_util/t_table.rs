use std::sync::atomic::{AtomicU64, Ordering};

use cozy_chess::{Board, Move, Square};

use crate::bm::bm_util::eval::Evaluation;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum EntryType {
    LowerBound,
    Exact,
    UpperBound,
}

#[derive(Debug, Copy, Clone)]
pub struct Analysis {
    exists: bool,
    depth: u8,
    entry_type: EntryType,
    score: Evaluation,
    table_move: Move,
}

impl Analysis {
    pub fn new(depth: u32, entry_type: EntryType, score: Evaluation, table_move: Move) -> Self {
        Self {
            exists: true,
            depth: depth as u8,
            entry_type,
            score,
            table_move,
        }
    }

    pub fn zero() -> Self {
        Self {
            exists: false,
            depth: 0,
            entry_type: EntryType::LowerBound,
            score: Evaluation::new(0),
            table_move: Move {
                from: Square::A1,
                to: Square::A1,
                promotion: None,
            },
        }
    }

    #[inline]
    pub fn depth(&self) -> u32 {
        self.depth as u32
    }

    #[inline]
    pub fn entry_type(&self) -> EntryType {
        self.entry_type
    }

    #[inline]
    pub fn score(&self) -> Evaluation {
        self.score
    }

    #[inline]
    pub fn table_move(&self) -> Move {
        self.table_move
    }
}

#[derive(Debug)]
pub struct Entry {
    hash: AtomicU64,
    analysis: AtomicU64,
}

impl Entry {
    fn zeroed() -> Self {
        unsafe {
            Self {
                hash: AtomicU64::new(std::mem::transmute(Analysis::zero())),
                analysis: AtomicU64::new(std::mem::transmute(Analysis::zero())),
            }
        }
    }
    fn zero(&self) {
        unsafe {
            self.hash
                .store(std::mem::transmute(Analysis::zero()), Ordering::Relaxed);
            self.analysis
                .store(std::mem::transmute(Analysis::zero()), Ordering::Relaxed);
        }
    }

    fn set_new(&self, hash: u64, entry: u64) {
        self.hash.store(hash, Ordering::Relaxed);
        self.analysis.store(entry, Ordering::Relaxed);
    }
}

#[derive(Debug)]
pub struct TranspositionTable {
    table: Box<[Entry]>,
    mask: usize,
}

impl TranspositionTable {
    pub fn new(size: usize) -> Self {
        let size = size.next_power_of_two();
        let table = (0..size).map(|_| Entry::zeroed()).collect::<Box<_>>();
        Self {
            table,
            mask: size - 1,
        }
    }

    #[inline]
    fn index(&self, hash: u64) -> usize {
        (hash as usize) & self.mask
    }

    #[cfg(not(target_feature = "sse"))]
    pub fn prefetch(&self, _: &Board) {}

    #[cfg(target_feature = "sse")]
    pub fn prefetch(&self, board: &Board) {
        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        let hash = board.hash();
        let index = self.index(hash);
        unsafe {
            let ptr = self.table.as_ptr().offset(index as isize);
            _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
        }
    }

    pub fn get(&self, board: &Board) -> Option<Analysis> {
        let hash = board.hash();
        let index = self.index(hash);

        let entry = &self.table[index];
        let hash_u64 = entry.hash.load(Ordering::Relaxed);
        let entry_u64 = entry.analysis.load(Ordering::Relaxed);
        if entry_u64 ^ hash == hash_u64 {
            let analysis: Analysis = unsafe { std::mem::transmute(entry_u64) };
            if analysis.exists {
                Some(analysis)
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn set(&self, board: &Board, entry: Analysis) {
        let hash = board.hash();
        let index = self.index(hash);
        let fetched_entry = &self.table[index];
        let analysis: Analysis =
            unsafe { std::mem::transmute(fetched_entry.analysis.load(Ordering::Relaxed)) };

        if !analysis.exists || Self::do_replace(&entry, &analysis) {
            let analysis_u64 = unsafe { std::mem::transmute::<Analysis, u64>(entry) };
            fetched_entry.set_new(hash ^ analysis_u64, analysis_u64);
        }
    }

    fn do_replace(a: &Analysis, b: &Analysis) -> bool {
        let a_extra_depth =
            matches!(a.entry_type(), EntryType::Exact | EntryType::LowerBound) as u8 * 2;
        let b_extra_depth =
            matches!(b.entry_type(), EntryType::Exact | EntryType::LowerBound) as u8 * 2;
        (a.depth + a_extra_depth) >= (b.depth + b_extra_depth) / 2
    }

    pub fn clean(&self) {
        self.table.iter().for_each(|entry| entry.zero());
    }
}
