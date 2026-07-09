//! The file-search "eye" — a hybrid ripgrep-class walker plus a ranked scorer.
//!
//! **Layer 1 (walker):** `ignore` crate — the same engine ripgrep uses.
//! Parallel, gitignore-aware, respects hidden-file conventions.
//!
//! **Layer 2 (matcher):** `grep-searcher` + `grep-regex` — again, ripgrep's
//! own machinery. Line-oriented regex matching over memory-mapped or
//! streamed files.
//!
//! **Layer 3 (scorer):** custom lightweight ranking that combines
//!  * match count per file (raw signal),
//!  * an inverse-path-depth bonus (files nearer the root rank higher —
//!    approximates the "salience" of a mirror-neuron's preferred stimulus),
//!  * a filename-hit bonus (matches in the filename outrank matches deep
//!    inside a file, mirroring the extra weight a mirror neuron gives to
//!    a familiar goal-object).
//!
//! The score is normalised to `[0, 1]` so downstream consumers can compare
//! results from different search invocations.

mod engine;
mod score;

pub use engine::SearchLimb;
pub use score::score_hits;

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// A single hit inside a file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    /// Path to the file containing the hit.
    pub path: PathBuf,
    /// 1-indexed line number.
    pub line: usize,
    /// The matching line's text (trimmed to at most 512 chars).
    pub excerpt: String,
    /// Normalised relevance score in `[0, 1]`.
    pub score: f32,
}

/// A ranked bundle of hits.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchResults {
    /// Total number of raw matches before top-k truncation.
    pub total_matches: usize,
    /// Number of files scanned.
    pub files_scanned: usize,
    /// Top-k hits, sorted by descending `score`.
    pub hits: Vec<SearchHit>,
}

impl SearchResults {
    /// Consume this bundle and return only the hits vector.
    pub fn into_top(self) -> Vec<SearchHit> {
        self.hits
    }
}
