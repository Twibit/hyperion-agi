//! `Intent` and `Outcome` — the *only* things that cross the actor boundary.
//!
//! Analogous to the abstract "goal state" that the pre-motor cortex receives
//! from higher-order association cortex. The [`MotorCortex`](crate::cortex::MotorCortex)
//! decomposes an `Intent` into concrete
//! [`MotorCommand`](crate::cortex::MotorCommand)s that a `Limb` can execute.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// A high-level goal handed to a [`Limb`](crate::limb::Limb).
///
/// Intents are declarative — they say *what* the caller wants, not *how* to
/// achieve it. Decomposition into low-level commands is the motor cortex's
/// job.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Intent {
    /// Read a file's contents.
    Read {
        /// Absolute or relative path.
        path: PathBuf,
    },

    /// Write bytes to a file, creating or truncating.
    Write {
        /// Destination path.
        path: PathBuf,
        /// Payload.
        data: Vec<u8>,
    },

    /// Append a line-annotation near a specific line.
    Annotate {
        /// Target file.
        path: PathBuf,
        /// 1-indexed line.
        line: usize,
        /// Text to attach as a `// TELEKINESIS: ...` comment.
        note: String,
    },

    /// Delete a file (with efference-copy backup so rollback is possible).
    Delete {
        /// Path to remove.
        path: PathBuf,
    },

    /// Search files under `root` (if set on the SearchLimb) for `query`,
    /// with optional path-regex `pattern`, returning the top-k hits.
    Search {
        /// Text or regex to search for inside files.
        query: String,
        /// Optional regex restricting which paths are searched.
        pattern: Option<String>,
        /// Cap on returned hits.
        top_k: usize,
    },

    /// Perform a network request.
    Fetch {
        /// HTTP method.
        method: String,
        /// Target URL.
        url: String,
        /// Optional JSON payload.
        payload: Option<String>,
    },
}

/// A result value returned from `Limb::reach`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Outcome {
    /// Bytes read from a file.
    Bytes {
        /// The raw bytes.
        data: Vec<u8>,
    },
    /// Number of bytes written.
    Wrote {
        /// Byte count.
        count: usize,
    },
    /// The file was successfully deleted.
    Deleted,
    /// The file was annotated at the given line.
    Annotated {
        /// Path of the file that was annotated.
        path: PathBuf,
        /// Line number.
        line: usize,
    },
    /// A search returned a set of hits.
    SearchHits {
        /// The results.
        results: crate::search::SearchResults,
    },
    /// A network response payload.
    NetworkResponse {
        /// The raw bytes.
        data: Vec<u8>,
    },
    /// A no-op / acknowledgment.
    Ack,
}

impl Outcome {
    /// Convert to bytes if this outcome is a [`Outcome::Bytes`].
    pub fn into_bytes(self) -> Option<Vec<u8>> {
        match self {
            Outcome::Bytes { data } => Some(data),
            _ => None,
        }
    }

    /// Convert to a [`SearchResults`](crate::search::SearchResults).
    pub fn into_search(self) -> Option<crate::search::SearchResults> {
        match self {
            Outcome::SearchHits { results } => Some(results),
            _ => None,
        }
    }
}
