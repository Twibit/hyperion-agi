//! Population-vector-style intent decomposition.
//!
//! Each `Intent` is unpacked into a *bundle* of small, individually-safe
//! `MotorCommand`s. No single command carries the whole plan — the plan
//! lives in the aggregate. That's the software analogue of Georgopoulos'
//! observation that motor cortex encodes direction as a vector sum across
//! ~200 neurons rather than as a single "labeled line."

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::error::{TkError, TkResult};
use crate::intent::Intent;

/// The smallest unit of work a [`Limb`](crate::limb::Limb) can execute.
///
/// Motor commands are always reversible in principle: every mutating variant
/// takes an [`EfferenceCopy`](super::EfferenceCopy) before it runs, so a
/// rollback can be issued when proprioception disagrees with prediction.
///
/// The variants use struct-style fields so that the serde representation
/// is stable and easily inspectable (JSON with a `op` discriminator tag).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum MotorCommand {
    /// Read the entire file into memory.
    ReadAll {
        /// Target path.
        path: PathBuf,
    },
    /// Truncate and write.
    WriteAll {
        /// Target path.
        path: PathBuf,
        /// Bytes to write.
        data: Vec<u8>,
    },
    /// Insert text near a specific line.
    InsertLine {
        /// Target path.
        path: PathBuf,
        /// 1-indexed line number.
        line: usize,
        /// Note text to insert (as a `// TELEKINESIS: ...` comment).
        note: String,
    },
    /// Unlink a file.
    Unlink {
        /// Target path.
        path: PathBuf,
    },
    /// Search: query, optional path filter, top-k limit.
    Search {
        /// Text or regex needle.
        query: String,
        /// Optional path filter (regex).
        pattern: Option<String>,
        /// Top-k cap.
        top_k: usize,
    },
    /// Network fetch request.
    Fetch {
        /// HTTP method.
        method: String,
        /// Target URL.
        url: String,
        /// Optional JSON payload.
        payload: Option<String>,
    },
}

impl MotorCommand {
    /// The path this command acts on, if any (search has no single path).
    pub fn path(&self) -> Option<&PathBuf> {
        match self {
            MotorCommand::ReadAll { path }
            | MotorCommand::WriteAll { path, .. }
            | MotorCommand::InsertLine { path, .. }
            | MotorCommand::Unlink { path } => Some(path),
            MotorCommand::Search { .. }
            | MotorCommand::Fetch { .. } => None,
        }
    }

    /// Whether this command mutates the filesystem (needs efference copy).
    pub fn is_mutating(&self) -> bool {
        matches!(
            self,
            MotorCommand::WriteAll { .. }
                | MotorCommand::InsertLine { .. }
                | MotorCommand::Unlink { .. }
        )
    }
}

/// Translates high-level intents into command bundles.
///
/// Currently every intent maps to exactly one command; the API is deliberately
/// vectorised (`Vec<MotorCommand>`) so that future planners can decompose
/// e.g. "refactor this module" into many primitive commands without changing
/// downstream code.
#[derive(Debug, Default, Clone)]
pub struct MotorCortex;

impl MotorCortex {
    /// Construct a fresh cortex.
    pub fn new() -> Self {
        Self
    }

    /// Decompose an intent into an executable bundle.
    pub fn plan(&self, intent: Intent) -> TkResult<Vec<MotorCommand>> {
        let bundle = match intent {
            Intent::Read { path } => vec![MotorCommand::ReadAll { path }],
            Intent::Write { path, data } => vec![MotorCommand::WriteAll { path, data }],
            Intent::Annotate { path, line, note } => {
                if line == 0 {
                    return Err(TkError::plan("annotate line index is 1-based"));
                }
                vec![MotorCommand::InsertLine { path, line, note }]
            }
            Intent::Delete { path } => vec![MotorCommand::Unlink { path }],
            Intent::Search {
                query,
                pattern,
                top_k,
            } => {
                if top_k == 0 {
                    return Err(TkError::plan("top_k must be > 0"));
                }
                vec![MotorCommand::Search {
                    query,
                    pattern,
                    top_k,
                }]
            }
            Intent::Fetch { method, url, payload } => vec![MotorCommand::Fetch { method, url, payload }],
        };
        Ok(bundle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_intent_produces_read_command() {
        let c = MotorCortex::new();
        let plan = c
            .plan(Intent::Read {
                path: "/tmp/x".into(),
            })
            .unwrap();
        assert_eq!(plan.len(), 1);
        assert!(matches!(plan[0], MotorCommand::ReadAll { .. }));
    }

    #[test]
    fn annotate_zero_line_rejected() {
        let c = MotorCortex::new();
        let err = c
            .plan(Intent::Annotate {
                path: "/tmp/x".into(),
                line: 0,
                note: "n".into(),
            })
            .unwrap_err();
        assert!(matches!(err, TkError::MotorPlan(_)));
    }

    #[test]
    fn search_topk_zero_rejected() {
        let c = MotorCortex::new();
        assert!(c
            .plan(Intent::Search {
                query: "x".into(),
                pattern: None,
                top_k: 0
            })
            .is_err());
    }

    #[test]
    fn is_mutating_classification() {
        assert!(!MotorCommand::ReadAll { path: "/x".into() }.is_mutating());
        assert!(MotorCommand::WriteAll {
            path: "/x".into(),
            data: vec![]
        }
        .is_mutating());
        assert!(MotorCommand::Unlink { path: "/x".into() }.is_mutating());
    }
}
