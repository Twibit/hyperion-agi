//! Crate-wide error type.
//!
//! We use `thiserror` for the library surface and re-export `anyhow::Result`
//! as `TkResult` only inside binaries and examples — library code returns
//! the strongly-typed [`TkError`].

use std::io;
use std::path::PathBuf;

use thiserror::Error;

/// Result alias used across the crate.
pub type TkResult<T> = std::result::Result<T, TkError>;

/// Every failure mode a `Limb` can produce.
///
/// The variants are named after the neurophysiology they model so that
/// tracing output reads like a diagnostic on a nervous system rather than
/// a stack trace.
#[derive(Debug, Error)]
pub enum TkError {
    /// A raw OS I/O error at the effector.
    #[error("effector I/O failure: {0}")]
    Effector(#[from] io::Error),

    /// The proprioceptive verifier detected that reality diverged from the
    /// efference-copy prediction (something else changed the file underneath
    /// us, or the write did not land as expected).
    #[error("reafference mismatch on {path}: expected {expected}, saw {actual}")]
    ReafferenceMismatch {
        /// Path whose observed state disagreed with the prediction.
        path: PathBuf,
        /// The predicted marker (typically a hash or size).
        expected: String,
        /// The actually-observed marker.
        actual: String,
    },

    /// The motor cortex could not translate this intent into concrete commands.
    #[error("motor planning failure: {0}")]
    MotorPlan(String),

    /// The pre-motor "readiness" gate rejected the command (policy denial,
    /// rate limit, sandbox violation).
    #[error("readiness gate denied command: {0}")]
    ReadinessDenied(String),

    /// A regex, glob, or pattern the caller supplied could not be parsed.
    #[error("invalid pattern: {0}")]
    Pattern(String),

    /// An actor channel closed unexpectedly — usually because the
    /// nervous system was shut down while a command was in flight.
    #[error("nervous system severed: {0}")]
    Severed(String),

    /// Any other error escalated with context.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl TkError {
    /// Convenience constructor for motor-planning failures.
    pub fn plan(msg: impl Into<String>) -> Self {
        TkError::MotorPlan(msg.into())
    }
}
