//! Proprioception — the sensory feedback that closes the motor loop.
//!
//! Sherrington called proprioception the "sixth sense": muscle spindles and
//! Golgi tendon organs continuously report actual limb position and force
//! back to the CNS. Without it, you can *move* but you cannot know if you
//! *succeeded*.
//!
//! In software, the analogue is dead-simple and essential: after every
//! mutating command, re-observe the affected resource, compare against the
//! [`EfferenceCopy`](crate::cortex::EfferenceCopy)'s prediction, and report.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::cortex::efference::Snapshot;
use crate::error::TkResult;

/// A single verification report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReafferenceReport {
    /// The path that was verified.
    pub path: PathBuf,
    /// Whether observed state matches the prediction.
    pub coherent: bool,
    /// Marker that was expected (from the efference copy).
    pub expected: String,
    /// Marker actually observed on disk.
    pub actual: String,
}

/// The verifier — stateless; owns no data. Analogous to a spinal reflex arc.
#[derive(Debug, Default, Clone, Copy)]
pub struct Proprioceptor;

impl Proprioceptor {
    /// Construct a fresh proprioceptor.
    pub fn new() -> Self {
        Self
    }

    /// Verify that `path`'s current state matches the *predicted post-state*
    /// derived from the pre-snapshot plus the intended write.
    ///
    /// For a write, `predicted_bytes` is what we wrote; we then read the
    /// file back and compare. For a delete, `predicted_bytes` should be
    /// `None` and the file should not exist.
    pub fn verify(
        &self,
        path: &Path,
        predicted_bytes: Option<&[u8]>,
    ) -> TkResult<ReafferenceReport> {
        let expected = Snapshot::marker_for(predicted_bytes);
        let actual_bytes = std::fs::read(path).ok();
        let actual = Snapshot::marker_for(actual_bytes.as_deref());
        Ok(ReafferenceReport {
            path: path.to_path_buf(),
            coherent: expected == actual,
            expected,
            actual,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn coherent_after_write() {
        let d = tempdir().unwrap();
        let p = d.path().join("a");
        std::fs::write(&p, b"payload").unwrap();
        let r = Proprioceptor::new().verify(&p, Some(b"payload")).unwrap();
        assert!(r.coherent);
    }

    #[test]
    fn incoherent_when_reality_diverges() {
        let d = tempdir().unwrap();
        let p = d.path().join("a");
        std::fs::write(&p, b"actual").unwrap();
        let r = Proprioceptor::new().verify(&p, Some(b"predicted")).unwrap();
        assert!(!r.coherent);
    }

    #[test]
    fn coherent_after_delete() {
        let d = tempdir().unwrap();
        let p = d.path().join("gone");
        let r = Proprioceptor::new().verify(&p, None).unwrap();
        assert!(r.coherent, "no-file matches no-predicted-bytes");
    }
}
