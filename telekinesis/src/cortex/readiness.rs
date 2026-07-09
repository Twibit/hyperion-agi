//! The Libet-style pre-motor readiness gate.
//!
//! In humans, the *Bereitschaftspotential* precedes conscious awareness of
//! intent to move by 350–550 ms. That window is the *only* opportunity a
//! supervisory system has to veto a motor plan before it becomes an act.
//! We give our software agents the same window.
//!
//! In this crate, `Readiness` is a plug-in policy layer: every command
//! passes through `Readiness::gate` before execution, and any registered
//! policy can veto. Rate-limits, sandbox restrictions, and human-in-the-loop
//! confirmations all attach here.

use std::path::Path;
use std::sync::Arc;

use crate::cortex::MotorCommand;
use crate::error::{TkError, TkResult};

/// Trait for a single readiness-gate policy.
///
/// Return `Ok(())` to allow the command, `Err(TkError::ReadinessDenied(..))`
/// to veto. Implementations should be *fast* — this runs on the hot path.
pub trait ReadinessPolicy: Send + Sync + 'static {
    /// Inspect the command; deny by returning an error.
    fn inspect(&self, cmd: &MotorCommand) -> TkResult<()>;
}

/// The composite readiness gate. Holds a list of policies; a command is
/// admitted only if every policy admits it.
#[derive(Clone, Default)]
pub struct Readiness {
    policies: Vec<Arc<dyn ReadinessPolicy>>,
}

impl std::fmt::Debug for Readiness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Readiness")
            .field("policies", &self.policies.len())
            .finish()
    }
}

impl Readiness {
    /// Empty gate — admits everything.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a policy.
    pub fn with_policy<P: ReadinessPolicy>(mut self, p: P) -> Self {
        self.policies.push(Arc::new(p));
        self
    }

    /// Run every policy in sequence. Short-circuits on the first veto.
    pub fn gate(&self, cmd: &MotorCommand) -> TkResult<()> {
        for p in &self.policies {
            p.inspect(cmd)?;
        }
        Ok(())
    }
}

/// A built-in policy that forbids operations outside a whitelisted root
/// directory — the software equivalent of a leash on the limb.
#[derive(Debug, Clone)]
pub struct SandboxRoot {
    root: std::path::PathBuf,
}

impl SandboxRoot {
    /// New sandbox pinned to `root`.
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
        }
    }
}

impl ReadinessPolicy for SandboxRoot {
    fn inspect(&self, cmd: &MotorCommand) -> TkResult<()> {
        let Some(p) = cmd.path() else { return Ok(()) };
        // Canonicalise best-effort; fall back to lexical comparison.
        let target = p
            .canonicalize()
            .unwrap_or_else(|_| p.clone());
        let root = self
            .root
            .canonicalize()
            .unwrap_or_else(|_| self.root.clone());
        if target.starts_with(&root) {
            Ok(())
        } else {
            Err(TkError::ReadinessDenied(format!(
                "path {target:?} outside sandbox root {root:?}"
            )))
        }
    }
}

/// A policy that forbids all mutating operations — turns the limb read-only.
#[derive(Debug, Clone, Default)]
pub struct ReadOnly;

impl ReadinessPolicy for ReadOnly {
    fn inspect(&self, cmd: &MotorCommand) -> TkResult<()> {
        if cmd.is_mutating() {
            Err(TkError::ReadinessDenied(
                "limb is configured read-only".into(),
            ))
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_gate_admits() {
        let r = Readiness::new();
        assert!(r
            .gate(&MotorCommand::ReadAll {
                path: "/tmp/x".into()
            })
            .is_ok());
    }

    #[test]
    fn read_only_denies_writes() {
        let r = Readiness::new().with_policy(ReadOnly);
        let err = r
            .gate(&MotorCommand::WriteAll {
                path: "/tmp/x".into(),
                data: vec![],
            })
            .unwrap_err();
        assert!(matches!(err, TkError::ReadinessDenied(_)));
    }

    #[test]
    fn read_only_admits_reads() {
        let r = Readiness::new().with_policy(ReadOnly);
        assert!(r
            .gate(&MotorCommand::ReadAll {
                path: "/tmp/x".into()
            })
            .is_ok());
    }
}
