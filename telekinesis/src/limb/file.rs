//! `FileLimb` — the filesystem "hand".
//!
//! Implements read / write / annotate / delete with full efference-copy
//! and proprioceptive verification wired in around every mutation. When
//! reafference disagrees with the prediction, the limb rolls the
//! filesystem back to its pre-command state and surfaces
//! [`TkError::ReafferenceMismatch`](crate::error::TkError::ReafferenceMismatch).

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;

use crate::cortex::{EfferenceCopy, MotorCommand};
use crate::error::{TkError, TkResult};
use crate::intent::Outcome;
use crate::mirror::Observer;
use crate::proprio::Proprioceptor;

/// The filesystem effector.
///
/// A single `FileLimb` is happy to be shared across many actor tasks (it
/// holds only an `Arc<EfferenceCopy>`), but the recommended pattern is one
/// limb → one actor via [`NervousSystem::grow_limb`](crate::actor::NervousSystem::grow_limb).
#[derive(Debug, Clone)]
pub struct FileLimb {
    name: String,
    efference: Arc<EfferenceCopy>,
    proprio: Proprioceptor,
    observer: Observer,
}

impl FileLimb {
    /// A fresh limb with a default-sized efference-copy buffer.
    pub fn new() -> Self {
        Self::named("file-hand")
    }

    /// A fresh limb with a custom name (used in tracing).
    pub fn named(name: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            observer: Observer::new(name.clone()),
            name,
            efference: Arc::new(EfferenceCopy::new()),
            proprio: Proprioceptor::new(),
        }
    }

    /// Configure the efference-copy buffer capacity.
    pub fn with_efference_capacity(mut self, cap: usize) -> Self {
        self.efference = Arc::new(EfferenceCopy::with_capacity(cap));
        self
    }
}

impl Default for FileLimb {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl super::Limb for FileLimb {
    fn name(&self) -> &str {
        &self.name
    }

    async fn execute(&self, cmd: MotorCommand) -> TkResult<Outcome> {
        self.observer.about_to_act(&cmd);
        match cmd {
            MotorCommand::ReadAll { path } => read_all(&path).map(|data| Outcome::Bytes { data }),

            MotorCommand::WriteAll { path, data } => {
                let snap_id = self.efference.snapshot(&path);
                let write_result = std::fs::write(&path, &data);
                if let Err(e) = write_result {
                    self.efference.take(snap_id);
                    return Err(TkError::Effector(e));
                }
                let report = self.proprio.verify(&path, Some(&data))?;
                self.observer.observed(&report);
                if !report.coherent {
                    self.rollback(snap_id, &path)?;
                    return Err(TkError::ReafferenceMismatch {
                        path: report.path,
                        expected: report.expected,
                        actual: report.actual,
                    });
                }
                self.efference.take(snap_id);
                Ok(Outcome::Wrote { count: data.len() })
            }

            MotorCommand::InsertLine { path, line, note } => {
                let snap_id = self.efference.snapshot(&path);
                let updated = match insert_line(&path, line, &note) {
                    Ok(u) => u,
                    Err(e) => {
                        self.efference.take(snap_id);
                        return Err(e);
                    }
                };
                let report = self.proprio.verify(&path, Some(&updated))?;
                self.observer.observed(&report);
                if !report.coherent {
                    self.rollback(snap_id, &path)?;
                    return Err(TkError::ReafferenceMismatch {
                        path: report.path,
                        expected: report.expected,
                        actual: report.actual,
                    });
                }
                self.efference.take(snap_id);
                Ok(Outcome::Annotated { path, line })
            }

            MotorCommand::Unlink { path } => {
                let snap_id = self.efference.snapshot(&path);
                if let Err(e) = std::fs::remove_file(&path) {
                    self.efference.take(snap_id);
                    return Err(TkError::Effector(e));
                }
                let report = self.proprio.verify(&path, None)?;
                self.observer.observed(&report);
                if !report.coherent {
                    self.rollback(snap_id, &path)?;
                    return Err(TkError::ReafferenceMismatch {
                        path: report.path,
                        expected: report.expected,
                        actual: report.actual,
                    });
                }
                self.efference.take(snap_id);
                Ok(Outcome::Deleted)
            }

            MotorCommand::Search { .. } | MotorCommand::Fetch { .. } => Err(TkError::plan(
                "FileLimb cannot execute Search/Fetch; route to appropriate limb",
            )),
        }
    }
}

impl FileLimb {
    fn rollback(&self, snap_id: uuid::Uuid, _path: &std::path::Path) -> TkResult<()> {
        let Some(snap) = self.efference.take(snap_id) else {
            return Ok(());
        };
        match snap.prior_bytes {
            Some(bytes) => std::fs::write(&snap.path, bytes)?,
            None => {
                // File didn't exist before — remove any spurious creation.
                let _ = std::fs::remove_file(&snap.path);
            }
        }
        Ok(())
    }
}

fn read_all(path: &PathBuf) -> TkResult<Vec<u8>> {
    Ok(std::fs::read(path)?)
}

fn insert_line(path: &PathBuf, line_1based: usize, note: &str) -> TkResult<Vec<u8>> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut buffer: Vec<String> = reader
        .lines()
        .collect::<Result<Vec<_>, _>>()?;
    let idx = line_1based.saturating_sub(1).min(buffer.len());
    let annotated = format!("// TELEKINESIS: {note}");
    buffer.insert(idx, annotated);
    let mut out = Vec::with_capacity(buffer.iter().map(|l| l.len() + 1).sum());
    for l in &buffer {
        writeln!(&mut out, "{l}").expect("writing to Vec cannot fail");
    }
    std::fs::write(path, &out)?;
    Ok(out)
}
