//! The `Limb` abstraction — pluggable effectors.
//!
//! A `Limb` is anything that can *execute a MotorCommand and return an
//! Outcome*. The three built-in limbs are:
//!
//! * [`FileLimb`] — filesystem hand
//! * [`crate::search::SearchLimb`] — file-search eye
//! * (extend by implementing the trait)

mod file;
mod net;

pub use file::FileLimb;
pub use net::NetLimb;

use async_trait::async_trait;
use tokio::sync::{mpsc, oneshot};

use crate::cortex::MotorCommand;
use crate::error::{TkError, TkResult};
use crate::intent::{Intent, Outcome};

/// The behavioural contract for an effector.
///
/// Implementations must be `Send + Sync` so they can be moved into their
/// own actor task; they own their own internal state.
#[async_trait]
pub trait Limb: Send + Sync + 'static {
    /// A human-readable identifier used in tracing.
    fn name(&self) -> &str;

    /// Execute a single low-level motor command.
    ///
    /// This is the *only* method a limb must implement — plan decomposition,
    /// efference copy, and verification are handled around the call by the
    /// nervous system.
    async fn execute(&self, cmd: MotorCommand) -> TkResult<Outcome>;

    /// Optional decomposition hook. The default implementation delegates
    /// to a fresh [`MotorCortex`](crate::cortex::MotorCortex); override
    /// only for limbs that need custom planning.
    fn plan(&self, intent: Intent) -> TkResult<Vec<MotorCommand>> {
        crate::cortex::MotorCortex::new().plan(intent)
    }
}

/// A cheap, cloneable handle to a running limb actor.
///
/// Sending an intent through the handle routes it to the actor's inbox and
/// awaits the result. Multiple callers may share a handle.
#[derive(Debug, Clone)]
pub struct LimbHandle {
    name: String,
    tx: mpsc::Sender<Envelope>,
}

/// Internal message between the handle and its actor task.
pub(crate) struct Envelope {
    pub(crate) intent: Intent,
    pub(crate) reply: oneshot::Sender<TkResult<Outcome>>,
}

impl LimbHandle {
    /// Wrap the sending half of an actor mailbox into a handle.
    pub(crate) fn new(name: String, tx: mpsc::Sender<Envelope>) -> Self {
        Self { name, tx }
    }

    /// Send an intent to the limb and await its outcome.
    ///
    /// This is the *only* public API on a limb. All plan/predict/verify
    /// machinery is transparent to the caller — exactly like biological
    /// motor control.
    pub async fn reach(&self, intent: Intent) -> TkResult<Outcome> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(Envelope { intent, reply: tx })
            .await
            .map_err(|_| TkError::Severed(format!("limb '{}' inbox closed", self.name)))?;
        rx.await
            .map_err(|_| TkError::Severed(format!("limb '{}' dropped reply", self.name)))?
    }

    /// The limb's name.
    pub fn name(&self) -> &str {
        &self.name
    }
}
