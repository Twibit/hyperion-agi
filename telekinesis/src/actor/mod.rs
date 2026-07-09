//! The actor mesh — the "nervous system."
//!
//! Each `Limb` runs in its own tokio task with a private mpsc mailbox.
//! Callers hold a cloneable [`LimbHandle`](crate::limb::LimbHandle) and send
//! `Intent`s over the channel; the actor plans, executes, verifies, and
//! replies via a `oneshot`.
//!
//! Cross-limb operations (e.g. "search then annotate") stay in caller code
//! — we deliberately do *not* let one limb reach into another's mailbox.
//! That's what keeps the model biologically plausible: hands do not
//! telepathically talk to eyes; the cortex mediates.

use std::sync::Arc;

use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::cortex::MotorCortex;
use crate::error::TkResult;
use crate::intent::Outcome;
use crate::limb::{Envelope, Limb, LimbHandle};

/// The top-level supervisor that owns every limb actor.
///
/// Drop the `NervousSystem` and every actor task shuts down cooperatively
/// as its inbox closes.
pub struct NervousSystem {
    tasks: Vec<JoinHandle<()>>,
    cortex: Arc<MotorCortex>,
}

impl NervousSystem {
    /// Construct a fresh nervous system with a default cortex.
    pub async fn new() -> TkResult<Self> {
        Ok(Self {
            tasks: Vec::new(),
            cortex: Arc::new(MotorCortex::new()),
        })
    }

    /// Grow a new limb: spawn its actor task, return a shared handle.
    pub async fn grow_limb<L: Limb>(&mut self, limb: L) -> LimbHandle {
        let (tx, mut rx) = mpsc::channel::<Envelope>(64);
        let handle = LimbHandle::new(limb.name().to_string(), tx);
        let cortex = Arc::clone(&self.cortex);
        let limb = Arc::new(limb);
        let task = tokio::spawn(async move {
            while let Some(env) = rx.recv().await {
                let plan = match cortex.plan(env.intent) {
                    Ok(p) => p,
                    Err(e) => {
                        let _ = env.reply.send(Err(e));
                        continue;
                    }
                };
                // For a single-command bundle, return that command's outcome
                // directly. For a multi-command bundle, return the last one's
                // outcome and short-circuit on error. This mirrors how a
                // motor plan aborts on the first proprioceptive failure.
                let mut last: TkResult<Outcome> = Ok(Outcome::Ack);
                for cmd in plan {
                    last = limb.execute(cmd).await;
                    if last.is_err() {
                        break;
                    }
                }
                let _ = env.reply.send(last);
            }
        });
        self.tasks.push(task);
        handle
    }

    /// Number of live limbs.
    pub fn limb_count(&self) -> usize {
        self.tasks.len()
    }

    /// Gracefully await every limb task after all handles have been dropped.
    ///
    /// Callers usually don't need this — dropping `NervousSystem` is enough.
    pub async fn shutdown(self) {
        for t in self.tasks {
            let _ = t.await;
        }
    }
}
