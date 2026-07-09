//! Mirror-neuron observability layer.
//!
//! Rizzolatti's F5 mirror neurons fire when the animal acts *and* when it
//! observes another performing the same action. That's a compact description
//! of what a good tracing layer does: it lets a supervisor process
//! "watch" a limb from the outside and reconstruct exactly what the limb did,
//! as if the supervisor had done it itself.
//!
//! Here `Observer` is a thin wrapper around [`tracing`] that emits structured
//! events for every command dispatch and every reafference report.

use tracing::{event, span, Level, Span};

use crate::cortex::MotorCommand;
use crate::proprio::ReafferenceReport;

/// A per-limb mirror-neuron observer. Cheap to clone.
#[derive(Debug, Clone)]
pub struct Observer {
    limb_name: String,
}

impl Observer {
    /// Build an observer labelled with the given limb name.
    pub fn new(limb_name: impl Into<String>) -> Self {
        Self {
            limb_name: limb_name.into(),
        }
    }

    /// Begin a tracing span for a single command dispatch.
    pub fn span(&self, cmd: &MotorCommand) -> Span {
        span!(
            Level::DEBUG,
            "reach",
            limb = %self.limb_name,
            op = std::any::type_name_of_val(cmd),
            path = ?cmd.path(),
        )
    }

    /// Emit a mirror event when the limb about to dispatch.
    pub fn about_to_act(&self, cmd: &MotorCommand) {
        event!(
            Level::DEBUG,
            limb = %self.limb_name,
            mutating = cmd.is_mutating(),
            "efferent command dispatched"
        );
    }

    /// Emit a mirror event for a reafference report.
    pub fn observed(&self, r: &ReafferenceReport) {
        if r.coherent {
            event!(
                Level::DEBUG,
                limb = %self.limb_name,
                path = %r.path.display(),
                "reafference coherent"
            );
        } else {
            event!(
                Level::WARN,
                limb = %self.limb_name,
                path = %r.path.display(),
                expected = %r.expected,
                actual = %r.actual,
                "reafference mismatch"
            );
        }
    }
}
