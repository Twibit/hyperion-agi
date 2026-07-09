//! The **motor cortex** — planning, prediction, and the readiness gate.
//!
//! Three components live here:
//!
//! * [`MotorCortex`] — decomposes a high-level [`Intent`](crate::intent::Intent)
//!   into a bundle of low-level [`MotorCommand`]s (Georgopoulos 1986,
//!   population vector coding).
//! * [`EfferenceCopy`] — snapshots the world state the cortex *predicts*
//!   after execution (von Holst & Mittelstaedt 1950).
//! * [`Readiness`] — a pre-motor gate that must accept every command
//!   (Libet 1983, readiness potential).

pub(crate) mod efference;
mod planner;
mod readiness;

pub use efference::EfferenceCopy;
pub use planner::{MotorCommand, MotorCortex};
pub use readiness::{ReadOnly, Readiness, ReadinessPolicy, SandboxRoot};
