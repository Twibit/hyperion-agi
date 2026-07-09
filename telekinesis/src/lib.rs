//! # telekinesis
//!
//! A biologically-grounded data-manipulation limb for AI agents.
//!
//! The crate treats file/data manipulation as a **sensorimotor loop**:
//!
//! ```text
//!   Intent  ─► MotorCortex ─► Limb ─► World
//!                 │                     │
//!                 ▼                     ▼
//!            EfferenceCopy ◄── Proprioceptor
//! ```
//!
//! See `docs/BIOLOGY.md` in the source tree for full theoretical grounding.
//!
//! ## Minimal example
//!
//! ```no_run
//! use telekinesis::prelude::*;
//!
//! # async fn demo() -> anyhow::Result<()> {
//! let mut ns = NervousSystem::new().await?;
//! let hand = ns.grow_limb(FileLimb::new()).await;
//! let out = hand.reach(Intent::Read { path: "Cargo.toml".into() }).await?;
//! println!("{:?}", out);
//! # Ok(()) }
//! ```

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod actor;
pub mod cortex;
pub mod error;
pub mod intent;
pub mod limb;
pub mod mirror;
pub mod proprio;
pub mod search;

/// The convenience re-export module.
///
/// Bring everything you need into scope with `use telekinesis::prelude::*;`.
pub mod prelude {
    pub use crate::actor::NervousSystem;
    pub use crate::cortex::{
        EfferenceCopy, MotorCommand, MotorCortex, ReadOnly, Readiness, ReadinessPolicy,
        SandboxRoot,
    };
    pub use crate::error::{TkError, TkResult};
    pub use crate::intent::{Intent, Outcome};
    pub use crate::limb::{FileLimb, Limb, LimbHandle};
    pub use crate::mirror::Observer;
    pub use crate::proprio::{Proprioceptor, ReafferenceReport};
    pub use crate::search::{SearchHit, SearchLimb, SearchResults};
}

/// Crate version string, exposed for observability.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
