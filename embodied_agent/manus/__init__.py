"""
Aurora-Manus (v10) — the efferent / agentic layer.

The five modules in this package address the *action-side* failure
modes that v3-v9 did not close.  v9 gave the organism a mind that could
refuse; v10 gives it hands that can act — and, crucially, *know* that
they acted.

    * MotorCortex          — schema-checked tool inventory with efference copy
    * AgencyMonitor        — Frith-Blakemore comparator; sense-of-agency scalar
    * SomatosensoryField   — proprioceptive body schema + haptic touch trace
    * IntrinsicDrive       — mastery / novelty / competence intrinsic reward
    * MirrorSystem         — observation-to-execution mapping via effect schemas

All five are strictly additive.  ``AuroraOmniaConfig().as_v9()`` disables
them and the runtime is byte-for-byte the v9 behaviour.
"""
from .motor_cortex import MotorCortex, ToolSpec, EfferenceCopy, MotorReturn
from .agency_monitor import AgencyMonitor, AgencyReport
from .somatosensory import SomatosensoryField
from .intrinsic_drive import IntrinsicDrive, IntrinsicSample
from .mirror_system import MirrorSystem, MirrorMatch

__all__ = [
    "MotorCortex", "ToolSpec", "EfferenceCopy", "MotorReturn",
    "AgencyMonitor", "AgencyReport",
    "SomatosensoryField",
    "IntrinsicDrive", "IntrinsicSample",
    "MirrorSystem", "MirrorMatch",
]
