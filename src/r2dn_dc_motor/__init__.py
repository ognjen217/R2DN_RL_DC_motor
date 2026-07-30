"""R2DN DC-motor control experiments."""

from r2dn_dc_motor.phase1_spec import Phase1Spec, load_phase1_spec
from r2dn_dc_motor.phase2_spec import Phase2Spec, load_phase2_spec
from r2dn_dc_motor.spec import ExperimentSpec, SpecValidationError, load_phase0_spec

__all__ = [
    "ExperimentSpec",
    "Phase1Spec",
    "Phase2Spec",
    "SpecValidationError",
    "load_phase0_spec",
    "load_phase1_spec",
    "load_phase2_spec",
]
