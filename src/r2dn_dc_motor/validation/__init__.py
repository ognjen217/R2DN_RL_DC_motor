"""Executable validation gates."""

from r2dn_dc_motor.validation.phase2 import (
    Phase2ValidationReport,
    ValidationCheck,
    generate_phase2_artifacts,
    run_phase2_validation,
)
from r2dn_dc_motor.validation.phase3 import (
    Phase3ValidationReport,
    ProbeWindowResult,
    generate_phase3_artifacts,
    run_phase3_validation,
)
from r2dn_dc_motor.validation.phase4 import (
    DatasetIntegrityCheck,
    Phase4ValidationReport,
    generate_phase4_validation_artifacts,
    run_phase4_validation,
)

__all__ = [
    "Phase2ValidationReport",
    "Phase3ValidationReport",
    "Phase4ValidationReport",
    "ProbeWindowResult",
    "DatasetIntegrityCheck",
    "ValidationCheck",
    "generate_phase2_artifacts",
    "generate_phase3_artifacts",
    "generate_phase4_validation_artifacts",
    "run_phase2_validation",
    "run_phase3_validation",
    "run_phase4_validation",
]
