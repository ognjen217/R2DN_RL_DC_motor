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
from r2dn_dc_motor.validation.phase5 import (
    BaselinePredictiveMetrics,
    Phase5Check,
    Phase5ValidationReport,
    generate_phase5_artifacts,
    run_phase5_validation,
)
from r2dn_dc_motor.validation.phase6 import (
    Phase6Check,
    Phase6ValidationReport,
    generate_phase6_artifacts,
    run_phase6_validation,
)

__all__ = [
    "Phase2ValidationReport",
    "Phase3ValidationReport",
    "Phase4ValidationReport",
    "Phase5Check",
    "Phase5ValidationReport",
    "Phase6Check",
    "Phase6ValidationReport",
    "BaselinePredictiveMetrics",
    "ProbeWindowResult",
    "DatasetIntegrityCheck",
    "ValidationCheck",
    "generate_phase2_artifacts",
    "generate_phase3_artifacts",
    "generate_phase4_validation_artifacts",
    "generate_phase5_artifacts",
    "generate_phase6_artifacts",
    "run_phase2_validation",
    "run_phase3_validation",
    "run_phase4_validation",
    "run_phase5_validation",
    "run_phase6_validation",
]
