"""World-model interfaces and implementations."""

from r2dn_dc_motor.models.isothermal_calibration import (
    ALLOWED_FIT_FEATURES,
    CALIBRATED_PARAMETER_NAMES,
    FORBIDDEN_FIT_FEATURES,
    CalibrationSufficientStatistics,
    IsothermalCalibrationCheckpoint,
    fit_global_isothermal_parameters,
    nominal_isothermal_parameters,
)
from r2dn_dc_motor.models.jax_runtime import JAXRuntime, inspect_jax_runtime
from r2dn_dc_motor.models.r2dn_adapter import (
    OfficialR2DNAdapter,
    R2DNArchitecture,
    R2DNBackendUnavailable,
)
from r2dn_dc_motor.models.r2dn_training import (
    ALLOWED_TRAINING_FEATURES,
    LoadedPhase6Checkpoint,
    Phase6CheckpointManifest,
    Phase6TrainingStudy,
    R2DNRunResult,
    ValidationRolloutMetrics,
    evaluate_validation_rollout,
    load_phase6_checkpoint,
    save_phase6_checkpoint,
    train_phase6_study,
)
from r2dn_dc_motor.models.temperature_probe import (
    HISTORY_FEATURE_NAMES,
    INSTANTANEOUS_FEATURE_NAMES,
    ProbeSamples,
    ProbeTrajectory,
    StandardizedRidge,
    build_probe_samples,
)

__all__ = [
    "ALLOWED_FIT_FEATURES",
    "ALLOWED_TRAINING_FEATURES",
    "CALIBRATED_PARAMETER_NAMES",
    "FORBIDDEN_FIT_FEATURES",
    "CalibrationSufficientStatistics",
    "HISTORY_FEATURE_NAMES",
    "INSTANTANEOUS_FEATURE_NAMES",
    "IsothermalCalibrationCheckpoint",
    "JAXRuntime",
    "LoadedPhase6Checkpoint",
    "OfficialR2DNAdapter",
    "ProbeSamples",
    "ProbeTrajectory",
    "Phase6CheckpointManifest",
    "Phase6TrainingStudy",
    "R2DNArchitecture",
    "R2DNBackendUnavailable",
    "R2DNRunResult",
    "StandardizedRidge",
    "ValidationRolloutMetrics",
    "build_probe_samples",
    "fit_global_isothermal_parameters",
    "inspect_jax_runtime",
    "evaluate_validation_rollout",
    "load_phase6_checkpoint",
    "nominal_isothermal_parameters",
    "save_phase6_checkpoint",
    "train_phase6_study",
]
