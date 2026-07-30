"""Validated trajectory containers, generators, and model-safe views."""

from r2dn_dc_motor.data.phase4_dataset import (
    NormalizationStatistics,
    Phase4Dataset,
    RawPhase4Trajectory,
)
from r2dn_dc_motor.data.phase4_generation import (
    DatasetGenerationError,
    GeneratedDataset,
    TrajectoryPlan,
    build_trajectory_plans,
    generate_phase4_dataset,
    simulate_trajectory_group,
)
from r2dn_dc_motor.data.r2dn_windows import R2DNWindowBatch, R2DNWindowSampler
from r2dn_dc_motor.data.sequences import FullTrajectoryBatch, ModelSequenceBatch

__all__ = [
    "DatasetGenerationError",
    "FullTrajectoryBatch",
    "GeneratedDataset",
    "ModelSequenceBatch",
    "NormalizationStatistics",
    "Phase4Dataset",
    "RawPhase4Trajectory",
    "R2DNWindowBatch",
    "R2DNWindowSampler",
    "TrajectoryPlan",
    "build_trajectory_plans",
    "generate_phase4_dataset",
    "simulate_trajectory_group",
]
