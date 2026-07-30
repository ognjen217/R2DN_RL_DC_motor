"""Time-major trajectory containers for system identification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


class SequenceValidationError(ValueError):
    """Raised when a trajectory violates the frozen data contract."""


def _as_float_array(value: FloatArray) -> FloatArray:
    return np.asarray(value, dtype=np.float32)


@dataclass(frozen=True)
class ModelSequenceBatch:
    """Temperature-free R2DN view of a time-major trajectory batch.

    ``observations`` has shape ``(T + 1, B, 2)`` and ``controls`` has shape
    ``(T, B, 1)``. Transition ``k`` uses ``[y_k, u_k]`` to predict ``y_{k+1}``.
    """

    observations: FloatArray
    controls: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "observations", _as_float_array(self.observations))
        object.__setattr__(self, "controls", _as_float_array(self.controls))
        self.validate()

    @property
    def time_steps(self) -> int:
        return int(self.controls.shape[0])

    @property
    def batch_size(self) -> int:
        return int(self.controls.shape[1])

    def validate(self) -> None:
        errors: list[str] = []
        observations = self.observations
        controls = self.controls

        if observations.ndim != 3:
            errors.append("observations must have shape (T + 1, B, 2)")
        if controls.ndim != 3:
            errors.append("controls must have shape (T, B, 1)")
        if errors:
            raise SequenceValidationError("\n".join(errors))

        if observations.shape[-1] != 2:
            errors.append("observation feature dimension must be 2")
        if controls.shape[-1] != 1:
            errors.append("control feature dimension must be 1")
        if observations.shape[0] != controls.shape[0] + 1:
            errors.append("observations must contain exactly one more time step than controls")
        if observations.shape[1] != controls.shape[1]:
            errors.append("observation and control batch dimensions must match")
        if controls.shape[0] < 1:
            errors.append("a sequence must contain at least one transition")
        if controls.shape[1] < 1:
            errors.append("a sequence batch must contain at least one trajectory")
        if not np.isfinite(observations).all():
            errors.append("observations contain NaN or infinite values")
        if not np.isfinite(controls).all():
            errors.append("controls contain NaN or infinite values")

        if errors:
            raise SequenceValidationError("\n".join(errors))

    def training_pairs(self) -> tuple[FloatArray, FloatArray]:
        """Return ``([y_k, u_k], y_{k+1})`` in time-batch-feature layout."""

        regressors = np.concatenate((self.observations[:-1], self.controls), axis=-1)
        targets = self.observations[1:]
        return regressors, targets

    def burn_in_inputs(
        self,
        steps: int,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Return measured burn-in inputs and the measured rollout seed ``y_B``."""

        if steps < 1:
            raise SequenceValidationError("burn-in must contain at least one transition")
        if steps > self.time_steps:
            raise SequenceValidationError("burn-in cannot be longer than the sequence")
        return (
            self.observations[:steps],
            self.controls[:steps],
            self.observations[steps],
        )


@dataclass(frozen=True)
class FullTrajectoryBatch:
    """Raw FULL-simulator trajectory with temperature retained for evaluation."""

    full_states: FloatArray
    controls: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "full_states", _as_float_array(self.full_states))
        object.__setattr__(self, "controls", _as_float_array(self.controls))
        self.validate()

    def validate(self) -> None:
        errors: list[str] = []
        states = self.full_states
        controls = self.controls

        if states.ndim != 3:
            errors.append("full_states must have shape (T + 1, B, 3)")
        if controls.ndim != 3:
            errors.append("controls must have shape (T, B, 1)")
        if errors:
            raise SequenceValidationError("\n".join(errors))

        if states.shape[-1] != 3:
            errors.append("FULL state order must be [current, speed, temperature]")
        if controls.shape[-1] != 1:
            errors.append("control feature dimension must be 1")
        if states.shape[0] != controls.shape[0] + 1:
            errors.append("full_states must contain exactly one more time step than controls")
        if states.shape[1] != controls.shape[1]:
            errors.append("state and control batch dimensions must match")
        if controls.shape[0] < 1 or controls.shape[1] < 1:
            errors.append("a FULL trajectory batch must be non-empty")
        if not np.isfinite(states).all() or not np.isfinite(controls).all():
            errors.append("FULL trajectory contains NaN or infinite values")

        if errors:
            raise SequenceValidationError("\n".join(errors))

    @property
    def temperature(self) -> FloatArray:
        """Evaluation-only winding temperature with shape ``(T + 1, B, 1)``."""

        return self.full_states[..., 2:3]

    def model_view(self) -> ModelSequenceBatch:
        """Drop temperature and return the only view allowed for R2DN training."""

        return ModelSequenceBatch(
            observations=self.full_states[..., :2].copy(),
            controls=self.controls.copy(),
        )
