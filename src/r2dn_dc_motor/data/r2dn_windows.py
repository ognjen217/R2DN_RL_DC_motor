"""Temperature-free, train-normalized windows for R2DN fitting and validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from r2dn_dc_motor.data.phase4_dataset import Phase4Dataset

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class R2DNWindowBatch:
    """A normalized time-major batch with provenance but no hidden variables."""

    observations: FloatArray
    controls: FloatArray
    trajectory_ids: tuple[str, ...]
    start_steps: tuple[int, ...]
    split: str

    def __post_init__(self) -> None:
        observations = np.asarray(self.observations, dtype=np.float32)
        controls = np.asarray(self.controls, dtype=np.float32)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "controls", controls)
        self.validate()

    @property
    def transitions(self) -> int:
        return int(self.controls.shape[0])

    @property
    def batch_size(self) -> int:
        return int(self.controls.shape[1])

    def validate(self) -> None:
        errors: list[str] = []
        if self.observations.ndim != 3 or self.observations.shape[-1] != 2:
            errors.append("observations must have shape (T + 1, B, 2)")
        if self.controls.ndim != 3 or self.controls.shape[-1] != 1:
            errors.append("controls must have shape (T, B, 1)")
        if errors:
            raise ValueError("\n".join(errors))
        if self.observations.shape[0] != self.controls.shape[0] + 1:
            errors.append("window observations must contain one extra sample")
        if self.observations.shape[1] != self.controls.shape[1]:
            errors.append("window batch dimensions do not match")
        if len(self.trajectory_ids) != self.batch_size:
            errors.append("trajectory provenance length does not match batch size")
        if len(self.start_steps) != self.batch_size:
            errors.append("window start provenance length does not match batch size")
        if self.split not in {"train", "validation"}:
            errors.append("R2DN windows may only come from train or validation")
        if self.transitions < 1 or self.batch_size < 1:
            errors.append("R2DN window batch must be non-empty")
        if not np.isfinite(self.observations).all():
            errors.append("normalized observations contain NaN or infinite values")
        if not np.isfinite(self.controls).all():
            errors.append("normalized controls contain NaN or infinite values")
        if errors:
            raise ValueError("\n".join(errors))


class R2DNWindowSampler:
    """Sample whole-trajectory-contained windows through the Phase-4 model view."""

    def __init__(
        self,
        dataset: Phase4Dataset,
        *,
        split: str,
        seed: int,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("R2DN sampler split must be train or validation")
        if seed < 0:
            raise ValueError("R2DN sampler seed must be non-negative")
        self.dataset = dataset
        self.split = split
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._normalization = dataset.normalization
        self._trajectories = {
            trajectory_id: dataset.model_view(trajectory_id)
            for trajectory_id in dataset.trajectory_ids(split)
        }
        if not self._trajectories:
            raise ValueError(f"dataset has no {split} trajectories")

    @property
    def trajectory_ids(self) -> tuple[str, ...]:
        return tuple(self._trajectories)

    def sample(
        self,
        *,
        batch_size: int,
        burn_in_steps: int,
        rollout_steps: int,
    ) -> R2DNWindowBatch:
        """Draw a reproducible random batch without crossing trajectory boundaries."""

        return self._sample_with_rng(
            self._rng,
            batch_size=batch_size,
            burn_in_steps=burn_in_steps,
            rollout_steps=rollout_steps,
        )

    def fixed_validation_windows(
        self,
        *,
        count: int,
        burn_in_steps: int,
        rollout_steps: int,
        seed: int,
    ) -> R2DNWindowBatch:
        """Draw deterministic validation windows independently of training RNG state."""

        if self.split != "validation":
            raise ValueError("fixed validation windows require the validation sampler")
        if seed < 0:
            raise ValueError("validation window seed must be non-negative")
        return self._sample_with_rng(
            np.random.default_rng(seed),
            batch_size=count,
            burn_in_steps=burn_in_steps,
            rollout_steps=rollout_steps,
        )

    def _sample_with_rng(
        self,
        rng: np.random.Generator,
        *,
        batch_size: int,
        burn_in_steps: int,
        rollout_steps: int,
    ) -> R2DNWindowBatch:
        if batch_size < 1 or burn_in_steps < 1 or rollout_steps < 1:
            raise ValueError("batch size, burn-in, and rollout length must be positive")
        required = burn_in_steps + rollout_steps
        eligible = tuple(
            trajectory_id
            for trajectory_id, sequence in self._trajectories.items()
            if sequence.time_steps >= required
        )
        if not eligible:
            raise ValueError(
                f"no {self.split} trajectory contains {required} transitions"
            )

        observation_windows: list[FloatArray] = []
        control_windows: list[FloatArray] = []
        trajectory_ids: list[str] = []
        start_steps: list[int] = []
        for _ in range(batch_size):
            trajectory_id = eligible[int(rng.integers(0, len(eligible)))]
            sequence = self._trajectories[trajectory_id]
            latest_start = sequence.time_steps - required
            start = int(rng.integers(0, latest_start + 1))
            stop = start + required
            observation_windows.append(sequence.observations[start : stop + 1, 0])
            control_windows.append(sequence.controls[start:stop, 0])
            trajectory_ids.append(trajectory_id)
            start_steps.append(start)

        observations = np.stack(observation_windows, axis=1)
        controls = np.stack(control_windows, axis=1)
        observations = (
            observations - self._normalization.observation_mean
        ) / self._normalization.observation_std
        controls = (
            controls - self._normalization.control_mean
        ) / self._normalization.control_std
        return R2DNWindowBatch(
            observations=observations,
            controls=controls,
            trajectory_ids=tuple(trajectory_ids),
            start_steps=tuple(start_steps),
            split=self.split,
        )
