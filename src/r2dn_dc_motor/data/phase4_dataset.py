"""Versioned on-disk Phase-4 dataset and temperature-safe model views."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from r2dn_dc_motor.data.sequences import ModelSequenceBatch, SequenceValidationError

FloatArray = NDArray[np.floating]
TRAJECTORY_ARRAY_NAMES = (
    "states",
    "commanded_voltages",
    "applied_voltages",
    "load_torques",
    "speed_references",
)


@dataclass(frozen=True)
class RawPhase4Trajectory:
    """One complete FULL trajectory with evaluation-only variables retained."""

    states: FloatArray
    commanded_voltages: FloatArray
    applied_voltages: FloatArray
    load_torques: FloatArray
    speed_references: FloatArray

    def __post_init__(self) -> None:
        for name in TRAJECTORY_ARRAY_NAMES:
            value = np.asarray(getattr(self, name), dtype=np.float32)
            object.__setattr__(self, name, value)
        self.validate()

    @property
    def transitions(self) -> int:
        return int(self.applied_voltages.shape[0])

    @property
    def temperature(self) -> FloatArray:
        """Temperature is exposed only through the raw/evaluation container."""

        return self.states[:, 2:3]

    def validate(self) -> None:
        errors: list[str] = []
        if self.states.ndim != 2 or self.states.shape[1:] != (3,):
            errors.append("states must have shape (T + 1, 3)")
        for name in TRAJECTORY_ARRAY_NAMES[1:]:
            value = getattr(self, name)
            if value.ndim != 2 or value.shape[1:] != (1,):
                errors.append(f"{name} must have shape (T, 1)")
        if errors:
            raise SequenceValidationError("\n".join(errors))

        transitions = self.states.shape[0] - 1
        if transitions < 1:
            errors.append("trajectory must contain at least one transition")
        for name in TRAJECTORY_ARRAY_NAMES[1:]:
            if getattr(self, name).shape[0] != transitions:
                errors.append(f"{name} length does not match states")
        for name in TRAJECTORY_ARRAY_NAMES:
            if not np.isfinite(getattr(self, name)).all():
                errors.append(f"{name} contains NaN or infinite values")
        if errors:
            raise SequenceValidationError("\n".join(errors))

    def model_view(self) -> ModelSequenceBatch:
        """Return the only view allowed for R2DN and ISO-CAL training."""

        return ModelSequenceBatch(
            observations=self.states[:, None, :2],
            controls=self.applied_voltages[:, None, :],
        )

    def arrays(self) -> dict[str, FloatArray]:
        return {name: getattr(self, name) for name in TRAJECTORY_ARRAY_NAMES}


@dataclass(frozen=True)
class NormalizationStatistics:
    """Train-only normalization for the frozen R2DN feature order."""

    observation_mean: FloatArray
    observation_std: FloatArray
    control_mean: FloatArray
    control_std: FloatArray
    observation_count: int
    control_count: int
    fit_split: str = "train"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_mean",
            np.asarray(self.observation_mean, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "observation_std",
            np.asarray(self.observation_std, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "control_mean",
            np.asarray(self.control_mean, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "control_std",
            np.asarray(self.control_std, dtype=np.float64),
        )
        self.validate()

    def validate(self) -> None:
        if self.observation_mean.shape != (2,) or self.observation_std.shape != (2,):
            raise ValueError("observation normalization must have two features")
        if self.control_mean.shape != (1,) or self.control_std.shape != (1,):
            raise ValueError("control normalization must have one feature")
        if self.observation_count < 1 or self.control_count < 1:
            raise ValueError("normalization counts must be positive")
        if self.fit_split != "train":
            raise ValueError("normalization may only be fit on train")
        values = (
            self.observation_mean,
            self.observation_std,
            self.control_mean,
            self.control_std,
        )
        if not all(np.isfinite(value).all() for value in values):
            raise ValueError("normalization contains NaN or infinite values")
        if np.any(self.observation_std <= 0.0) or np.any(self.control_std <= 0.0):
            raise ValueError("normalization standard deviations must be positive")

    def save(self, path: Path | str) -> None:
        np.savez(
            Path(path),
            observation_mean=self.observation_mean,
            observation_std=self.observation_std,
            control_mean=self.control_mean,
            control_std=self.control_std,
            observation_count=np.asarray(self.observation_count, dtype=np.int64),
            control_count=np.asarray(self.control_count, dtype=np.int64),
            fit_split=np.asarray(self.fit_split),
        )

    @classmethod
    def load(cls, path: Path | str) -> NormalizationStatistics:
        with np.load(Path(path), allow_pickle=False) as payload:
            return cls(
                observation_mean=payload["observation_mean"],
                observation_std=payload["observation_std"],
                control_mean=payload["control_mean"],
                control_std=payload["control_std"],
                observation_count=int(payload["observation_count"]),
                control_count=int(payload["control_count"]),
                fit_split=str(payload["fit_split"]),
            )


class Phase4Dataset:
    """Read and validate a generated Phase-4 dataset directory."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self._records = {
            record["trajectory_id"]: record
            for record in self.manifest["trajectories"]
        }
        if len(self._records) != len(self.manifest["trajectories"]):
            raise ValueError("manifest contains duplicate trajectory IDs")
        self.normalization = NormalizationStatistics.load(
            self.root / self.manifest["normalization"]["path"]
        )
        self.validate_manifest_fingerprint()

    @property
    def fingerprint(self) -> str:
        return str(self.manifest["dataset_fingerprint"])

    def trajectory_ids(self, split: str | None = None) -> tuple[str, ...]:
        records = self.manifest["trajectories"]
        if split is not None:
            records = [record for record in records if record["split"] == split]
        return tuple(record["trajectory_id"] for record in records)

    def record(self, trajectory_id: str) -> dict[str, Any]:
        try:
            return dict(self._records[trajectory_id])
        except KeyError as error:
            raise KeyError(f"unknown trajectory ID: {trajectory_id}") from error

    def load_trajectory(
        self,
        trajectory_id: str,
        *,
        verify_hash: bool = True,
    ) -> RawPhase4Trajectory:
        record = self.record(trajectory_id)
        path = self.root / record["path"]
        with np.load(path, allow_pickle=False) as payload:
            trajectory = RawPhase4Trajectory(
                **{name: payload[name] for name in TRAJECTORY_ARRAY_NAMES}
            )
        if verify_hash:
            actual = trajectory_content_sha256(trajectory)
            if actual != record["content_sha256"]:
                raise ValueError(f"trajectory content hash mismatch: {trajectory_id}")
        return trajectory

    def model_view(self, trajectory_id: str) -> ModelSequenceBatch:
        """Load a trajectory through the temperature-free model boundary."""

        return self.load_trajectory(trajectory_id).model_view()

    def validate_manifest_fingerprint(self) -> None:
        expected = self.manifest.get("dataset_fingerprint")
        payload = dict(self.manifest)
        payload.pop("dataset_fingerprint", None)
        actual = canonical_sha256(payload)
        if expected != actual:
            raise ValueError("dataset manifest fingerprint mismatch")


def trajectory_content_sha256(trajectory: RawPhase4Trajectory) -> str:
    """Hash logical array content independently of ZIP metadata."""

    digest = hashlib.sha256()
    for name, value in trajectory.arrays().items():
        array = np.ascontiguousarray(value)
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
