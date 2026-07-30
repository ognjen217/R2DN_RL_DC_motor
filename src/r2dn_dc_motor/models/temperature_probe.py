"""Diagnostic, temperature-free history probe used only by Phase 3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from r2dn_dc_motor.plants import MotorParameters

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]

HISTORY_FEATURE_NAMES = (
    "resistance_least_squares_ohm",
    "resistance_median_ohm",
    "rms_current_a",
    "mean_abs_current_a",
    "mean_speed_rad_s",
    "mean_voltage_v",
)
INSTANTANEOUS_FEATURE_NAMES = (
    "armature_current_a",
    "angular_speed_rad_s",
    "previous_armature_voltage_v",
)


@dataclass(frozen=True)
class ProbeTrajectory:
    """One temperature-labelled trajectory; temperature is never a feature."""

    trajectory_id: int
    states: FloatArray
    controls_v: FloatArray

    def validate(self) -> None:
        states = np.asarray(self.states)
        controls = np.asarray(self.controls_v)
        if states.ndim != 2 or states.shape[1] != 3:
            raise ValueError("probe trajectory states must have shape (T+1, 3)")
        if controls.shape != (states.shape[0] - 1,):
            raise ValueError("probe controls must have shape (T,)")
        if not np.isfinite(states).all() or not np.isfinite(controls).all():
            raise ValueError("probe trajectory must contain only finite values")


@dataclass(frozen=True)
class ProbeSamples:
    """Feature matrices and temperature targets with trajectory provenance."""

    history_features: FloatArray
    instantaneous_features: FloatArray
    target_temperature_c: FloatArray
    trajectory_ids: IntArray

    def validate(self) -> None:
        sample_count = self.target_temperature_c.shape[0]
        if self.history_features.shape != (sample_count, len(HISTORY_FEATURE_NAMES)):
            raise ValueError("history probe feature shape is invalid")
        if self.instantaneous_features.shape != (
            sample_count,
            len(INSTANTANEOUS_FEATURE_NAMES),
        ):
            raise ValueError("instantaneous probe feature shape is invalid")
        if self.trajectory_ids.shape != (sample_count,):
            raise ValueError("probe trajectory IDs must have one entry per sample")
        arrays = (
            self.history_features,
            self.instantaneous_features,
            self.target_temperature_c,
        )
        if sample_count == 0 or any(not np.isfinite(value).all() for value in arrays):
            raise ValueError("probe samples must be non-empty and finite")


@dataclass(frozen=True)
class StandardizedRidge:
    """Small deterministic ridge regressor with train-only normalization."""

    feature_mean: FloatArray
    feature_scale: FloatArray
    target_mean: float
    target_scale: float
    weights: FloatArray

    @classmethod
    def fit(
        cls,
        features: FloatArray,
        targets: FloatArray,
        *,
        l2: float,
    ) -> StandardizedRidge:
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if x.ndim != 2 or y.shape != (x.shape[0],) or x.shape[0] < 2:
            raise ValueError("ridge training arrays have incompatible shapes")
        if (
            not np.isfinite(l2)
            or l2 <= 0.0
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
        ):
            raise ValueError("ridge inputs and regularization must be finite")

        feature_mean = np.mean(x, axis=0)
        feature_scale = np.std(x, axis=0)
        feature_scale = np.where(feature_scale < 1e-12, 1.0, feature_scale)
        target_mean = float(np.mean(y))
        target_scale = float(np.std(y))
        if target_scale < 1e-12:
            raise ValueError("ridge targets must have non-zero variance")

        normalized_x = (x - feature_mean) / feature_scale
        normalized_y = (y - target_mean) / target_scale
        gram = normalized_x.T @ normalized_x
        weights = np.linalg.solve(
            gram + float(l2) * np.eye(x.shape[1]),
            normalized_x.T @ normalized_y,
        )
        return cls(
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            target_mean=target_mean,
            target_scale=target_scale,
            weights=weights,
        )

    def predict(self, features: FloatArray) -> FloatArray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.weights.shape[0]:
            raise ValueError("ridge prediction feature shape is invalid")
        if not np.isfinite(x).all():
            raise ValueError("ridge prediction features must be finite")
        normalized = (x - self.feature_mean) / self.feature_scale
        return normalized @ self.weights * self.target_scale + self.target_mean


def build_probe_samples(
    trajectories: tuple[ProbeTrajectory, ...],
    *,
    parameters: MotorParameters,
    control_period_s: float,
    history_steps: int,
    sample_stride_steps: int,
    minimum_informative_current_a: float,
) -> ProbeSamples:
    """Build dynamic-residual history features without exposing temperature."""

    if history_steps < 2:
        raise ValueError("history probe requires at least two control intervals")
    if (
        sample_stride_steps <= 0
        or not np.isfinite(control_period_s)
        or control_period_s <= 0.0
        or not np.isfinite(minimum_informative_current_a)
        or minimum_informative_current_a <= 0.0
    ):
        raise ValueError("probe strides and current threshold must be positive")

    history_rows: list[FloatArray] = []
    instant_rows: list[FloatArray] = []
    targets: list[float] = []
    trajectory_ids: list[int] = []

    for trajectory in trajectories:
        trajectory.validate()
        states = np.asarray(trajectory.states, dtype=np.float64)
        controls = np.asarray(trajectory.controls_v, dtype=np.float64)
        final_step = controls.shape[0]
        for end in range(history_steps, final_step + 1, sample_stride_steps):
            start = end - history_steps
            current = states[start : end + 1, 0]
            speed = states[start : end + 1, 1]
            interval_voltage = controls[start:end]
            current_rate = np.diff(current) / control_period_s
            midpoint_current = 0.5 * (current[:-1] + current[1:])
            midpoint_speed = 0.5 * (speed[:-1] + speed[1:])
            rms_current = float(np.sqrt(np.mean(midpoint_current**2)))
            if rms_current < minimum_informative_current_a:
                continue

            electrical_residual = (
                interval_voltage
                - parameters.armature_inductance_h * current_rate
                - parameters.back_emf_constant_v_s_per_rad * midpoint_speed
            )
            denominator = float(np.sum(midpoint_current**2))
            resistance_least_squares = float(
                np.sum(midpoint_current * electrical_residual)
                / (denominator + 1e-12)
            )
            informative = np.abs(midpoint_current) >= minimum_informative_current_a
            if np.any(informative):
                resistance_median = float(
                    np.median(
                        electrical_residual[informative]
                        / midpoint_current[informative]
                    )
                )
            else:
                resistance_median = resistance_least_squares

            history_rows.append(
                np.asarray(
                    (
                        resistance_least_squares,
                        resistance_median,
                        rms_current,
                        float(np.mean(np.abs(midpoint_current))),
                        float(np.mean(midpoint_speed)),
                        float(np.mean(interval_voltage)),
                    ),
                    dtype=np.float64,
                )
            )
            instant_rows.append(
                np.asarray(
                    (
                        states[end, 0],
                        states[end, 1],
                        controls[end - 1],
                    ),
                    dtype=np.float64,
                )
            )
            targets.append(float(states[end, 2]))
            trajectory_ids.append(trajectory.trajectory_id)

    samples = ProbeSamples(
        history_features=np.asarray(history_rows, dtype=np.float64),
        instantaneous_features=np.asarray(instant_rows, dtype=np.float64),
        target_temperature_c=np.asarray(targets, dtype=np.float64),
        trajectory_ids=np.asarray(trajectory_ids, dtype=np.int64),
    )
    samples.validate()
    return samples
