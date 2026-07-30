"""Temperature-free global calibration and checkpointing for ISO-CAL."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from r2dn_dc_motor.data import Phase4Dataset
from r2dn_dc_motor.phase2_spec import Phase2Spec
from r2dn_dc_motor.plants.isothermal import IsothermalParameters

CALIBRATED_PARAMETER_NAMES = (
    "effective_resistance_ohm",
    "viscous_friction_n_m_s_per_rad",
)
ALLOWED_FIT_FEATURES = (
    "armature_current_a",
    "angular_speed_rad_s",
    "armature_voltage_v",
)
FORBIDDEN_FIT_FEATURES = (
    "winding_temperature_c",
    "load_torque_n_m",
    "angular_speed_reference_rad_s",
    "commanded_armature_voltage_v",
)


@dataclass(frozen=True)
class CalibrationSufficientStatistics:
    """Streaming normal equations accumulated over all train transitions."""

    transition_count: int
    trajectory_count: int
    resistance_numerator: float
    resistance_denominator: float
    friction_numerator: float
    friction_denominator: float

    def validate(self, minimum_regressor_energy: float) -> None:
        values = asdict(self)
        if self.transition_count < 1 or self.trajectory_count < 1:
            raise ValueError("calibration requires train trajectories and transitions")
        if not all(math.isfinite(float(value)) for value in values.values()):
            raise ValueError("calibration statistics must be finite")
        if self.resistance_denominator <= minimum_regressor_energy:
            raise ValueError("insufficient current excitation to calibrate resistance")
        if self.friction_denominator <= minimum_regressor_energy:
            raise ValueError("insufficient speed excitation to calibrate friction")


@dataclass(frozen=True)
class IsothermalCalibrationCheckpoint:
    """One global ISO-CAL parameter set bound to one Phase-4 dataset."""

    schema_version: int
    model_name: str
    dataset_fingerprint: str
    fit_split: str
    fit_trajectory_ids: tuple[str, ...]
    allowed_fit_features: tuple[str, ...]
    forbidden_fit_features: tuple[str, ...]
    calibrated_parameters: tuple[str, ...]
    parameters: IsothermalParameters
    method: str
    selection_policy: str
    temperature_used: bool
    load_torque_used: bool
    resistance_bounds: tuple[float, float]
    friction_bounds: tuple[float, float]
    minimum_regressor_energy: float
    sufficient_statistics: CalibrationSufficientStatistics

    def validate(self, dataset: Phase4Dataset | None = None) -> None:
        errors: list[str] = []
        if self.schema_version != 1:
            errors.append("checkpoint schema_version must be 1")
        if self.model_name != "ISO-CAL":
            errors.append("checkpoint model_name must be ISO-CAL")
        if self.fit_split != "train":
            errors.append("ISO-CAL may only fit the train split")
        if self.allowed_fit_features != ALLOWED_FIT_FEATURES:
            errors.append("ISO-CAL fit feature order changed")
        if self.forbidden_fit_features != FORBIDDEN_FIT_FEATURES:
            errors.append("ISO-CAL forbidden fit feature catalog changed")
        if self.calibrated_parameters != CALIBRATED_PARAMETER_NAMES:
            errors.append("ISO-CAL calibrated parameter catalog changed")
        if self.temperature_used or self.load_torque_used:
            errors.append("temperature and trajectory load torque must not enter calibration")
        for name, bounds in (
            ("resistance", self.resistance_bounds),
            ("friction", self.friction_bounds),
        ):
            if not 0.0 < bounds[0] < bounds[1]:
                errors.append(f"{name} bounds must be positive and ordered")
        if (
            not math.isfinite(self.minimum_regressor_energy)
            or self.minimum_regressor_energy <= 0.0
        ):
            errors.append("minimum regressor energy must be positive")
        if len(set(self.fit_trajectory_ids)) != len(self.fit_trajectory_ids):
            errors.append("fit trajectory IDs must be unique")
        if dataset is not None:
            expected = dataset.trajectory_ids("train")
            if self.dataset_fingerprint != dataset.fingerprint:
                errors.append("checkpoint dataset fingerprint mismatch")
            if self.fit_trajectory_ids != expected:
                errors.append("checkpoint must use every whole train trajectory exactly once")
        try:
            self.parameters.validate()
            self.sufficient_statistics.validate(self.minimum_regressor_energy)
        except ValueError as error:
            errors.append(str(error))
        if errors:
            raise ValueError("\n".join(f"- {error}" for error in errors))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fit_trajectory_ids"] = list(self.fit_trajectory_ids)
        payload["allowed_fit_features"] = list(self.allowed_fit_features)
        payload["forbidden_fit_features"] = list(self.forbidden_fit_features)
        payload["calibrated_parameters"] = list(self.calibrated_parameters)
        payload["resistance_bounds"] = list(self.resistance_bounds)
        payload["friction_bounds"] = list(self.friction_bounds)
        return payload

    def save(self, path: Path | str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        dataset: Phase4Dataset | None = None,
    ) -> IsothermalCalibrationCheckpoint:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["fit_trajectory_ids"] = tuple(payload["fit_trajectory_ids"])
        payload["allowed_fit_features"] = tuple(payload["allowed_fit_features"])
        payload["forbidden_fit_features"] = tuple(payload["forbidden_fit_features"])
        payload["calibrated_parameters"] = tuple(payload["calibrated_parameters"])
        payload["resistance_bounds"] = tuple(payload["resistance_bounds"])
        payload["friction_bounds"] = tuple(payload["friction_bounds"])
        payload["parameters"] = IsothermalParameters(**payload["parameters"])
        payload["sufficient_statistics"] = CalibrationSufficientStatistics(
            **payload["sufficient_statistics"]
        )
        checkpoint = cls(**payload)
        checkpoint.validate(dataset)
        return checkpoint


def nominal_isothermal_parameters(phase2: Phase2Spec) -> IsothermalParameters:
    """Build ISO-NOM from the frozen Phase-2 non-thermal parameters."""

    p = phase2.motor_parameters
    return IsothermalParameters(
        armature_inductance_h=p.armature_inductance_h,
        effective_resistance_ohm=p.reference_resistance_ohm,
        back_emf_constant_v_s_per_rad=p.back_emf_constant_v_s_per_rad,
        torque_constant_n_m_per_a=p.torque_constant_n_m_per_a,
        inertia_kg_m2=p.inertia_kg_m2,
        viscous_friction_n_m_s_per_rad=p.viscous_friction_n_m_s_per_rad,
        default_load_torque_n_m=p.default_load_torque_n_m,
    )


def fit_global_isothermal_parameters(
    dataset: Phase4Dataset,
    phase2: Phase2Spec,
    *,
    resistance_bounds: tuple[float, float],
    friction_bounds: tuple[float, float],
    minimum_regressor_energy: float,
    forbidden_fit_features: tuple[str, ...],
    method: str,
    selection_policy: str,
) -> IsothermalCalibrationCheckpoint:
    """Fit effective ``R`` and ``b`` from the temperature-free train view.

    The midpoint discretization turns each differential equation into an
    independent scalar least-squares problem. Statistics are streamed over all
    whole train trajectories, so the final profile does not need to be loaded
    into memory at once.
    """

    nominal = nominal_isothermal_parameters(phase2)
    dt = phase2.integration_settings.control_period_s
    r_num = r_den = b_num = b_den = 0.0
    transitions = 0
    train_ids = dataset.trajectory_ids("train")

    for trajectory_id in train_ids:
        view = dataset.model_view(trajectory_id)
        observations = np.asarray(view.observations[:, 0, :], dtype=np.float64)
        controls = np.asarray(view.controls[:, 0, 0], dtype=np.float64)
        left = observations[:-1]
        right = observations[1:]
        midpoint = 0.5 * (left + right)
        derivative = (right - left) / dt

        current = midpoint[:, 0]
        speed = midpoint[:, 1]
        resistance_target = (
            controls
            - nominal.back_emf_constant_v_s_per_rad * speed
            - nominal.armature_inductance_h * derivative[:, 0]
        )
        friction_target = (
            nominal.torque_constant_n_m_per_a * current
            - nominal.default_load_torque_n_m
            - nominal.inertia_kg_m2 * derivative[:, 1]
        )
        r_num += float(current @ resistance_target)
        r_den += float(current @ current)
        b_num += float(speed @ friction_target)
        b_den += float(speed @ speed)
        transitions += controls.shape[0]

    statistics = CalibrationSufficientStatistics(
        transition_count=transitions,
        trajectory_count=len(train_ids),
        resistance_numerator=r_num,
        resistance_denominator=r_den,
        friction_numerator=b_num,
        friction_denominator=b_den,
    )
    statistics.validate(minimum_regressor_energy)
    resistance = _bounded_ratio(r_num, r_den, resistance_bounds)
    friction = _bounded_ratio(b_num, b_den, friction_bounds)
    parameters = IsothermalParameters(
        **{
            **asdict(nominal),
            "effective_resistance_ohm": resistance,
            "viscous_friction_n_m_s_per_rad": friction,
        }
    )
    checkpoint = IsothermalCalibrationCheckpoint(
        schema_version=1,
        model_name="ISO-CAL",
        dataset_fingerprint=dataset.fingerprint,
        fit_split="train",
        fit_trajectory_ids=train_ids,
        allowed_fit_features=ALLOWED_FIT_FEATURES,
        forbidden_fit_features=forbidden_fit_features,
        calibrated_parameters=CALIBRATED_PARAMETER_NAMES,
        parameters=parameters,
        method=method,
        selection_policy=selection_policy,
        temperature_used=False,
        load_torque_used=False,
        resistance_bounds=resistance_bounds,
        friction_bounds=friction_bounds,
        minimum_regressor_energy=minimum_regressor_energy,
        sufficient_statistics=statistics,
    )
    checkpoint.validate(dataset)
    return checkpoint


def _bounded_ratio(
    numerator: float,
    denominator: float,
    bounds: tuple[float, float],
) -> float:
    low, high = bounds
    if not 0.0 < low < high:
        raise ValueError("calibration bounds must be positive and ordered")
    return float(np.clip(numerator / denominator, low, high))
