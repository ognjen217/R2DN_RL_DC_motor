"""Typed Phase-5 contract for the two physical baseline models."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from r2dn_dc_motor.models.isothermal_calibration import (
    ALLOWED_FIT_FEATURES,
    CALIBRATED_PARAMETER_NAMES,
    FORBIDDEN_FIT_FEATURES,
)
from r2dn_dc_motor.phase2_spec import Phase2Spec, load_phase2_spec
from r2dn_dc_motor.phase4_spec import Phase4Spec, load_phase4_spec
from r2dn_dc_motor.spec import ExperimentSpec, SpecValidationError, load_phase0_spec

REQUIRED_PHASE5_CHECKS = {
    "models_share_temperature_free_interface",
    "iso_nom_matches_phase2_nominal_parameters",
    "calibration_uses_train_only_model_view",
    "checkpoint_is_single_global_and_dataset_bound",
    "evaluation_splits_never_select_parameters",
    "required_metrics_are_finite",
}
EVALUATION_SPLITS = ("validation", "id_test", "ood_test")
FIXED_PARAMETER_NAMES = (
    "armature_inductance_h",
    "back_emf_constant_v_s_per_rad",
    "torque_constant_n_m_per_a",
    "inertia_kg_m2",
    "default_load_torque_n_m",
)


@dataclass(frozen=True)
class Phase5Spec:
    """Executable source of truth for ISO-NOM/ISO-CAL calibration and evaluation."""

    schema_version: int
    phase: dict[str, Any]
    interface: dict[str, Any]
    models: dict[str, Any]
    calibration: dict[str, Any]
    evaluation: dict[str, Any]
    validation: dict[str, Any]

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
        *,
        phase0: ExperimentSpec | None = None,
        phase2: Phase2Spec | None = None,
        phase4: Phase4Spec | None = None,
    ) -> Phase5Spec:
        spec = cls(
            schema_version=int(raw["schema_version"]),
            phase=dict(raw["phase"]),
            interface=dict(raw["interface"]),
            models={name: dict(values) for name, values in raw["models"].items()},
            calibration=dict(raw["calibration"]),
            evaluation=dict(raw["evaluation"]),
            validation=dict(raw["validation"]),
        )
        phase0 = phase0 or load_phase0_spec()
        phase2 = phase2 or load_phase2_spec(phase0=phase0)
        phase4 = phase4 or load_phase4_spec(
            phase0=phase0,
            phase2=phase2,
        )
        spec.validate(phase0=phase0, phase2=phase2, phase4=phase4)
        return spec

    @property
    def resistance_bounds(self) -> tuple[float, float]:
        return _pair(self.calibration["effective_resistance_ohm_bounds"])

    @property
    def friction_bounds(self) -> tuple[float, float]:
        return _pair(
            self.calibration["viscous_friction_n_m_s_per_rad_bounds"]
        )

    @property
    def rollout_horizons_s(self) -> dict[str, float]:
        return {
            "short": float(self.evaluation["short_rollout_horizon_s"]),
            "medium": float(self.evaluation["medium_rollout_horizon_s"]),
            "long": float(self.evaluation["long_rollout_horizon_s"]),
        }

    def validate(
        self,
        *,
        phase0: ExperimentSpec,
        phase2: Phase2Spec,
        phase4: Phase4Spec,
    ) -> None:
        errors: list[str] = []
        if self.schema_version != 1:
            errors.append("schema_version must be 1")
        if self.phase.get("number") != 5:
            errors.append("phase.number must be 5")
        if self.phase.get("status") != "implemented":
            errors.append("Phase 5 status must be implemented")

        if tuple(self.interface.get("observation", ())) != ALLOWED_FIT_FEATURES[:2]:
            errors.append("Phase-5 observation order drifted from the model view")
        if tuple(self.interface.get("control", ())) != ALLOWED_FIT_FEATURES[2:]:
            errors.append("Phase-5 control order drifted from the model view")
        if tuple(self.interface.get("forbidden_fit_features", ())) != (
            FORBIDDEN_FIT_FEATURES
        ):
            errors.append("forbidden Phase-5 fit feature catalog changed")
        if self.interface.get("load_policy") != "fixed_nominal_unobserved":
            errors.append("load torque must remain an unobserved fixed nominal assumption")
        if tuple(phase4.features["model_observation"]) != ALLOWED_FIT_FEATURES[:2]:
            errors.append("Phase-5 observations drifted from Phase 4")
        if tuple(phase4.features["model_control"]) != ALLOWED_FIT_FEATURES[2:]:
            errors.append("Phase-5 control drifted from Phase 4")
        if phase0.signals.hidden_state[0] not in FORBIDDEN_FIT_FEATURES:
            errors.append("hidden temperature must be explicitly forbidden")

        errors.extend(self._validate_models())
        errors.extend(self._validate_calibration(phase2))
        errors.extend(self._validate_evaluation(phase2, phase4))
        if set(self.validation.get("required_checks", ())) != REQUIRED_PHASE5_CHECKS:
            errors.append("Phase-5 validation check catalog changed")
        if errors:
            raise SpecValidationError("\n".join(f"- {error}" for error in errors))

    def _validate_models(self) -> list[str]:
        errors: list[str] = []
        if set(self.models) != {"iso_nom", "iso_cal"}:
            return ["Phase 5 must define exactly ISO-NOM and ISO-CAL"]
        nominal = self.models["iso_nom"]
        calibrated = self.models["iso_cal"]
        if nominal.get("name") != "ISO-NOM":
            errors.append("nominal model name must be ISO-NOM")
        if calibrated.get("name") != "ISO-CAL":
            errors.append("calibrated model name must be ISO-CAL")
        if {
            nominal.get("structure"),
            calibrated.get("structure"),
        } != {"two_state_isothermal_dc_motor"}:
            errors.append("ISO-NOM and ISO-CAL must have the identical structure")
        if nominal.get("thermal_state") is not False:
            errors.append("ISO-NOM must not contain a thermal state")
        if calibrated.get("thermal_state") is not False:
            errors.append("ISO-CAL must not contain a thermal state")
        if nominal.get("parameter_source") != "phase2_nominal":
            errors.append("ISO-NOM parameters must come from Phase 2")
        if calibrated.get("parameter_source") != "single_global_train_fit":
            errors.append("ISO-CAL must use one global train fit")
        if tuple(calibrated.get("calibrated_parameters", ())) != (
            CALIBRATED_PARAMETER_NAMES
        ):
            errors.append("ISO-CAL may calibrate only effective R and b")
        if tuple(calibrated.get("fixed_parameters", ())) != FIXED_PARAMETER_NAMES:
            errors.append("ISO-CAL fixed parameter catalog changed")
        return errors

    def _validate_calibration(self, phase2: Phase2Spec) -> list[str]:
        errors: list[str] = []
        if self.calibration.get("fit_split") != "train":
            errors.append("calibration fit split must be train")
        if (
            self.calibration.get("trajectory_policy")
            != "all_whole_train_trajectories"
        ):
            errors.append("calibration must use all whole train trajectories")
        if (
            self.calibration.get("method")
            != "continuous_midpoint_linear_least_squares"
        ):
            errors.append("calibration method changed")
        if (
            self.calibration.get("selection_policy")
            != "single_locked_fit_no_validation_tuning"
        ):
            errors.append("validation/test data must not tune ISO-CAL")
        if self.calibration.get("temperature_used") is not False:
            errors.append("temperature must not enter ISO-CAL")
        if self.calibration.get("load_torque_used") is not False:
            errors.append("trajectory load torque must not enter ISO-CAL")
        minimum_energy = float(
            self.calibration.get("minimum_regressor_energy", 0.0)
        )
        if not math.isfinite(minimum_energy) or minimum_energy <= 0.0:
            errors.append("minimum regressor energy must be positive")
        for name, bounds in (
            ("effective resistance", self.resistance_bounds),
            ("viscous friction", self.friction_bounds),
        ):
            if not (0.0 < bounds[0] < bounds[1]):
                errors.append(f"{name} bounds must be positive and ordered")
        nominal = phase2.motor_parameters
        if not self.resistance_bounds[0] <= (
            nominal.reference_resistance_ohm
        ) <= self.resistance_bounds[1]:
            errors.append("effective-resistance bounds must contain ISO-NOM")
        if not self.friction_bounds[0] <= (
            nominal.viscous_friction_n_m_s_per_rad
        ) <= self.friction_bounds[1]:
            errors.append("friction bounds must contain ISO-NOM")
        return errors

    def _validate_evaluation(
        self,
        phase2: Phase2Spec,
        phase4: Phase4Spec,
    ) -> list[str]:
        errors: list[str] = []
        if tuple(self.evaluation.get("splits", ())) != EVALUATION_SPLITS:
            errors.append("Phase-5 evaluation splits must be validation/ID/OOD")
        if (
            self.evaluation.get("normalization")
            != "phase4_train_only_observation_std"
        ):
            errors.append("evaluation must reuse Phase-4 train-only normalization")
        one_step = float(self.evaluation.get("one_step_horizon_s", math.nan))
        if not math.isclose(
            one_step,
            phase2.integration_settings.control_period_s,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            errors.append("one-step horizon must equal the control period")
        horizons = self.rollout_horizons_s
        if not 0.0 < horizons["short"] < horizons["medium"] < horizons["long"]:
            errors.append("rollout horizons must be positive and increasing")
        if not all(
            horizon in phase2.time_horizons["prediction_horizons_s"]
            for horizon in horizons.values()
        ):
            errors.append("Phase-5 rollout horizons drifted from Phase 2")
        cold = float(self.evaluation.get("cold_initial_temperature_max_c", math.nan))
        hot = float(self.evaluation.get("hot_initial_temperature_min_c", math.nan))
        training_bands = phase4.domain["temperature_bands_c"]
        if cold != float(training_bands[0][1]):
            errors.append("cold threshold must equal the cold training-band maximum")
        if hot != float(training_bands[-1][0]):
            errors.append("hot threshold must equal the hot training-band minimum")
        if int(self.evaluation.get("runtime_repeats", 0)) < 1:
            errors.append("runtime repeats must be positive")
        return errors

    def summary(self) -> str:
        horizons = self.rollout_horizons_s
        return "\n".join(
            [
                "PHASE 5 SPEC: PASS",
                "models: ISO-NOM, ISO-CAL",
                "state: armature_current_a, angular_speed_rad_s",
                "control: armature_voltage_v",
                "ISO-CAL fit: effective R and b",
                "fit data: every whole train trajectory",
                "forbidden during fit: temperature, load, reference, commanded voltage",
                "selection: one locked global fit; no validation/test tuning",
                (
                    "rollouts [s]: "
                    f"short={horizons['short']:g}, "
                    f"medium={horizons['medium']:g}, "
                    f"long={horizons['long']:g}"
                ),
            ]
        )


def _pair(value: Any) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise SpecValidationError("expected a two-element Phase-5 bound")
    return float(value[0]), float(value[1])


def load_phase5_spec(
    path: Path | str | None = None,
    *,
    phase0: ExperimentSpec | None = None,
    phase2: Phase2Spec | None = None,
    phase4: Phase4Spec | None = None,
) -> Phase5Spec:
    """Load and validate the canonical Phase-5 TOML contract."""

    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "phase5.toml"
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    return Phase5Spec.from_dict(
        raw,
        phase0=phase0,
        phase2=phase2,
        phase4=phase4,
    )
