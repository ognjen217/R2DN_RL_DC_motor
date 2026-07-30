"""Typed Phase-3 hidden-temperature observability specification."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from r2dn_dc_motor.phase2_spec import Phase2Spec, load_phase2_spec
from r2dn_dc_motor.spec import SpecValidationError

REQUIRED_GATE_CHECKS = {
    "thermal_signal_exceeds_solver_error",
    "open_loop_outputs_are_separated",
    "closed_loop_control_is_affected",
    "temperature_is_not_in_probe_features",
    "pilot_split_is_by_trajectory",
    "instantaneous_probe_is_nontrivial",
    "history_probe_recovers_temperature",
}
ALLOWED_PROBE_SIGNALS = (
    "armature_current_a",
    "angular_speed_rad_s",
    "armature_voltage_v",
)
FORBIDDEN_PROBE_SIGNALS = ("winding_temperature_c",)


@dataclass(frozen=True)
class Phase3Spec:
    """Executable source of truth for the Phase-3 go/no-go experiment."""

    schema_version: int
    phase: dict[str, Any]
    paired_experiment: dict[str, Any]
    solver_comparison: dict[str, Any]
    closed_loop: dict[str, Any]
    pilot: dict[str, Any]
    probe: dict[str, Any]
    gate: dict[str, Any]

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
        phase2: Phase2Spec | None = None,
    ) -> Phase3Spec:
        spec = cls(
            schema_version=int(raw["schema_version"]),
            phase=dict(raw["phase"]),
            paired_experiment=dict(raw["paired_experiment"]),
            solver_comparison=dict(raw["solver_comparison"]),
            closed_loop=dict(raw["closed_loop"]),
            pilot=dict(raw["pilot"]),
            probe=dict(raw["probe"]),
            gate=dict(raw["gate"]),
        )
        spec.validate(phase2 or load_phase2_spec())
        return spec

    @property
    def test_trajectory_ids(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.pilot["test_trajectory_ids"])

    @property
    def train_trajectory_ids(self) -> tuple[int, ...]:
        test = set(self.test_trajectory_ids)
        count = int(self.pilot["trajectory_count"])
        return tuple(index for index in range(count) if index not in test)

    @property
    def history_windows_s(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.probe["history_windows_s"])

    def steps(self, duration_s: float, phase2: Phase2Spec) -> int:
        """Convert an exact duration to whole frozen control periods."""

        if not math.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError("Phase-3 duration must be positive and finite")
        raw = float(duration_s) / phase2.integration_settings.control_period_s
        rounded = int(round(raw))
        if not math.isclose(raw, rounded, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("Phase-3 duration must contain whole control periods")
        return rounded

    def validate(self, phase2: Phase2Spec) -> None:
        """Reject experimental drift and hidden-temperature leakage."""

        errors: list[str] = []
        if self.schema_version != 1:
            errors.append("schema_version must be 1")
        if self.phase.get("number") != 3:
            errors.append("phase.number must be 3")
        if self.phase.get("status") != "implemented":
            errors.append("Phase 3 status must be implemented")

        errors.extend(self._validate_paired_experiment(phase2))
        errors.extend(self._validate_solver_comparison(phase2))
        errors.extend(self._validate_closed_loop(phase2))
        errors.extend(self._validate_pilot(phase2))
        errors.extend(self._validate_probe(phase2))
        errors.extend(self._validate_gate())

        if errors:
            raise SpecValidationError("\n".join(f"- {error}" for error in errors))

    def summary(self) -> str:
        return "\n".join(
            [
                "PHASE 3 SPEC: PASS",
                (
                    "paired temperatures: "
                    f"{float(self.paired_experiment['cold_temperature_c']):g} C, "
                    f"{float(self.paired_experiment['hot_temperature_c']):g} C"
                ),
                (
                    "pilot: "
                    f"{int(self.pilot['trajectory_count'])} trajectories "
                    f"({len(self.train_trajectory_ids)} train / "
                    f"{len(self.test_trajectory_ids)} test)"
                ),
                f"probe inputs: {', '.join(self.probe['allowed_signals'])}",
                f"selected history: {float(self.probe['selected_history_s']):g} s",
                "temperature: training target and evaluation only",
            ]
        )

    def _validate_paired_experiment(self, phase2: Phase2Spec) -> list[str]:
        errors: list[str] = []
        required = {
            "cold_temperature_c",
            "hot_temperature_c",
            "initial_current_a",
            "initial_speed_rad_s",
            "duration_s",
            "voltage_hold_s",
            "voltage_pattern_v",
            "load_torque_n_m",
            "tracking_reference_rad_s",
        }
        if set(self.paired_experiment) != required:
            return ["paired-experiment parameter catalog changed"]

        pair = self.paired_experiment
        numeric_names = required - {"voltage_pattern_v"}
        for name in numeric_names:
            if not isinstance(pair[name], int | float) or not math.isfinite(
                float(pair[name])
            ):
                errors.append(f"paired_experiment.{name} must be finite")
        temperature = phase2.limits["winding_temperature_c"]
        cold = float(pair["cold_temperature_c"])
        hot = float(pair["hot_temperature_c"])
        if not temperature.minimum <= cold < hot <= temperature.maximum:
            errors.append("paired temperatures must be ordered and inside plant limits")
        if not math.isclose(
            cold,
            phase2.motor_parameters.ambient_temperature_c,
            abs_tol=1e-12,
        ):
            errors.append("cold paired trajectory must start at ambient temperature")
        if not math.isclose(
            hot,
            float(phase2.validation["hot_temperature_c"]),
            abs_tol=1e-12,
        ):
            errors.append("hot paired trajectory drifted from Phase 2")

        for name, limit_name in (
            ("initial_current_a", "armature_current_a"),
            ("initial_speed_rad_s", "angular_speed_rad_s"),
            ("tracking_reference_rad_s", "angular_speed_rad_s"),
        ):
            value = float(pair[name])
            bounds = phase2.limits[limit_name]
            if not bounds.minimum <= value <= bounds.maximum:
                errors.append(f"paired {name} lies outside {limit_name} limits")

        voltage = phase2.limits["armature_voltage_v"]
        pattern = tuple(float(value) for value in pair["voltage_pattern_v"])
        if not pattern:
            errors.append("paired voltage pattern must not be empty")
        if any(not voltage.minimum <= value <= voltage.maximum for value in pattern):
            errors.append("paired voltage pattern exceeds actuator limits")
        errors.extend(
            self._whole_period_errors(
                phase2,
                {
                    "paired duration": float(pair["duration_s"]),
                    "paired voltage hold": float(pair["voltage_hold_s"]),
                },
            )
        )
        return errors

    def _validate_solver_comparison(self, phase2: Phase2Spec) -> list[str]:
        errors: list[str] = []
        if set(self.solver_comparison) != {"fine_step_divisor", "output_scales"}:
            return ["solver-comparison parameter catalog changed"]
        divisor = int(self.solver_comparison["fine_step_divisor"])
        if (
            not isinstance(self.solver_comparison["fine_step_divisor"], int)
            or divisor < 2
        ):
            errors.append("fine solver must use at least twice as many RK4 steps")
        scales = tuple(float(value) for value in self.solver_comparison["output_scales"])
        if any(not math.isfinite(value) or value <= 0.0 for value in scales):
            errors.append("solver-comparison output scales must be positive and finite")
        expected = (
            phase2.limits["armature_current_a"].maximum,
            phase2.limits["angular_speed_rad_s"].maximum,
        )
        if scales != expected:
            errors.append("solver-comparison output scales drifted from plant limits")
        return errors

    def _validate_closed_loop(self, phase2: Phase2Spec) -> list[str]:
        errors: list[str] = []
        required = {
            "controller",
            "reference_rad_s",
            "duration_s",
            "proportional_gain_v_per_rad_s",
            "integral_gain_v_per_rad",
            "anti_windup",
        }
        if set(self.closed_loop) != required:
            return ["closed-loop parameter catalog changed"]
        if self.closed_loop["controller"] != "diagnostic_speed_pi":
            errors.append("Phase 3 permits only the diagnostic speed PI")
        if self.closed_loop["anti_windup"] != "conditional_integration":
            errors.append("diagnostic PI must use conditional-integration anti-windup")
        for name in (
            "reference_rad_s",
            "duration_s",
            "proportional_gain_v_per_rad_s",
            "integral_gain_v_per_rad",
        ):
            value = float(self.closed_loop[name])
            if not math.isfinite(value) or value <= 0.0:
                errors.append(f"closed_loop.{name} must be positive")
        reference = float(self.closed_loop["reference_rad_s"])
        speed = phase2.limits["angular_speed_rad_s"]
        if not 0.0 < reference < speed.maximum:
            errors.append("closed-loop reference must be positive and inside speed limits")
        errors.extend(
            self._whole_period_errors(
                phase2,
                {"closed-loop duration": float(self.closed_loop["duration_s"])},
            )
        )
        return errors

    def _validate_pilot(self, phase2: Phase2Spec) -> list[str]:
        errors: list[str] = []
        required = {
            "trajectory_count",
            "test_trajectory_ids",
            "seed_base",
            "duration_s",
            "initial_temperature_c",
            "initial_current_a",
            "initial_speed_rad_s",
            "load_torque_n_m",
            "voltage_hold_s",
            "voltage_levels_v",
            "split_policy",
            "raw_storage",
        }
        if set(self.pilot) != required:
            return ["pilot parameter catalog changed"]

        count = int(self.pilot["trajectory_count"])
        test_ids = self.test_trajectory_ids
        if count < 8:
            errors.append("Phase-3 pilot requires at least eight trajectories")
        if (
            not test_ids
            or len(set(test_ids)) != len(test_ids)
            or any(index < 0 or index >= count for index in test_ids)
        ):
            errors.append("test trajectory IDs must be unique and inside the pilot")
        if len(self.train_trajectory_ids) <= len(test_ids):
            errors.append("pilot must retain more training than test trajectories")
        if self.pilot["split_policy"] != "whole_trajectory_stratified_temperature":
            errors.append("pilot split must be stratified and performed by whole trajectory")
        if self.pilot["raw_storage"] != "in_memory_diagnostic_only":
            errors.append("Phase 3 must not create the final dataset")
        for name in (
            "duration_s",
            "initial_current_a",
            "initial_speed_rad_s",
            "load_torque_n_m",
            "voltage_hold_s",
        ):
            if not isinstance(self.pilot[name], int | float) or not math.isfinite(
                float(self.pilot[name])
            ):
                errors.append(f"pilot.{name} must be finite")

        low, high = _pair(self.pilot["initial_temperature_c"])
        reset_low, reset_high = phase2.reset_ranges.temperature_c
        if not (math.isclose(low, reset_low) and math.isclose(high, reset_high)):
            errors.append("pilot temperature range drifted from Phase-2 reset domain")

        voltage = phase2.limits["armature_voltage_v"]
        levels = tuple(float(value) for value in self.pilot["voltage_levels_v"])
        if len(levels) < 3 or 0.0 not in levels:
            errors.append("pilot excitation requires at least three levels including zero")
        if any(not voltage.minimum <= value <= voltage.maximum for value in levels):
            errors.append("pilot voltage level exceeds actuator limits")
        errors.extend(
            self._whole_period_errors(
                phase2,
                {
                    "pilot duration": float(self.pilot["duration_s"]),
                    "pilot voltage hold": float(self.pilot["voltage_hold_s"]),
                },
            )
        )
        return errors

    def _validate_probe(self, phase2: Phase2Spec) -> list[str]:
        errors: list[str] = []
        required = {
            "name",
            "allowed_signals",
            "forbidden_signals",
            "history_windows_s",
            "selected_history_s",
            "sample_stride_s",
            "minimum_informative_current_a",
            "ridge_l2",
            "temperature_use",
        }
        if set(self.probe) != required:
            return ["probe parameter catalog changed"]
        if tuple(self.probe["allowed_signals"]) != ALLOWED_PROBE_SIGNALS:
            errors.append("probe inputs drifted from temperature-free [i, omega, u]")
        if tuple(self.probe["forbidden_signals"]) != FORBIDDEN_PROBE_SIGNALS:
            errors.append("hidden temperature must remain explicitly forbidden as a feature")
        if any(
            "temperature" in str(signal).lower()
            for signal in self.probe["allowed_signals"]
        ):
            errors.append("hidden temperature leaked into probe features")
        if self.probe["temperature_use"] != "training_target_and_evaluation_only":
            errors.append("probe temperature-use policy changed")
        if self.probe["name"] != "electrical_residual_history_ridge":
            errors.append("diagnostic probe definition changed")

        windows = self.history_windows_s
        selected = float(self.probe["selected_history_s"])
        if not windows or tuple(sorted(set(windows))) != windows:
            errors.append("probe history windows must be unique and increasing")
        if selected not in windows:
            errors.append("selected probe history must be one of the evaluated windows")
        durations = {
            "probe sample stride": float(self.probe["sample_stride_s"]),
            **{f"probe history {index}": value for index, value in enumerate(windows)},
        }
        errors.extend(self._whole_period_errors(phase2, durations))
        if selected >= float(self.pilot["duration_s"]):
            errors.append("selected history must be shorter than the pilot trajectory")
        if float(self.probe["minimum_informative_current_a"]) <= 0.0:
            errors.append("minimum informative current must be positive")
        ridge_l2 = float(self.probe["ridge_l2"])
        if not math.isfinite(ridge_l2) or ridge_l2 <= 0.0:
            errors.append("probe ridge regularization must be positive")
        return errors

    def _validate_gate(self) -> list[str]:
        errors: list[str] = []
        required_fields = {
            "name",
            "required_checks",
            "minimum_signal_to_solver_ratio",
            "minimum_max_current_difference_a",
            "minimum_max_speed_difference_rad_s",
            "minimum_max_acceleration_difference_rad_s2",
            "minimum_relative_resistance_difference",
            "minimum_tracking_iae_relative_difference",
            "minimum_control_effort_relative_difference",
            "minimum_instantaneous_probe_mae_c",
            "maximum_history_probe_mae_c",
            "minimum_history_mae_improvement_fraction",
            "minimum_test_samples",
        }
        if set(self.gate) != required_fields:
            return ["Gate-1 parameter catalog changed"]
        if self.gate["name"] != "gate_1_hidden_temperature_observability":
            errors.append("Phase 3 must implement Gate 1 observability")
        if set(self.gate["required_checks"]) != REQUIRED_GATE_CHECKS:
            errors.append("Gate-1 validation check catalog changed")
        for name in required_fields - {"name", "required_checks"}:
            value = float(self.gate[name])
            if not math.isfinite(value) or value <= 0.0:
                errors.append(f"gate.{name} must be positive")
        for name in (
            "minimum_relative_resistance_difference",
            "minimum_tracking_iae_relative_difference",
            "minimum_control_effort_relative_difference",
            "minimum_history_mae_improvement_fraction",
        ):
            if float(self.gate[name]) >= 1.0:
                errors.append(f"gate.{name} must be below one")
        return errors

    @staticmethod
    def _whole_period_errors(
        phase2: Phase2Spec,
        durations: dict[str, float],
    ) -> list[str]:
        errors: list[str] = []
        period = phase2.integration_settings.control_period_s
        for name, duration in durations.items():
            if not math.isfinite(duration) or duration <= 0.0:
                errors.append(f"{name} must contain positive whole control periods")
                continue
            raw = duration / period
            if not math.isclose(
                raw,
                round(raw),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                errors.append(f"{name} must contain positive whole control periods")
        return errors


def _pair(value: Any) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError("expected a two-element range")
    return float(value[0]), float(value[1])


def load_phase3_spec(
    path: Path | str | None = None,
    phase2: Phase2Spec | None = None,
) -> Phase3Spec:
    """Load and validate the canonical Phase-3 TOML file."""

    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "phase3.toml"
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    return Phase3Spec.from_dict(raw, phase2=phase2)
