"""Typed Phase-2 electrothermal plant specification and invariant checks."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from r2dn_dc_motor.plants import (
    ElectrothermalDCMotor,
    IntegrationSettings,
    MotorLimits,
    MotorParameters,
    MotorState,
    ResetRanges,
)
from r2dn_dc_motor.spec import ExperimentSpec, Range, SpecValidationError, load_phase0_spec

REQUIRED_GATE_CHECKS = {
    "zero_input_returns_to_rest",
    "current_heats_toward_equilibrium",
    "zero_current_cools_to_ambient",
    "resistance_increases_with_temperature",
    "hot_motor_draws_less_current",
    "rk4_step_converges",
    "alpha_zero_is_isothermal",
    "thermal_power_balance_is_consistent",
}


@dataclass(frozen=True)
class Phase2Spec:
    """Executable source of truth for the FULL plant."""

    schema_version: int
    phase: dict[str, Any]
    state: dict[str, Any]
    parameters: dict[str, dict[str, float]]
    limits: dict[str, Range]
    reset: dict[str, Any]
    numerics: dict[str, Any]
    time_horizons: dict[str, Any]
    validation: dict[str, Any]
    gate: dict[str, Any]

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
        phase0: ExperimentSpec | None = None,
    ) -> Phase2Spec:
        spec = cls(
            schema_version=int(raw["schema_version"]),
            phase=dict(raw["phase"]),
            state=dict(raw["state"]),
            parameters={
                group: {key: float(value) for key, value in values.items()}
                for group, values in raw["parameters"].items()
            },
            limits={
                name: Range.from_dict(value)
                for name, value in raw["limits"].items()
            },
            reset=dict(raw["reset"]),
            numerics=dict(raw["numerics"]),
            time_horizons=dict(raw["time_horizons"]),
            validation=dict(raw["validation"]),
            gate=dict(raw["gate"]),
        )
        spec.validate(phase0 or load_phase0_spec())
        return spec

    @property
    def motor_parameters(self) -> MotorParameters:
        electrical = self.parameters["electrical"]
        mechanical = self.parameters["mechanical"]
        thermal = self.parameters["thermal"]
        return MotorParameters(
            armature_inductance_h=electrical["armature_inductance_h"],
            reference_resistance_ohm=electrical["reference_resistance_ohm"],
            reference_temperature_c=electrical["reference_temperature_c"],
            resistance_temperature_coefficient_per_c=electrical[
                "resistance_temperature_coefficient_per_c"
            ],
            back_emf_constant_v_s_per_rad=electrical[
                "back_emf_constant_v_s_per_rad"
            ],
            torque_constant_n_m_per_a=electrical["torque_constant_n_m_per_a"],
            inertia_kg_m2=mechanical["inertia_kg_m2"],
            viscous_friction_n_m_s_per_rad=mechanical[
                "viscous_friction_n_m_s_per_rad"
            ],
            default_load_torque_n_m=mechanical["default_load_torque_n_m"],
            thermal_capacitance_j_per_c=thermal["thermal_capacitance_j_per_c"],
            thermal_resistance_c_per_w=thermal["thermal_resistance_c_per_w"],
            ambient_temperature_c=thermal["ambient_temperature_c"],
        )

    @property
    def motor_limits(self) -> MotorLimits:
        voltage = self.limits["armature_voltage_v"]
        current = self.limits["armature_current_a"]
        speed = self.limits["angular_speed_rad_s"]
        temperature = self.limits["winding_temperature_c"]
        return MotorLimits(
            minimum_voltage_v=voltage.minimum,
            maximum_voltage_v=voltage.maximum,
            minimum_current_a=current.minimum,
            maximum_current_a=current.maximum,
            minimum_speed_rad_s=speed.minimum,
            maximum_speed_rad_s=speed.maximum,
            minimum_temperature_c=temperature.minimum,
            maximum_temperature_c=temperature.maximum,
        )

    @property
    def integration_settings(self) -> IntegrationSettings:
        return IntegrationSettings(
            control_period_s=float(self.numerics["control_period_s"]),
            integrator_step_s=float(self.numerics["integrator_step_s"]),
        )

    @property
    def default_state(self) -> MotorState:
        return MotorState(
            current_a=float(self.reset["default_current_a"]),
            speed_rad_s=float(self.reset["default_speed_rad_s"]),
            temperature_c=float(self.reset["default_temperature_c"]),
        )

    @property
    def reset_ranges(self) -> ResetRanges:
        return ResetRanges(
            current_a=_pair(self.reset["random_current_a"]),
            speed_rad_s=_pair(self.reset["random_speed_rad_s"]),
            temperature_c=_pair(self.reset["random_temperature_c"]),
        )

    def build_plant(
        self,
        *,
        parameters: MotorParameters | None = None,
        integration: IntegrationSettings | None = None,
    ) -> ElectrothermalDCMotor:
        """Build the canonical FULL plant or an explicit validation variant."""

        return ElectrothermalDCMotor(
            parameters=parameters or self.motor_parameters,
            limits=self.motor_limits,
            integration=integration or self.integration_settings,
            default_state=self.default_state,
            reset_ranges=self.reset_ranges,
        )

    def validate(self, phase0: ExperimentSpec) -> None:
        """Validate Phase 2 against frozen experimental and numerical contracts."""

        errors: list[str] = []
        if self.schema_version != 1:
            errors.append("schema_version must be 1")
        if self.phase.get("number") != 2:
            errors.append("phase.number must be 2")
        if self.phase.get("status") != "implemented":
            errors.append("Phase 2 status must be implemented")

        if tuple(self.state.get("order", ())) != phase0.signals.state:
            errors.append("FULL state order drifted from Phase 0")
        if tuple(self.state.get("measured", ())) != phase0.signals.plant_output:
            errors.append("measured state order drifted from Phase 0")
        if tuple(self.state.get("hidden", ())) != phase0.signals.hidden_state:
            errors.append("hidden state order drifted from Phase 0")

        if set(self.parameters) != {"electrical", "mechanical", "thermal"}:
            errors.append("motor parameter groups must be electrical, mechanical, and thermal")
        else:
            errors.extend(self._validate_parameters())

        expected_limit_names = {
            "armature_voltage_v",
            "armature_current_a",
            "angular_speed_rad_s",
            "winding_temperature_c",
        }
        if set(self.limits) != expected_limit_names:
            errors.append("Phase-2 limits are incomplete or contain unexpected entries")
        for name, bounds in self.limits.items():
            errors.extend(bounds.validate(name))
            phase0_bounds = phase0.limits.get(name)
            if phase0_bounds and bounds != phase0_bounds:
                errors.append(f"{name} limits drifted from Phase 0")

        errors.extend(self._validate_numerics(phase0))
        errors.extend(self._validate_reset())
        errors.extend(self._validate_horizons())
        errors.extend(self._validate_validation_plan())

        if self.gate.get("name") != "gate_0_physical_plausibility":
            errors.append("Phase 2 must implement Gate 0 physical plausibility")
        if set(self.gate.get("required_checks", ())) != REQUIRED_GATE_CHECKS:
            errors.append("Gate 0 validation check catalog changed")

        if not errors:
            try:
                plant = self.build_plant()
                equilibrium = plant.constant_current_thermal_equilibrium_c(
                    float(self.validation["validation_current_a"])
                )
                temperature_limits = self.limits["winding_temperature_c"]
                if not (
                    temperature_limits.minimum
                    < equilibrium
                    < temperature_limits.maximum
                ):
                    errors.append(
                        "validation current equilibrium must stay inside temperature limits"
                    )
            except (KeyError, TypeError, ValueError) as error:
                errors.append(f"plant configuration cannot be constructed: {error}")

        if errors:
            raise SpecValidationError("\n".join(f"- {error}" for error in errors))

    def time_constants_s(self) -> dict[str, float]:
        return self.build_plant().time_constants_s()

    def summary(self) -> str:
        constants = self.time_constants_s()
        p = self.motor_parameters
        integration = self.integration_settings
        return "\n".join(
            [
                "PHASE 2 SPEC: PASS",
                "plant: FULL electrothermal DC motor",
                "state: armature_current_a, angular_speed_rad_s, winding_temperature_c",
                (
                    "parameters: "
                    f"L={p.armature_inductance_h:g} H, "
                    f"R0={p.reference_resistance_ohm:g} ohm, "
                    f"J={p.inertia_kg_m2:g} kg m^2, "
                    f"Cth={p.thermal_capacitance_j_per_c:g} J/C"
                ),
                (
                    "time constants: "
                    f"electrical={constants['electrical']:.6g} s, "
                    f"mechanical={constants['mechanical']:.6g} s, "
                    f"thermal={constants['thermal']:.6g} s"
                ),
                (
                    "integration: "
                    f"Ts={integration.control_period_s:g} s, "
                    f"h={integration.integrator_step_s:g} s, "
                    f"substeps={integration.substeps_per_control}"
                ),
                f"pilot burn-in: {float(self.time_horizons['pilot_burn_in_s']):g} s",
                f"core episode: {float(self.time_horizons['core_episode_s']):g} s",
            ]
        )

    def _validate_parameters(self) -> list[str]:
        errors: list[str] = []
        required = {
            "electrical": {
                "armature_inductance_h",
                "reference_resistance_ohm",
                "reference_temperature_c",
                "resistance_temperature_coefficient_per_c",
                "back_emf_constant_v_s_per_rad",
                "torque_constant_n_m_per_a",
            },
            "mechanical": {
                "inertia_kg_m2",
                "viscous_friction_n_m_s_per_rad",
                "default_load_torque_n_m",
            },
            "thermal": {
                "thermal_capacitance_j_per_c",
                "thermal_resistance_c_per_w",
                "ambient_temperature_c",
            },
        }
        for group, fields in required.items():
            actual = set(self.parameters[group])
            if actual != fields:
                errors.append(f"{group} parameter catalog changed")
            for name, value in self.parameters[group].items():
                if not math.isfinite(value):
                    errors.append(f"{group}.{name} must be finite")

        if errors:
            return errors

        p = self.motor_parameters
        positive = {
            "armature_inductance_h": p.armature_inductance_h,
            "reference_resistance_ohm": p.reference_resistance_ohm,
            "back_emf_constant_v_s_per_rad": p.back_emf_constant_v_s_per_rad,
            "torque_constant_n_m_per_a": p.torque_constant_n_m_per_a,
            "inertia_kg_m2": p.inertia_kg_m2,
            "thermal_capacitance_j_per_c": p.thermal_capacitance_j_per_c,
            "thermal_resistance_c_per_w": p.thermal_resistance_c_per_w,
        }
        for name, value in positive.items():
            if value <= 0.0:
                errors.append(f"{name} must be positive")
        if p.viscous_friction_n_m_s_per_rad < 0.0:
            errors.append("viscous friction must be non-negative")
        if p.resistance_temperature_coefficient_per_c <= 0.0:
            errors.append("temperature coefficient must be positive in the core FULL model")
        if not math.isclose(
            p.torque_constant_n_m_per_a,
            p.back_emf_constant_v_s_per_rad,
            rel_tol=1e-12,
        ):
            errors.append("torque and back-EMF constants must match in coherent SI units")

        temperature = self.limits.get("winding_temperature_c")
        if temperature:
            if not temperature.minimum <= p.reference_temperature_c <= temperature.maximum:
                errors.append("resistance reference temperature lies outside plant limits")
            if not temperature.minimum <= p.ambient_temperature_c <= temperature.maximum:
                errors.append("ambient temperature lies outside plant limits")
        return errors

    def _validate_numerics(self, phase0: ExperimentSpec) -> list[str]:
        errors: list[str] = []
        if self.numerics.get("method") != "classical_rk4":
            errors.append("FULL plant must use classical RK4")
        if self.numerics.get("voltage_policy") != "clip_to_limits":
            errors.append("voltage must be clipped to the frozen actuator limits")
        if (
            self.numerics.get("state_limit_policy")
            != "terminate_without_state_clipping"
        ):
            errors.append("state-limit violations must terminate without state clipping")
        if self.numerics.get("nonfinite_policy") != "terminate":
            errors.append("non-finite states must terminate the rollout")

        control_period = float(self.numerics.get("control_period_s", 0.0))
        integrator_step = float(self.numerics.get("integrator_step_s", 0.0))
        if not math.isclose(
            control_period,
            float(phase0.temporal_representation["control_period_s"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            errors.append("control period drifted from Phase 0")
        if not math.isclose(
            integrator_step,
            float(phase0.temporal_representation["plant_integrator_step_s"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            errors.append("integrator step drifted from Phase 0")
        try:
            integration = self.integration_settings
            _ = integration.substeps_per_control
        except (KeyError, TypeError, ValueError) as error:
            errors.append(str(error))

        if not errors and set(self.parameters) == {"electrical", "mechanical", "thermal"}:
            electrical_tau = (
                self.motor_parameters.armature_inductance_h
                / self.motor_parameters.reference_resistance_ohm
            )
            if integrator_step > electrical_tau / 20.0:
                errors.append("RK4 step must resolve the electrical time constant by >=20 steps")
        return errors

    def _validate_reset(self) -> list[str]:
        errors: list[str] = []
        required = {
            "default_current_a",
            "default_speed_rad_s",
            "default_temperature_c",
            "random_current_a",
            "random_speed_rad_s",
            "random_temperature_c",
        }
        if set(self.reset) != required:
            errors.append("reset configuration catalog changed")
            return errors

        mappings = (
            ("random_current_a", "armature_current_a"),
            ("random_speed_rad_s", "angular_speed_rad_s"),
            ("random_temperature_c", "winding_temperature_c"),
        )
        for reset_name, limit_name in mappings:
            try:
                low, high = _pair(self.reset[reset_name])
            except (TypeError, ValueError) as error:
                errors.append(f"{reset_name}: {error}")
                continue
            if low >= high:
                errors.append(f"{reset_name} must be an increasing range")
            limits = self.limits.get(limit_name)
            if limits and not (limits.minimum <= low < high <= limits.maximum):
                errors.append(f"{reset_name} must stay inside {limit_name} limits")

        defaults = (
            ("default_current_a", "armature_current_a"),
            ("default_speed_rad_s", "angular_speed_rad_s"),
            ("default_temperature_c", "winding_temperature_c"),
        )
        for reset_name, limit_name in defaults:
            value = float(self.reset[reset_name])
            limits = self.limits.get(limit_name)
            if limits and not limits.minimum <= value <= limits.maximum:
                errors.append(f"{reset_name} must stay inside {limit_name} limits")
        return errors

    def _validate_horizons(self) -> list[str]:
        errors: list[str] = []
        control_period = float(self.numerics.get("control_period_s", 0.0))
        burn_in = float(self.time_horizons.get("pilot_burn_in_s", 0.0))
        episode = float(self.time_horizons.get("core_episode_s", 0.0))
        horizons = tuple(
            float(value)
            for value in self.time_horizons.get("prediction_horizons_s", ())
        )
        if burn_in <= 0.0 or episode <= 0.0:
            errors.append("burn-in and episode durations must be positive")
        if not horizons or tuple(sorted(horizons)) != horizons:
            errors.append("prediction horizons must be non-empty and increasing")
        if any(value <= 0.0 or value > episode for value in horizons):
            errors.append("prediction horizons must be positive and no longer than an episode")
        if control_period > 0.0 and any(
            not math.isclose(
                value / control_period,
                round(value / control_period),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for value in horizons
        ):
            errors.append("prediction horizons must contain whole control periods")

        if set(self.parameters) == {"electrical", "mechanical", "thermal"}:
            try:
                constants = self.time_constants_s()
                if burn_in < 5.0 * constants["mechanical"]:
                    errors.append("pilot burn-in must cover at least five mechanical constants")
                if episode < 4.0 * constants["thermal"]:
                    errors.append("core episode must cover at least four thermal constants")
            except (KeyError, ValueError):
                pass
        return errors

    def _validate_validation_plan(self) -> list[str]:
        errors: list[str] = []
        required = {
            "drive_voltage_v",
            "validation_current_a",
            "hot_temperature_c",
            "zero_input_duration_s",
            "cooling_duration_s",
            "hot_cold_duration_s",
            "convergence_duration_s",
            "convergence_scaled_tolerance",
            "zero_input_current_tolerance_a",
            "zero_input_speed_tolerance_rad_s",
            "alpha_zero_output_tolerance",
            "plot_response_duration_s",
        }
        if set(self.validation) != required:
            errors.append("Phase-2 validation parameter catalog changed")
            return errors
        for name, value in self.validation.items():
            if not isinstance(value, int | float) or not math.isfinite(float(value)):
                errors.append(f"validation.{name} must be finite")
        positive = required - {"hot_temperature_c"}
        for name in positive:
            if float(self.validation[name]) <= 0.0:
                errors.append(f"validation.{name} must be positive")

        voltage = self.limits.get("armature_voltage_v")
        if voltage and not voltage.minimum <= float(
            self.validation["drive_voltage_v"]
        ) <= voltage.maximum:
            errors.append("validation drive voltage lies outside actuator limits")
        current = self.limits.get("armature_current_a")
        if current and not current.minimum < float(
            self.validation["validation_current_a"]
        ) < current.maximum:
            errors.append("validation current lies outside plant limits")
        temperature = self.limits.get("winding_temperature_c")
        if temperature and not temperature.minimum < float(
            self.validation["hot_temperature_c"]
        ) < temperature.maximum:
            errors.append("hot validation temperature lies outside plant limits")
        return errors


def _pair(value: Any) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError("expected a two-element range")
    return float(value[0]), float(value[1])


def load_phase2_spec(
    path: Path | str | None = None,
    phase0: ExperimentSpec | None = None,
) -> Phase2Spec:
    """Load and validate the canonical Phase-2 TOML file."""

    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "phase2.toml"
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    return Phase2Spec.from_dict(raw, phase0=phase0)
