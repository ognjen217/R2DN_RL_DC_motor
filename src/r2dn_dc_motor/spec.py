"""Typed Phase-0 experiment specification and invariant checks."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SpecValidationError(ValueError):
    """Raised when the frozen experimental specification is inconsistent."""


@dataclass(frozen=True)
class Range:
    """Closed physical range for one signal."""

    minimum: float
    maximum: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Range:
        return cls(minimum=float(raw["minimum"]), maximum=float(raw["maximum"]))

    def validate(self, name: str) -> list[str]:
        errors: list[str] = []
        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            errors.append(f"{name}: bounds must be finite")
        if self.minimum >= self.maximum:
            errors.append(f"{name}: minimum must be smaller than maximum")
        return errors

    def contains(self, other: Range) -> bool:
        return self.minimum <= other.minimum and other.maximum <= self.maximum


@dataclass(frozen=True)
class SignalSpec:
    """Signals visible at the plant, world-model, and policy boundaries."""

    state: tuple[str, ...]
    plant_output: tuple[str, ...]
    hidden_state: tuple[str, ...]
    control: str
    known_disturbance: str
    reference: str
    world_model_input: tuple[str, ...]
    world_model_output: tuple[str, ...]
    policy_observation: tuple[str, ...]
    policy_action: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SignalSpec:
        return cls(
            state=tuple(raw["state"]),
            plant_output=tuple(raw["plant_output"]),
            hidden_state=tuple(raw["hidden_state"]),
            control=raw["control"],
            known_disturbance=raw["known_disturbance"],
            reference=raw["reference"],
            world_model_input=tuple(raw["world_model_input"]),
            world_model_output=tuple(raw["world_model_output"]),
            policy_observation=tuple(raw["policy_observation"]),
            policy_action=raw["policy_action"],
        )


@dataclass(frozen=True)
class ExperimentSpec:
    """Frozen, validated definition of the core experiment."""

    schema_version: int
    experiment: dict[str, Any]
    signals: SignalSpec
    assumptions: dict[str, Any]
    limits: dict[str, Range]
    models: dict[str, tuple[str, ...]]
    temporal_representation: dict[str, Any]
    metrics: dict[str, Any]
    r2dn_reference: dict[str, Any]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentSpec:
        spec = cls(
            schema_version=int(raw["schema_version"]),
            experiment=dict(raw["experiment"]),
            signals=SignalSpec.from_dict(raw["signals"]),
            assumptions=dict(raw["assumptions"]),
            limits={name: Range.from_dict(value) for name, value in raw["limits"].items()},
            models={
                "required": tuple(raw["models"]["required"]),
                "optional": tuple(raw["models"]["optional"]),
            },
            temporal_representation=dict(raw["temporal_representation"]),
            metrics={
                **raw["metrics"],
                "primary": tuple(raw["metrics"]["primary"]),
                "secondary": tuple(raw["metrics"]["secondary"]),
            },
            r2dn_reference=dict(raw["r2dn_reference"]),
        )
        spec.validate()
        return spec

    def validate(self) -> None:
        """Validate scientific and interface invariants for Phase 0."""

        errors: list[str] = []
        s = self.signals

        if self.schema_version != 1:
            errors.append("schema_version must be 1")
        if self.experiment.get("phase") != 0:
            errors.append("experiment.phase must be 0")

        errors.extend(_duplicates("signals.state", s.state))
        errors.extend(_duplicates("signals.plant_output", s.plant_output))
        errors.extend(_duplicates("signals.world_model_input", s.world_model_input))
        errors.extend(_duplicates("signals.policy_observation", s.policy_observation))

        state = set(s.state)
        plant_output = set(s.plant_output)
        hidden = set(s.hidden_state)
        world_input = set(s.world_model_input)
        world_output = set(s.world_model_output)
        policy_observation = set(s.policy_observation)

        if not plant_output <= state:
            errors.append("plant outputs must be part of the full state")
        if not hidden <= state:
            errors.append("hidden states must be part of the full state")
        if plant_output & hidden:
            errors.append("hidden states must not appear in plant outputs")
        if hidden & world_input:
            errors.append("hidden temperature leaked into world-model inputs")
        if hidden & world_output:
            errors.append("hidden temperature leaked into world-model outputs")
        if hidden & policy_observation:
            errors.append("hidden temperature leaked into policy observations")
        if s.control not in world_input:
            errors.append("world-model input must contain the control signal")
        if s.policy_action != s.control:
            errors.append("policy action must equal the plant control signal")

        expected_policy = {
            "armature_current_a",
            "angular_speed_rad_s",
            "angular_speed_reference_rad_s",
            "previous_armature_voltage_v",
        }
        if policy_observation != expected_policy:
            errors.append("policy observation changed from the frozen core interface")

        required_models = {"FULL", "ISO-NOM", "ISO-CAL", "R2DN", "PI", "PPO-FULL"}
        if set(self.models["required"]) != required_models:
            errors.append("required model catalog changed from the frozen comparison")

        expected_primary = {"long_rollout_nrmse", "full_plant_speed_iae"}
        if set(self.metrics["primary"]) != expected_primary:
            errors.append("primary metrics changed from the frozen experiment")

        expected_hypothesis = "ppo_r2dn_has_smaller_full_plant_transfer_gap_than_ppo_iso_cal"
        if self.experiment.get("hypothesis") != expected_hypothesis:
            errors.append("core hypothesis changed")

        if self.assumptions.get("only_hidden_dynamic_state") != "winding_temperature_c":
            errors.append(
                "temperature must be the only hidden dynamic state in the core experiment"
            )
        if self.assumptions.get("temperature_available_to_world_models") is not False:
            errors.append("temperature must be unavailable to all world models")
        if self.assumptions.get("temperature_available_to_policy") is not False:
            errors.append("temperature must be unavailable to the policy")

        required_limits = {
            "armature_voltage_v",
            "armature_current_a",
            "angular_speed_rad_s",
            "angular_speed_reference_rad_s",
            "winding_temperature_c",
        }
        if set(self.limits) != required_limits:
            errors.append("physical limit catalog is incomplete or contains unexpected entries")
        for name, bounds in self.limits.items():
            errors.extend(bounds.validate(name))

        voltage = self.limits.get("armature_voltage_v")
        if voltage and not math.isclose(voltage.minimum, -voltage.maximum):
            errors.append("voltage action range must be symmetric around zero")

        speed = self.limits.get("angular_speed_rad_s")
        speed_reference = self.limits.get("angular_speed_reference_rad_s")
        if speed and speed_reference and not speed.contains(speed_reference):
            errors.append("speed-reference range must stay inside the plant speed range")

        control_period = float(self.temporal_representation.get("control_period_s", 0.0))
        integrator_step = float(
            self.temporal_representation.get("plant_integrator_step_s", 0.0)
        )
        if control_period <= 0.0 or integrator_step <= 0.0:
            errors.append("time steps must be positive")
        elif control_period < integrator_step:
            errors.append("plant integrator step must not exceed the control period")
        elif not math.isclose(
            control_period / integrator_step,
            round(control_period / integrator_step),
        ):
            errors.append("control period must contain an integer number of RK4 steps")

        if self.temporal_representation.get("r2dn") != "discrete_recurrent_state_space":
            errors.append(
                "R2DN must follow the discrete recurrent formulation of the reference code"
            )
        for name in ("full_plant", "iso_nom", "iso_cal"):
            if self.temporal_representation.get(name) != "continuous_ode_rk4":
                errors.append(f"{name} must use the shared continuous ODE/RK4 representation")

        reference = self.r2dn_reference
        if reference.get("repository") != "https://github.com/nic-barbara/R2DN":
            errors.append("R2DN repository must point to the authors' official implementation")
        commit = str(reference.get("commit", ""))
        if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
            errors.append("R2DN reference must be pinned to a full lowercase commit SHA")
        if reference.get("license") != "MIT":
            errors.append("unexpected upstream R2DN license")
        if reference.get("backend") != "jax_flax":
            errors.append("upstream R2DN backend must be recorded as JAX/Flax")

        if errors:
            raise SpecValidationError("\n".join(f"- {error}" for error in errors))

    def summary(self) -> str:
        """Return a compact human-readable summary after validation."""

        return "\n".join(
            [
                "PHASE 0: PASS",
                f"experiment: {self.experiment['id']}",
                f"state: {', '.join(self.signals.state)}",
                f"observations: {', '.join(self.signals.plant_output)}",
                f"hidden: {', '.join(self.signals.hidden_state)}",
                f"control: {self.signals.control}",
                f"required models: {', '.join(self.models['required'])}",
                f"primary metrics: {', '.join(self.metrics['primary'])}",
                f"R2DN commit: {self.r2dn_reference['commit']}",
            ]
        )


def _duplicates(name: str, values: tuple[str, ...]) -> list[str]:
    duplicates = sorted({value for value in values if values.count(value) > 1})
    return [f"{name} contains duplicates: {', '.join(duplicates)}"] if duplicates else []


def load_phase0_spec(path: Path | str | None = None) -> ExperimentSpec:
    """Load and validate the canonical Phase-0 TOML file."""

    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "phase0.toml"
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    return ExperimentSpec.from_dict(raw)
