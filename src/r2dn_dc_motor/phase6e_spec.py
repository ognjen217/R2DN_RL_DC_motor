"""Typed Phase-6E contract for full-curriculum larger-latent experiments."""

from __future__ import annotations

import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from r2dn_dc_motor.phase2_spec import Phase2Spec, load_phase2_spec
from r2dn_dc_motor.phase6_spec import Phase6Spec, load_phase6_spec
from r2dn_dc_motor.phase6b_spec import Phase6BSpec, load_phase6b_spec
from r2dn_dc_motor.spec import SpecValidationError

REQUIRED_PHASE6E_CHECKS = {
    "all_declared_latents_and_seeds_are_evaluated",
    "all_training_and_selection_rollouts_are_finite",
    "all_contractivity_margins_are_positive",
    "selection_uses_only_locked_multisine_scenarios",
    "canonical_phase6c_scenario_is_held_out",
    "selected_checkpoint_matches_locked_median_rule",
}


@dataclass(frozen=True)
class MultisineScenario:
    """One fixed synthetic validation excitation."""

    name: str
    kind: str
    amplitudes_v: tuple[float, ...]
    frequencies_hz: tuple[float, ...]
    phases_rad: tuple[float, ...]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MultisineScenario:
        return cls(
            name=str(raw["name"]),
            kind=str(raw["kind"]),
            amplitudes_v=tuple(float(value) for value in raw["amplitudes_v"]),
            frequencies_hz=tuple(float(value) for value in raw["frequencies_hz"]),
            phases_rad=tuple(float(value) for value in raw["phases_rad"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase6EProfile:
    """One executable larger-latent search budget."""

    name: str
    base_phase6_profile: str
    latent_sizes: tuple[int, ...]
    seeds: tuple[int, ...]
    burn_in_steps: int
    selection_duration_s: float
    selection_chunk_steps: int
    selection_split: str
    selection_anchor_indices: tuple[int, ...]
    tie_relative_tolerance: float
    target_combined_nrmse: float
    multisine_scenarios: tuple[MultisineScenario, ...]

    @classmethod
    def from_dict(cls, name: str, raw: dict[str, Any]) -> Phase6EProfile:
        return cls(
            name=name,
            base_phase6_profile=str(raw["base_phase6_profile"]),
            latent_sizes=tuple(int(value) for value in raw["latent_sizes"]),
            seeds=tuple(int(value) for value in raw["seeds"]),
            burn_in_steps=int(raw["burn_in_steps"]),
            selection_duration_s=float(raw["selection_duration_s"]),
            selection_chunk_steps=int(raw["selection_chunk_steps"]),
            selection_split=str(raw["selection_split"]),
            selection_anchor_indices=tuple(
                int(value) for value in raw["selection_anchor_indices"]
            ),
            tie_relative_tolerance=float(raw["tie_relative_tolerance"]),
            target_combined_nrmse=float(raw["target_combined_nrmse"]),
            multisine_scenarios=tuple(
                MultisineScenario.from_dict(value)
                for value in raw["multisine_scenarios"]
            ),
        )


@dataclass(frozen=True)
class Phase6ESpec:
    """Executable source of truth for Phase 6E."""

    schema_version: int
    phase: dict[str, Any]
    selection: dict[str, Any]
    profiles: dict[str, Phase6EProfile]
    validation: dict[str, Any]

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
        *,
        phase2: Phase2Spec | None = None,
        phase6: Phase6Spec | None = None,
        phase6b: Phase6BSpec | None = None,
    ) -> Phase6ESpec:
        phase6 = phase6 or load_phase6_spec()
        spec = cls(
            schema_version=int(raw["schema_version"]),
            phase=dict(raw["phase"]),
            selection=dict(raw["selection"]),
            profiles={
                name: Phase6EProfile.from_dict(name, values)
                for name, values in raw["profiles"].items()
            },
            validation=dict(raw["validation"]),
        )
        spec.validate(
            phase2=phase2 or load_phase2_spec(),
            phase6=phase6,
            phase6b=phase6b or load_phase6b_spec(phase6=phase6),
        )
        return spec

    def profile(self, name: str) -> Phase6EProfile:
        try:
            return self.profiles[name]
        except KeyError as error:
            choices = ", ".join(sorted(self.profiles))
            raise SpecValidationError(
                f"unknown Phase-6E profile {name!r}; choose one of: {choices}"
            ) from error

    def validate(
        self,
        *,
        phase2: Phase2Spec,
        phase6: Phase6Spec,
        phase6b: Phase6BSpec,
    ) -> None:
        errors: list[str] = []
        if self.schema_version != 1:
            errors.append("schema_version must be 1")
        if self.phase.get("number") != "6E" or self.phase.get("status") != "implemented":
            errors.append("Phase 6E must be named 6E and marked implemented")
        locked_selection = {
            "fit_split": "train",
            "selection_split": "synthetic_validation_multisine",
            "selection_reference": "full_rk4",
            "id_test_used_for_selection": False,
            "ood_test_used_for_selection": False,
            "stress_used_for_selection": False,
            "canonical_phase6c_scenario_used_for_selection": False,
            "run_metric": "median_multisine_combined_nrmse",
            "latent_aggregation": "median_across_seeds",
            "mode": "min",
            "near_tie_policy": "smallest_latent_within_relative_tolerance",
        }
        for key, expected in locked_selection.items():
            if self.selection.get(key) != expected:
                errors.append(f"Phase-6E selection rule changed: {key}")
        if set(self.profiles) != {"ci", "final"}:
            errors.append("Phase 6E must define exactly ci and final profiles")
        for profile in self.profiles.values():
            errors.extend(self._validate_profile(profile, phase2=phase2, phase6=phase6))
        final = self.profiles.get("final")
        if final is not None:
            if final.latent_sizes != (8, 12, 16):
                errors.append("final Phase-6E latent catalog must be [8, 12, 16]")
            if final.seeds != (17, 29, 43):
                errors.append("final Phase-6E seeds must remain [17, 29, 43]")
            if final.burn_in_steps != 250:
                errors.append("final Phase-6E burn-in must remain 250 steps")
            if not math.isclose(final.selection_duration_s, 100.0):
                errors.append("final Phase-6E selection horizon must remain 100 seconds")
            if len(final.multisine_scenarios) != 3:
                errors.append("final Phase-6E must use exactly three selection scenarios")
            canonical = next(
                scenario for scenario in phase6b.scenarios if scenario["name"] == "multisine"
            )
            canonical_signature = _scenario_signature(canonical)
            if any(
                _scenario_signature(value.to_dict()) == canonical_signature
                for value in final.multisine_scenarios
            ):
                errors.append("canonical Phase-6C multisine leaked into Phase-6E selection")
        if set(self.validation.get("required_checks", ())) != REQUIRED_PHASE6E_CHECKS:
            errors.append("Phase-6E validation check catalog changed")
        safe_voltage = float(self.validation.get("maximum_safe_voltage_v", math.nan))
        if not math.isfinite(safe_voltage) or safe_voltage <= 0.0:
            errors.append("Phase-6E safe voltage must be finite and positive")
        if errors:
            raise SpecValidationError("\n".join(f"- {error}" for error in errors))

    def _validate_profile(
        self,
        profile: Phase6EProfile,
        *,
        phase2: Phase2Spec,
        phase6: Phase6Spec,
    ) -> list[str]:
        errors: list[str] = []
        base = phase6.profile(profile.base_phase6_profile)
        if not profile.latent_sizes or tuple(sorted(set(profile.latent_sizes))) != (
            profile.latent_sizes
        ):
            errors.append(f"{profile.name}: latent sizes must be unique and increasing")
        if min(profile.latent_sizes, default=0) <= 2:
            errors.append(f"{profile.name}: latent sizes must exceed the two outputs")
        if not profile.seeds or len(set(profile.seeds)) != len(profile.seeds):
            errors.append(f"{profile.name}: seeds must be non-empty and unique")
        if min(profile.seeds, default=-1) < 0:
            errors.append(f"{profile.name}: seeds must be non-negative")
        if profile.burn_in_steps != base.burn_in_steps:
            errors.append(f"{profile.name}: burn-in drifted from its Phase-6 profile")
        if profile.selection_split != "validation":
            errors.append(f"{profile.name}: selection anchors must come from validation")
        if not profile.multisine_scenarios:
            errors.append(f"{profile.name}: at least one multisine scenario is required")
        if len(profile.selection_anchor_indices) != len(profile.multisine_scenarios):
            errors.append(f"{profile.name}: every scenario needs one validation anchor")
        if min(profile.selection_anchor_indices, default=-1) < 0:
            errors.append(f"{profile.name}: anchor indices must be non-negative")
        if len(set(profile.selection_anchor_indices)) != len(
            profile.selection_anchor_indices
        ):
            errors.append(f"{profile.name}: selection anchors must be distinct")
        dt = phase2.integration_settings.control_period_s
        steps_float = profile.selection_duration_s / dt
        steps = int(round(steps_float))
        if (
            profile.selection_duration_s <= 0.0
            or not math.isclose(steps_float, steps, rel_tol=0.0, abs_tol=1e-9)
        ):
            errors.append(f"{profile.name}: duration must contain an integer number of steps")
        if profile.selection_chunk_steps < 1 or steps % profile.selection_chunk_steps:
            errors.append(f"{profile.name}: chunk size must divide the selection horizon")
        if not 0.0 <= profile.tie_relative_tolerance < 1.0:
            errors.append(f"{profile.name}: tie tolerance must lie in [0, 1)")
        if not math.isfinite(profile.target_combined_nrmse) or (
            profile.target_combined_nrmse <= 0.0
        ):
            errors.append(f"{profile.name}: target NRMSE must be finite and positive")
        safe_voltage = float(self.validation["maximum_safe_voltage_v"])
        names = tuple(value.name for value in profile.multisine_scenarios)
        if len(set(names)) != len(names):
            errors.append(f"{profile.name}: multisine scenario names must be unique")
        for scenario in profile.multisine_scenarios:
            sizes = {
                len(scenario.amplitudes_v),
                len(scenario.frequencies_hz),
                len(scenario.phases_rad),
            }
            if scenario.kind != "multisine" or sizes != {len(scenario.amplitudes_v)}:
                errors.append(f"{profile.name}/{scenario.name}: malformed multisine vectors")
            if not scenario.amplitudes_v or min(scenario.frequencies_hz, default=0.0) <= 0:
                errors.append(f"{profile.name}/{scenario.name}: empty or nonpositive frequency")
            if sum(abs(value) for value in scenario.amplitudes_v) > safe_voltage:
                errors.append(f"{profile.name}/{scenario.name}: voltage exceeds safe limit")
        return errors

    def summary(self, profile_name: str = "final") -> str:
        profile = self.profile(profile_name)
        return "\n".join(
            (
                "PHASE 6E SPEC: PASS",
                "goal: test larger latent dimensions under the full Phase-6 curriculum",
                f"profile: {profile.name}",
                f"latents: {list(profile.latent_sizes)}",
                f"seeds: {list(profile.seeds)}",
                f"burn-in: {profile.burn_in_steps} steps",
                (
                    "selection: median across seeds of median combined NRMSE on "
                    f"{len(profile.multisine_scenarios)} x "
                    f"{profile.selection_duration_s:g} s held-out multisine rollouts"
                ),
                (
                    f"tie rule: prefer smaller latent within "
                    f"{100 * profile.tie_relative_tolerance:g}%"
                ),
                "canonical Phase-6C 1000 s multisine remains held out",
            )
        )


def _scenario_signature(raw: dict[str, Any]) -> tuple[Any, ...]:
    return (
        tuple(float(value) for value in raw["amplitudes_v"]),
        tuple(float(value) for value in raw["frequencies_hz"]),
        tuple(float(value) for value in raw["phases_rad"]),
    )


def phase6e_protocol_payload(spec: Phase6ESpec, phase6: Phase6Spec) -> dict[str, Any]:
    """Return every setting that can change training or selection."""

    return {
        "phase6": {
            "interface": phase6.interface,
            "architecture": phase6.architecture,
            "loss": phase6.loss,
            "optimizer": phase6.optimizer,
            "profiles": {
                name: asdict(profile) for name, profile in phase6.profiles.items()
            },
        },
        "phase6e": {
            "phase": spec.phase,
            "selection": spec.selection,
            "profiles": {
                name: asdict(profile) for name, profile in spec.profiles.items()
            },
            "validation": spec.validation,
        },
    }


def load_phase6e_spec(
    path: Path | str | None = None,
    *,
    phase2: Phase2Spec | None = None,
    phase6: Phase6Spec | None = None,
    phase6b: Phase6BSpec | None = None,
) -> Phase6ESpec:
    """Load and validate the versioned Phase-6E experiment contract."""

    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "phase6e.toml"
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    return Phase6ESpec.from_dict(
        raw,
        phase2=phase2,
        phase6=phase6,
        phase6b=phase6b,
    )
