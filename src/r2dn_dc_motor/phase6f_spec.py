"""Typed Phase-6F contract for the latent-16 optimizer-floor ablation."""

from __future__ import annotations

import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from r2dn_dc_motor.phase2_spec import Phase2Spec, load_phase2_spec
from r2dn_dc_motor.phase6_spec import Phase6Spec, load_phase6_spec
from r2dn_dc_motor.phase6b_spec import Phase6BSpec, load_phase6b_spec
from r2dn_dc_motor.phase6e_spec import (
    MultisineScenario,
    Phase6ESpec,
    load_phase6e_spec,
)
from r2dn_dc_motor.spec import SpecValidationError

REQUIRED_PHASE6F_CHECKS = {
    "baseline_checkpoint_is_phase6e_winner",
    "all_declared_optimizer_variants_are_evaluated",
    "all_training_and_selection_rollouts_are_finite",
    "all_contractivity_margins_are_positive",
    "selection_scenarios_are_new_and_canonical_is_held_out",
    "selected_checkpoint_matches_locked_tie_rule",
}


@dataclass(frozen=True)
class OptimizerVariant:
    """One controlled optimizer/budget candidate."""

    name: str
    source: str
    schedule: str
    initial_learning_rate: float
    final_learning_rate: float
    stage_update_multiplier: int

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> OptimizerVariant:
        return cls(
            name=str(raw["name"]),
            source=str(raw["source"]),
            schedule=str(raw["schedule"]),
            initial_learning_rate=float(raw["initial_learning_rate"]),
            final_learning_rate=float(raw["final_learning_rate"]),
            stage_update_multiplier=int(raw["stage_update_multiplier"]),
        )

    def optimizer_payload(self, phase6: Phase6Spec) -> dict[str, Any]:
        """Return the exact optimizer settings consumed by the trainer."""

        return {
            "schedule": self.schedule,
            "initial_learning_rate": self.initial_learning_rate,
            "final_learning_rate": self.final_learning_rate,
            "weight_decay": float(phase6.optimizer["weight_decay"]),
            "gradient_clip_norm": float(phase6.optimizer["gradient_clip_norm"]),
        }


@dataclass(frozen=True)
class Phase6FProfile:
    """One executable optimizer-ablation budget."""

    name: str
    base_phase6_profile: str
    latent_size: int
    seed: int
    burn_in_steps: int
    selection_duration_s: float
    selection_chunk_steps: int
    selection_split: str
    selection_anchor_indices: tuple[int, ...]
    tie_relative_tolerance: float
    target_combined_nrmse: float
    variants: tuple[OptimizerVariant, ...]
    multisine_scenarios: tuple[MultisineScenario, ...]

    @classmethod
    def from_dict(cls, name: str, raw: dict[str, Any]) -> Phase6FProfile:
        return cls(
            name=name,
            base_phase6_profile=str(raw["base_phase6_profile"]),
            latent_size=int(raw["latent_size"]),
            seed=int(raw["seed"]),
            burn_in_steps=int(raw["burn_in_steps"]),
            selection_duration_s=float(raw["selection_duration_s"]),
            selection_chunk_steps=int(raw["selection_chunk_steps"]),
            selection_split=str(raw["selection_split"]),
            selection_anchor_indices=tuple(
                int(value) for value in raw["selection_anchor_indices"]
            ),
            tie_relative_tolerance=float(raw["tie_relative_tolerance"]),
            target_combined_nrmse=float(raw["target_combined_nrmse"]),
            variants=tuple(OptimizerVariant.from_dict(value) for value in raw["variants"]),
            multisine_scenarios=tuple(
                MultisineScenario.from_dict(value)
                for value in raw["multisine_scenarios"]
            ),
        )


@dataclass(frozen=True)
class Phase6FSpec:
    """Executable source of truth for Phase 6F."""

    schema_version: int
    phase: dict[str, Any]
    selection: dict[str, Any]
    profiles: dict[str, Phase6FProfile]
    validation: dict[str, Any]

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
        *,
        phase2: Phase2Spec | None = None,
        phase6: Phase6Spec | None = None,
        phase6b: Phase6BSpec | None = None,
        phase6e: Phase6ESpec | None = None,
    ) -> Phase6FSpec:
        phase2 = phase2 or load_phase2_spec()
        phase6 = phase6 or load_phase6_spec()
        phase6b = phase6b or load_phase6b_spec(phase2=phase2, phase6=phase6)
        phase6e = phase6e or load_phase6e_spec(
            phase2=phase2,
            phase6=phase6,
            phase6b=phase6b,
        )
        spec = cls(
            schema_version=int(raw["schema_version"]),
            phase=dict(raw["phase"]),
            selection=dict(raw["selection"]),
            profiles={
                name: Phase6FProfile.from_dict(name, values)
                for name, values in raw["profiles"].items()
            },
            validation=dict(raw["validation"]),
        )
        spec.validate(phase2=phase2, phase6=phase6, phase6b=phase6b, phase6e=phase6e)
        return spec

    def profile(self, name: str) -> Phase6FProfile:
        try:
            return self.profiles[name]
        except KeyError as error:
            choices = ", ".join(sorted(self.profiles))
            raise SpecValidationError(
                f"unknown Phase-6F profile {name!r}; choose one of: {choices}"
            ) from error

    def validate(
        self,
        *,
        phase2: Phase2Spec,
        phase6: Phase6Spec,
        phase6b: Phase6BSpec,
        phase6e: Phase6ESpec,
    ) -> None:
        errors: list[str] = []
        if self.schema_version != 1:
            errors.append("schema_version must be 1")
        if self.phase.get("number") != "6F" or self.phase.get("status") != "implemented":
            errors.append("Phase 6F must be named 6F and marked implemented")
        locked_selection = {
            "fit_split": "train",
            "selection_split": "synthetic_validation_multisine",
            "selection_reference": "full_rk4",
            "phase6e_checkpoint_is_baseline": True,
            "canonical_phase6c_scenario_used_for_selection": False,
            "phase6e_selection_scenarios_reused": False,
            "run_metric": "median_multisine_combined_nrmse",
            "mode": "min",
            "near_tie_policy": "lowest_training_budget_within_relative_tolerance",
        }
        for key, expected in locked_selection.items():
            if self.selection.get(key) != expected:
                errors.append(f"Phase-6F selection rule changed: {key}")
        if set(self.profiles) != {"ci", "final"}:
            errors.append("Phase 6F must define exactly ci and final profiles")
        for profile in self.profiles.values():
            errors.extend(
                self._validate_profile(
                    profile,
                    phase2=phase2,
                    phase6=phase6,
                    phase6b=phase6b,
                    phase6e=phase6e,
                )
            )
        final = self.profiles.get("final")
        if final is not None:
            if final.latent_size != 16 or final.seed != 43:
                errors.append("final Phase-6F model must remain latent-16/seed-43")
            if final.burn_in_steps != 250:
                errors.append("final Phase-6F burn-in must remain 250 steps")
            if not math.isclose(final.selection_duration_s, 100.0):
                errors.append("final Phase-6F selection horizon must remain 100 seconds")
            if final.selection_anchor_indices != (3, 0, 2):
                errors.append("final Phase-6F anchors must remain [3, 0, 2]")
            if int(phase6.architecture["feature_size"]) != 32 or tuple(
                phase6.architecture["hidden_sizes"]
            ) != (32, 32):
                errors.append("Phase-6F network must remain 32/[32,32]")
        if set(self.validation.get("required_checks", ())) != REQUIRED_PHASE6F_CHECKS:
            errors.append("Phase-6F validation check catalog changed")
        if errors:
            raise SpecValidationError("\n".join(f"- {error}" for error in errors))

    def _validate_profile(
        self,
        profile: Phase6FProfile,
        *,
        phase2: Phase2Spec,
        phase6: Phase6Spec,
        phase6b: Phase6BSpec,
        phase6e: Phase6ESpec,
    ) -> list[str]:
        errors: list[str] = []
        base = phase6.profile(profile.base_phase6_profile)
        if profile.latent_size <= 2 or profile.seed < 0:
            errors.append(f"{profile.name}: latent and seed are invalid")
        if profile.burn_in_steps != base.burn_in_steps:
            errors.append(f"{profile.name}: burn-in drifted from Phase 6")
        if profile.selection_split != "validation":
            errors.append(f"{profile.name}: selection anchors must be validation data")
        if len(profile.selection_anchor_indices) != len(profile.multisine_scenarios):
            errors.append(f"{profile.name}: every scenario needs one anchor")
        maximum_anchor = phase6b.profile(profile.base_phase6_profile).stress_anchors_per_split
        if min(profile.selection_anchor_indices, default=-1) < 0 or max(
            profile.selection_anchor_indices,
            default=maximum_anchor,
        ) >= maximum_anchor:
            errors.append(f"{profile.name}: selection anchor index is unavailable")
        steps_float = profile.selection_duration_s / phase2.integration_settings.control_period_s
        steps = int(round(steps_float))
        if profile.selection_duration_s <= 0.0 or not math.isclose(
            steps_float,
            steps,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            errors.append(f"{profile.name}: duration must contain an integer step count")
        if profile.selection_chunk_steps < 1 or steps % profile.selection_chunk_steps:
            errors.append(f"{profile.name}: selection chunk must divide the horizon")
        if not 0.0 <= profile.tie_relative_tolerance < 1.0:
            errors.append(f"{profile.name}: tie tolerance must lie in [0, 1)")
        if not math.isfinite(profile.target_combined_nrmse) or (
            profile.target_combined_nrmse <= 0.0
        ):
            errors.append(f"{profile.name}: target NRMSE must be finite and positive")
        expected_variants = (
            ("baseline_phase6e", "phase6e_checkpoint", "constant", 1),
            ("cosine_1x", "train", "cosine_decay", 1),
            ("cosine_2x", "train", "cosine_decay", 2),
        )
        actual_variants = tuple(
            (value.name, value.source, value.schedule, value.stage_update_multiplier)
            for value in profile.variants
        )
        if actual_variants != expected_variants:
            errors.append(f"{profile.name}: optimizer variant catalog changed")
        for variant in profile.variants:
            rates = (variant.initial_learning_rate, variant.final_learning_rate)
            if not all(math.isfinite(value) and value > 0.0 for value in rates):
                errors.append(f"{profile.name}/{variant.name}: learning rates are invalid")
            if variant.final_learning_rate > variant.initial_learning_rate:
                errors.append(f"{profile.name}/{variant.name}: LR schedule must not increase")
            if variant.schedule == "constant" and not math.isclose(*rates):
                errors.append(f"{profile.name}/{variant.name}: constant LR endpoints differ")
            base_learning_rate = float(phase6.optimizer["learning_rate"])
            if not math.isclose(variant.initial_learning_rate, base_learning_rate):
                errors.append(f"{profile.name}/{variant.name}: initial LR drifted")
            expected_final = (
                base_learning_rate
                if variant.source == "phase6e_checkpoint"
                else 1.0e-5
            )
            if not math.isclose(variant.final_learning_rate, expected_final):
                errors.append(f"{profile.name}/{variant.name}: final LR drifted")
        safe_voltage = float(self.validation.get("maximum_safe_voltage_v", math.nan))
        phase6e_signatures = {
            _scenario_signature(value.to_dict())
            for value in phase6e.profile(profile.name).multisine_scenarios
        }
        canonical = next(value for value in phase6b.scenarios if value["name"] == "multisine")
        canonical_signature = _scenario_signature(canonical)
        names = tuple(value.name for value in profile.multisine_scenarios)
        if len(set(names)) != len(names):
            errors.append(f"{profile.name}: multisine names must be unique")
        for scenario in profile.multisine_scenarios:
            sizes = {
                len(scenario.amplitudes_v),
                len(scenario.frequencies_hz),
                len(scenario.phases_rad),
            }
            signature = _scenario_signature(scenario.to_dict())
            if scenario.kind != "multisine" or sizes != {len(scenario.amplitudes_v)}:
                errors.append(f"{profile.name}/{scenario.name}: malformed multisine")
            if not scenario.amplitudes_v or min(scenario.frequencies_hz, default=0.0) <= 0:
                errors.append(f"{profile.name}/{scenario.name}: invalid frequency vector")
            if sum(abs(value) for value in scenario.amplitudes_v) > safe_voltage:
                errors.append(f"{profile.name}/{scenario.name}: voltage exceeds safe limit")
            if signature == canonical_signature or signature in phase6e_signatures:
                errors.append(f"{profile.name}/{scenario.name}: selection scenario was reused")
        return errors

    def summary(self, profile_name: str = "final") -> str:
        profile = self.profile(profile_name)
        base_profile = load_phase6_spec().profile(profile.base_phase6_profile)
        base_updates = sum(
            stage.updates for stage in base_profile.final_stages
        )
        budgets = [base_updates * value.stage_update_multiplier for value in profile.variants]
        return "\n".join(
            (
                "PHASE 6F SPEC: PASS",
                "goal: determine whether the remaining error is an optimizer floor",
                f"profile: {profile.name}",
                f"fixed model: latent={profile.latent_size}, seed={profile.seed}, 32/[32,32]",
                f"variants: {[value.name for value in profile.variants]}",
                f"update budgets: {budgets}",
                "cosine schedule: 1e-3 -> 1e-5",
                (
                    f"selection: {len(profile.multisine_scenarios)} x "
                    f"{profile.selection_duration_s:g} s new held-out multisine rollouts"
                ),
                "canonical Phase-6C 1000 s multisine remains post-selection evidence",
            )
        )


def _scenario_signature(raw: dict[str, Any]) -> tuple[Any, ...]:
    return (
        tuple(float(value) for value in raw["amplitudes_v"]),
        tuple(float(value) for value in raw["frequencies_hz"]),
        tuple(float(value) for value in raw["phases_rad"]),
    )


def phase6f_protocol_payload(
    spec: Phase6FSpec,
    phase6: Phase6Spec,
    phase6e: Phase6ESpec,
) -> dict[str, Any]:
    """Return every setting that can change training or selection."""

    return {
        "phase6": {
            "interface": phase6.interface,
            "architecture": phase6.architecture,
            "loss": phase6.loss,
            "optimizer": phase6.optimizer,
            "profiles": {name: asdict(value) for name, value in phase6.profiles.items()},
        },
        "phase6e_baseline": {
            "phase": phase6e.phase,
            "selection": phase6e.selection,
        },
        "phase6f": {
            "phase": spec.phase,
            "selection": spec.selection,
            "profiles": {name: asdict(value) for name, value in spec.profiles.items()},
            "validation": spec.validation,
        },
    }


def load_phase6f_spec(
    path: Path | str | None = None,
    *,
    phase2: Phase2Spec | None = None,
    phase6: Phase6Spec | None = None,
    phase6b: Phase6BSpec | None = None,
    phase6e: Phase6ESpec | None = None,
) -> Phase6FSpec:
    """Load and validate the versioned Phase-6F experiment contract."""

    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "phase6f.toml"
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    return Phase6FSpec.from_dict(
        raw,
        phase2=phase2,
        phase6=phase6,
        phase6b=phase6b,
        phase6e=phase6e,
    )
