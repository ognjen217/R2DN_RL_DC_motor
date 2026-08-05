"""Typed Phase-6D contract for accuracy-improvement ablations."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from r2dn_dc_motor.phase6_spec import (
    CurriculumStage,
    Phase6Spec,
    load_phase6_spec,
)
from r2dn_dc_motor.spec import SpecValidationError

VARIANT_NAMES = (
    "A_control",
    "B_latent8",
    "C_burnin1000",
    "D_rollout5000",
)
REQUIRED_PHASE6D_CHECKS = {
    "all_declared_variants_and_seeds_are_evaluated",
    "all_runs_are_finite",
    "all_contractivity_margins_are_positive",
    "selection_uses_only_fixed_validation_windows",
    "selected_checkpoint_matches_locked_median_rule",
}


@dataclass(frozen=True)
class AccuracyVariant:
    """One controlled Phase-6D model/training intervention."""

    name: str
    latent_size: int
    burn_in_steps: int
    curriculum: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AccuracyVariant:
        return cls(
            name=str(raw["name"]),
            latent_size=int(raw["latent_size"]),
            burn_in_steps=int(raw["burn_in_steps"]),
            curriculum=str(raw["curriculum"]),
        )


@dataclass(frozen=True)
class Phase6DProfile:
    """One executable accuracy-ablation budget."""

    name: str
    base_phase6_profile: str
    seeds: tuple[int, ...]
    selection_horizon_steps: int
    validation_windows: int
    validation_window_seed: int
    tie_relative_tolerance: float
    target_combined_nrmse: float
    variants: tuple[AccuracyVariant, ...]

    @classmethod
    def from_dict(cls, name: str, raw: dict[str, Any]) -> Phase6DProfile:
        return cls(
            name=name,
            base_phase6_profile=str(raw["base_phase6_profile"]),
            seeds=tuple(int(value) for value in raw["seeds"]),
            selection_horizon_steps=int(raw["selection_horizon_steps"]),
            validation_windows=int(raw["validation_windows"]),
            validation_window_seed=int(raw["validation_window_seed"]),
            tie_relative_tolerance=float(raw["tie_relative_tolerance"]),
            target_combined_nrmse=float(raw["target_combined_nrmse"]),
            variants=tuple(AccuracyVariant.from_dict(value) for value in raw["variants"]),
        )


@dataclass(frozen=True)
class Phase6DSpec:
    """Executable source of truth for Phase 6D."""

    schema_version: int
    phase: dict[str, Any]
    selection: dict[str, Any]
    screen: dict[str, Any]
    profiles: dict[str, Phase6DProfile]
    curricula: dict[str, tuple[CurriculumStage, ...]]
    validation: dict[str, Any]

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
        *,
        phase6: Phase6Spec | None = None,
    ) -> Phase6DSpec:
        spec = cls(
            schema_version=int(raw["schema_version"]),
            phase=dict(raw["phase"]),
            selection=dict(raw["selection"]),
            screen=dict(raw["screen"]),
            profiles={
                name: Phase6DProfile.from_dict(name, values)
                for name, values in raw["profiles"].items()
            },
            curricula={
                name: tuple(CurriculumStage.from_dict(stage) for stage in values)
                for name, values in raw["curricula"].items()
            },
            validation=dict(raw["validation"]),
        )
        spec.validate(phase6=phase6 or load_phase6_spec())
        return spec

    def profile(self, name: str) -> Phase6DProfile:
        try:
            return self.profiles[name]
        except KeyError as error:
            choices = ", ".join(sorted(self.profiles))
            raise SpecValidationError(
                f"unknown Phase-6D profile {name!r}; choose one of: {choices}"
            ) from error

    def stages(self, variant: AccuracyVariant) -> tuple[CurriculumStage, ...]:
        return self.curricula[variant.curriculum]

    def validate(self, *, phase6: Phase6Spec) -> None:
        errors: list[str] = []
        if self.schema_version != 1:
            errors.append("schema_version must be 1")
        if self.phase.get("number") != "6D" or self.phase.get("status") != "implemented":
            errors.append("Phase 6D must be named 6D and marked implemented")
        locked_selection = {
            "fit_split": "train",
            "selection_split": "validation",
            "id_test_used_for_selection": False,
            "ood_test_used_for_selection": False,
            "stress_used_for_selection": False,
            "metric": "validation_free_rollout_nrmse",
            "aggregation": "median_across_seeds",
            "mode": "min",
            "near_tie_policy": "first_simpler_variant_within_relative_tolerance",
        }
        for key, expected in locked_selection.items():
            if self.selection.get(key) != expected:
                errors.append(f"Phase-6D selection rule changed: {key}")
        if set(self.profiles) != {"ci", "final"}:
            errors.append("Phase 6D must define exactly ci and final profiles")
        for profile in self.profiles.values():
            errors.extend(self._validate_profile(profile, phase6))
        if set(self.curricula) != {
            "ci_baseline",
            "ci_extended",
            "final_baseline",
            "final_extended",
        }:
            errors.append("Phase-6D curriculum catalog changed")
        for name, stages in self.curricula.items():
            if not stages:
                errors.append(f"{name}: curriculum must not be empty")
            for stage in stages:
                errors.extend(stage.validate())
        final_base = phase6.profile("final").final_stages
        if self.curricula.get("final_baseline") != final_base:
            errors.append("final control curriculum drifted from Phase 6")
        extended = self.curricula.get("final_extended", ())
        if extended[: len(final_base)] != final_base:
            errors.append("extended curriculum must preserve the complete Phase-6 prefix")
        if not extended or extended[-1].rollout_steps != 5000:
            errors.append("extended curriculum must end with a 5000-step rollout")
        burn_ins = tuple(int(value) for value in self.screen.get("burn_in_steps", ()))
        if burn_ins != (250, 500, 1000):
            errors.append("screen burn-in catalog must remain [250, 500, 1000]")
        if int(self.screen.get("validation_horizon_steps", 0)) != 10_000:
            errors.append("screen validation horizon must remain 10 seconds")
        if set(self.validation.get("required_checks", ())) != REQUIRED_PHASE6D_CHECKS:
            errors.append("Phase-6D validation check catalog changed")
        if errors:
            raise SpecValidationError("\n".join(f"- {error}" for error in errors))

    def _validate_profile(
        self,
        profile: Phase6DProfile,
        phase6: Phase6Spec,
    ) -> list[str]:
        errors: list[str] = []
        phase6.profile(profile.base_phase6_profile)
        if not profile.seeds or len(set(profile.seeds)) != len(profile.seeds):
            errors.append(f"{profile.name}: seeds must be non-empty and unique")
        if min(profile.seeds, default=-1) < 0:
            errors.append(f"{profile.name}: seeds must be non-negative")
        if profile.name == "final" and profile.seeds != (17, 29, 43):
            errors.append("final Phase-6D seeds must remain [17, 29, 43]")
        if profile.selection_horizon_steps < 1 or profile.validation_windows < 1:
            errors.append(f"{profile.name}: validation budget must be positive")
        if profile.validation_window_seed < 0:
            errors.append(f"{profile.name}: validation seed must be non-negative")
        if not 0.0 <= profile.tie_relative_tolerance < 1.0:
            errors.append(f"{profile.name}: tie tolerance must lie in [0, 1)")
        if not math.isfinite(profile.target_combined_nrmse) or (
            profile.target_combined_nrmse <= 0.0
        ):
            errors.append(f"{profile.name}: target NRMSE must be finite and positive")
        if tuple(value.name for value in profile.variants) != VARIANT_NAMES:
            errors.append(f"{profile.name}: variant order or catalog changed")
        if len({value.name for value in profile.variants}) != len(profile.variants):
            errors.append(f"{profile.name}: variants must be unique")
        for variant in profile.variants:
            if variant.latent_size <= 2 or variant.burn_in_steps < 1:
                errors.append(f"{profile.name}/{variant.name}: invalid latent or burn-in")
            if variant.curriculum not in self.curricula:
                errors.append(f"{profile.name}/{variant.name}: unknown curriculum")
        if profile.name == "final":
            expected = (
                ("A_control", 4, 250, "final_baseline"),
                ("B_latent8", 8, 250, "final_baseline"),
                ("C_burnin1000", 8, 1000, "final_baseline"),
                ("D_rollout5000", 8, 1000, "final_extended"),
            )
            actual = tuple(
                (value.name, value.latent_size, value.burn_in_steps, value.curriculum)
                for value in profile.variants
            )
            if actual != expected:
                errors.append("final Phase-6D controlled-ablation variants changed")
            if profile.selection_horizon_steps != 10_000:
                errors.append("final Phase-6D selection horizon must remain 10 seconds")
        return errors

    def summary(self, profile_name: str = "final") -> str:
        profile = self.profile(profile_name)
        return "\n".join(
            (
                "PHASE 6D SPEC: PASS",
                "goal: improve long-rollout accuracy without relaxing contraction",
                f"profile: {profile.name}",
                f"variants: {', '.join(value.name for value in profile.variants)}",
                f"seeds: {list(profile.seeds)}",
                (
                    "selection: median validation NRMSE over "
                    f"{profile.selection_horizon_steps} steps"
                ),
                "ID/OOD/stress remain post-selection evidence",
            )
        )


def load_phase6d_spec(
    path: Path | str | None = None,
    *,
    phase6: Phase6Spec | None = None,
) -> Phase6DSpec:
    """Load and validate the versioned Phase-6D experiment contract."""

    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "phase6d.toml"
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    return Phase6DSpec.from_dict(raw, phase6=phase6)
