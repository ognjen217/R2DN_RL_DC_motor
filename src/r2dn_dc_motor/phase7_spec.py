"""Typed Phase-7 contract for improved pure-R2DN identification and evaluation."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from r2dn_dc_motor.phase6_spec import (
    FORBIDDEN_TRAINING_FEATURES,
    CurriculumStage,
)
from r2dn_dc_motor.spec import SpecValidationError

REQUIRED_PHASE7_CHECKS = {
    "broadband_dataset_is_full_only_and_temperature_hidden",
    "increment_statistics_use_train_only_model_view",
    "all_variants_and_seeds_are_evaluated",
    "absolute_state_output_and_contracting_parameterization_are_retained",
    "all_contractivity_margins_are_positive",
    "selection_uses_validation_only",
    "test_bank_is_excluded_from_selection",
    "final_report_includes_mean_median_worst_and_divergence",
}
TEST_BANK_CATEGORIES = ("id", "excitation_ood", "thermal_ood")
TEST_BANK_SCENARIO_NAMES = ("prbs", "sine", "multisine")
TEST_BANK_MAXIMUM_TEMPERATURE_C = 110.0
TEST_BANK_MINIMUM_ANCHOR_BURN_IN_STEPS = 250
LOCKED_TEST_BANK_SCENARIOS = (
    {
        "name": "prbs",
        "kind": "prbs",
        "amplitude_v": 5.0,
        "hold_steps": 250,
        "seed": 78011,
    },
    {
        "name": "sine",
        "kind": "sine",
        "amplitude_v": 7.0,
        "frequency_hz": 0.25,
        "phase_rad": 0.35,
    },
    {
        "name": "multisine",
        "kind": "multisine",
        "amplitudes_v": [3.0, 2.5, 2.0],
        "frequencies_hz": [0.07, 0.37, 1.43],
        "phases_rad": [0.2, 1.1, 2.0],
    },
)


@dataclass(frozen=True)
class Phase7Variant:
    name: str
    curriculum: str
    feature_size: int
    hidden_sizes: tuple[int, ...]
    delta_weight: float
    use_multihorizon_loss: bool

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Phase7Variant:
        return cls(
            name=str(raw["name"]),
            curriculum=str(raw["curriculum"]),
            feature_size=int(raw["feature_size"]),
            hidden_sizes=tuple(int(value) for value in raw["hidden_sizes"]),
            delta_weight=float(raw["delta_weight"]),
            use_multihorizon_loss=bool(raw["use_multihorizon_loss"]),
        )


@dataclass(frozen=True)
class Phase7Profile:
    name: str
    seeds: tuple[int, ...]
    burn_in_steps: int
    validation_horizon_steps: int
    validation_window_seed: int
    validation_windows: int
    history_log_interval: int
    base_stages: tuple[CurriculumStage, ...]
    multiscale_stages: tuple[CurriculumStage, ...]

    @classmethod
    def from_dict(cls, name: str, raw: dict[str, Any]) -> Phase7Profile:
        return cls(
            name=name,
            seeds=tuple(int(value) for value in raw["seeds"]),
            burn_in_steps=int(raw["burn_in_steps"]),
            validation_horizon_steps=int(raw["validation_horizon_steps"]),
            validation_window_seed=int(raw["validation_window_seed"]),
            validation_windows=int(raw["validation_windows"]),
            history_log_interval=int(raw["history_log_interval"]),
            base_stages=tuple(
                CurriculumStage.from_dict(value) for value in raw["base_stages"]
            ),
            multiscale_stages=tuple(
                CurriculumStage.from_dict(value)
                for value in raw["multiscale_stages"]
            ),
        )

    def stages(self, name: str) -> tuple[CurriculumStage, ...]:
        if name == "base":
            return self.base_stages
        if name == "multiscale":
            return self.multiscale_stages
        raise SpecValidationError(f"unknown Phase-7 curriculum: {name}")


@dataclass(frozen=True)
class Phase7Spec:
    schema_version: int
    phase: dict[str, Any]
    interface: dict[str, Any]
    dataset: dict[str, Any]
    training: dict[str, Any]
    variants: tuple[Phase7Variant, ...]
    objective: dict[str, Any]
    profiles: dict[str, Phase7Profile]
    test_bank: dict[str, Any]
    validation: dict[str, Any]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Phase7Spec:
        spec = cls(
            schema_version=int(raw["schema_version"]),
            phase=dict(raw["phase"]),
            interface=dict(raw["interface"]),
            dataset=dict(raw["dataset"]),
            training=dict(raw["training"]),
            variants=tuple(Phase7Variant.from_dict(value) for value in raw["variants"]),
            objective=dict(raw["objective"]),
            profiles={
                name: Phase7Profile.from_dict(name, values)
                for name, values in raw["profiles"].items()
            },
            test_bank=dict(raw["test_bank"]),
            validation=dict(raw["validation"]),
        )
        spec.validate()
        return spec

    def profile(self, name: str) -> Phase7Profile:
        try:
            return self.profiles[name]
        except KeyError as error:
            raise SpecValidationError(f"unknown Phase-7 profile: {name}") from error

    def validate(self) -> None:
        errors: list[str] = []
        if self.schema_version != 1:
            errors.append("schema_version must be 1")
        if self.phase.get("number") != 7 or self.phase.get("status") != "implemented":
            errors.append("Phase 7 must be numbered 7 and marked implemented")
        if tuple(self.interface.get("observation", ())) != (
            "armature_current_a",
            "angular_speed_rad_s",
        ):
            errors.append("Phase-7 observations must be current and angular speed")
        if tuple(self.interface.get("control", ())) != ("armature_voltage_v",):
            errors.append("Phase-7 control must be applied armature voltage")
        if tuple(self.interface.get("forbidden_training_features", ())) != (
            FORBIDDEN_TRAINING_FEATURES
        ):
            errors.append("Phase-7 forbidden feature catalog changed")
        if self.interface.get("hidden_evaluation_only") != ["winding_temperature_c"]:
            errors.append("temperature must remain evaluation-only")
        if self.dataset.get("required_version") != "2.0.0":
            errors.append("Phase 7 must require the broadband v2 dataset")
        if self.dataset.get("normalization_fit_split") != "train":
            errors.append("Phase-7 normalization must remain train-only")
        if self.training.get("model_type") != "ContractingR2DN":
            errors.append("Phase-7 primary model must remain ContractingR2DN")
        if self.training.get("absolute_state_output_retained") is not True:
            errors.append("Phase 7 must retain the certified absolute-state interface")
        if self.training.get("polar_parameterization") is not True:
            errors.append("Phase 7 must retain polar contraction parameterization")
        if self.training.get("selection_split") != "validation":
            errors.append("Phase-7 selection must use validation only")
        if int(self.training.get("latent_size", 0)) != 16:
            errors.append("Phase-7 comparison must start from latent size 16")

        names = tuple(value.name for value in self.variants)
        if names != (
            "broadband_standard",
            "broadband_delta_multiscale",
            "broadband_delta_multiscale_wide",
        ):
            errors.append("Phase-7 variant catalog or order changed")
        for variant in self.variants:
            if variant.curriculum not in {"base", "multiscale"}:
                errors.append(f"{variant.name}: unknown curriculum")
            if variant.feature_size < 1 or not variant.hidden_sizes:
                errors.append(f"{variant.name}: invalid architecture width")
            if not math.isfinite(variant.delta_weight) or variant.delta_weight < 0.0:
                errors.append(f"{variant.name}: invalid delta loss weight")

        horizons = tuple(int(value) for value in self.objective["rollout_horizon_steps"])
        weights = tuple(float(value) for value in self.objective["rollout_horizon_weights"])
        if horizons != (1, 10, 100, 1000, 5000):
            errors.append("Phase-7 multi-horizon catalog changed")
        if len(weights) != len(horizons) or any(value <= 0.0 for value in weights):
            errors.append("Phase-7 horizon weights must be positive and aligned")
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-12):
            errors.append("Phase-7 horizon weights must sum to one")

        if set(self.profiles) != {"ci", "final"}:
            errors.append("Phase 7 must define ci and final profiles")
        else:
            final = self.profiles["final"]
            if final.seeds != (17, 29, 43) or final.burn_in_steps != 250:
                errors.append("final Phase-7 seeds or burn-in changed")
            if final.validation_horizon_steps != 5000:
                errors.append("final Phase-7 selection must use a 5000-step rollout")
            if max(stage.rollout_steps for stage in final.multiscale_stages) != 5000:
                errors.append("multiscale curriculum must reach 5000 steps")
            for profile in self.profiles.values():
                if not profile.seeds or len(set(profile.seeds)) != len(profile.seeds):
                    errors.append(f"{profile.name}: seeds must be non-empty and unique")
                if min(
                    profile.burn_in_steps,
                    profile.validation_horizon_steps,
                    profile.validation_windows,
                    profile.history_log_interval,
                ) < 1:
                    errors.append(f"{profile.name}: invalid training budget")
                for stage in (*profile.base_stages, *profile.multiscale_stages):
                    errors.extend(stage.validate())

        if tuple(self.test_bank.get("categories", ())) != TEST_BANK_CATEGORIES:
            errors.append("test-bank category catalog changed")
        if tuple(self.test_bank.get("scenario_names", ())) != TEST_BANK_SCENARIO_NAMES:
            errors.append("test-bank scenario catalog changed")
        if self.test_bank.get("scenario_source") != "phase7_locked_long_horizon":
            errors.append("test bank must use its Phase-7 long-horizon scenarios")
        if tuple(self.test_bank.get("scenarios", ())) != LOCKED_TEST_BANK_SCENARIOS:
            errors.append("locked thermally safe test-bank scenarios changed")
        if (
            int(self.test_bank.get("minimum_anchor_burn_in_steps", 0))
            != TEST_BANK_MINIMUM_ANCHOR_BURN_IN_STEPS
        ):
            errors.append("test-bank minimum anchor burn-in must remain 250 steps")
        if not math.isclose(
            float(self.test_bank.get("preflight_maximum_temperature_c", math.nan)),
            TEST_BANK_MAXIMUM_TEMPERATURE_C,
            abs_tol=1e-12,
        ):
            errors.append("test-bank preflight temperature ceiling must remain 110 C")
        if tuple(float(value) for value in self.test_bank.get("horizons_s", ())) != (
            1.0,
            10.0,
            100.0,
            1000.0,
        ):
            errors.append("test-bank horizons must be 1/10/100/1000 s")
        if self.test_bank.get("evaluation_used_for_selection") is not False:
            errors.append("test bank must remain excluded from model selection")
        if int(self.test_bank.get("ci_cases_per_category", 0)) != 1:
            errors.append("CI test bank must use one case per category")
        if int(self.test_bank.get("final_cases_per_category", 0)) < 3:
            errors.append("final test bank must use at least three cases per category")
        if set(self.validation.get("required_checks", ())) != REQUIRED_PHASE7_CHECKS:
            errors.append("Phase-7 validation check catalog changed")
        if errors:
            raise SpecValidationError("\n".join(f"- {error}" for error in errors))

    def summary(self, profile_name: str = "final") -> str:
        profile = self.profile(profile_name)
        return "\n".join(
            (
                "PHASE 7 SPEC: PASS",
                "primary model: pure ContractingR2DN (temperature hidden)",
                f"dataset: {self.dataset['required_dataset_id']} v2.0.0",
                f"variants: {', '.join(value.name for value in self.variants)}",
                f"seeds: {list(profile.seeds)}",
                f"burn-in: {profile.burn_in_steps} steps",
                f"selection horizon: {profile.validation_horizon_steps} steps",
                "final test horizons: 1, 10, 100, 1000 s",
            )
        )


def load_phase7_spec(path: Path | str | None = None) -> Phase7Spec:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "phase7.toml"
    return Phase7Spec.from_dict(tomllib.loads(Path(path).read_text(encoding="utf-8")))
