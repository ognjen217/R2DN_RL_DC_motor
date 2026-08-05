"""Typed Phase-6B contract for latent search and autoregressive stress tests."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from r2dn_dc_motor.phase2_spec import Phase2Spec, load_phase2_spec
from r2dn_dc_motor.phase4_spec import Phase4Spec, load_phase4_spec
from r2dn_dc_motor.phase6_spec import Phase6Spec, load_phase6_spec
from r2dn_dc_motor.spec import SpecValidationError

STRESS_SPLITS = ("validation", "id_test", "ood_test")
STRESS_SCENARIOS = (
    "zero_voltage",
    "constant_positive",
    "constant_negative",
    "prbs",
    "sine",
    "multisine",
    "positive_voltage_limit",
    "negative_voltage_limit",
)
REQUIRED_PHASE6B_CHECKS = {
    "latent_catalog_is_fully_evaluated",
    "pilot_uses_three_seeds_and_fixed_validation_windows",
    "architecture_selection_uses_locked_median_and_tie_rule",
    "final_checkpoint_uses_locked_validation_selection",
    "selection_excludes_id_ood_and_stress_results",
    "held_out_replay_is_finite",
    "long_stress_rollouts_are_finite_and_physically_bounded",
    "zero_input_has_no_sustained_tail_energy_growth",
    "autoregressive_perturbations_have_no_sustained_tail_growth",
}


@dataclass(frozen=True)
class Phase6BProfile:
    """One search and stress-test budget."""

    name: str
    base_phase6_profile: str
    candidate_latent_sizes: tuple[int, ...]
    pilot_seeds: tuple[int, ...]
    pilot_validation_horizon_steps: int
    pilot_validation_seed: int
    validation_windows: int
    tie_relative_tolerance: float
    adaptive_enabled: bool
    adaptive_reference_latent: int
    adaptive_boundary_latent: int
    adaptive_candidate_latent: int
    adaptive_improvement_threshold: float
    final_training_seeds: tuple[int, ...]
    selection_horizon_steps: int
    selection_validation_seed: int
    stress_horizon_steps: tuple[int, ...]
    stress_chunk_steps: int
    stress_anchors_per_split: int
    stress_anchor_seed: int
    perturbation_epsilon_normalized: float
    replay_horizon_steps: tuple[int, ...]
    replay_windows_per_split: int
    replay_window_seed: int

    @classmethod
    def from_dict(cls, name: str, raw: dict[str, Any]) -> Phase6BProfile:
        return cls(
            name=name,
            base_phase6_profile=str(raw["base_phase6_profile"]),
            candidate_latent_sizes=tuple(int(value) for value in raw["candidate_latent_sizes"]),
            pilot_seeds=tuple(int(value) for value in raw["pilot_seeds"]),
            pilot_validation_horizon_steps=int(raw["pilot_validation_horizon_steps"]),
            pilot_validation_seed=int(raw["pilot_validation_seed"]),
            validation_windows=int(raw["validation_windows"]),
            tie_relative_tolerance=float(raw["tie_relative_tolerance"]),
            adaptive_enabled=bool(raw["adaptive_enabled"]),
            adaptive_reference_latent=int(raw["adaptive_reference_latent"]),
            adaptive_boundary_latent=int(raw["adaptive_boundary_latent"]),
            adaptive_candidate_latent=int(raw["adaptive_candidate_latent"]),
            adaptive_improvement_threshold=float(raw["adaptive_improvement_threshold"]),
            final_training_seeds=tuple(int(value) for value in raw["final_training_seeds"]),
            selection_horizon_steps=int(raw["selection_horizon_steps"]),
            selection_validation_seed=int(raw["selection_validation_seed"]),
            stress_horizon_steps=tuple(int(value) for value in raw["stress_horizon_steps"]),
            stress_chunk_steps=int(raw["stress_chunk_steps"]),
            stress_anchors_per_split=int(raw["stress_anchors_per_split"]),
            stress_anchor_seed=int(raw["stress_anchor_seed"]),
            perturbation_epsilon_normalized=float(
                raw["perturbation_epsilon_normalized"]
            ),
            replay_horizon_steps=tuple(int(value) for value in raw["replay_horizon_steps"]),
            replay_windows_per_split=int(raw["replay_windows_per_split"]),
            replay_window_seed=int(raw["replay_window_seed"]),
        )

    def validate(self, phase6: Phase6Spec) -> list[str]:
        errors: list[str] = []
        base = phase6.profile(self.base_phase6_profile)
        if not self.candidate_latent_sizes or len(set(self.candidate_latent_sizes)) != len(
            self.candidate_latent_sizes
        ):
            errors.append(f"{self.name}: latent candidates must be non-empty and unique")
        if tuple(sorted(self.candidate_latent_sizes)) != self.candidate_latent_sizes:
            errors.append(f"{self.name}: latent candidates must be strictly increasing")
        if min(self.candidate_latent_sizes, default=0) <= 2:
            errors.append(f"{self.name}: latent states must exceed measured output size")
        if len(self.pilot_seeds) < 3 or len(set(self.pilot_seeds)) != len(self.pilot_seeds):
            errors.append(f"{self.name}: at least three unique pilot seeds are required")
        if min(self.pilot_seeds, default=-1) < 0:
            errors.append(f"{self.name}: pilot seeds must be non-negative")
        if not (0.0 <= self.tie_relative_tolerance < 1.0):
            errors.append(f"{self.name}: tie tolerance must lie in [0, 1)")
        positive = (
            self.pilot_validation_horizon_steps,
            self.validation_windows,
            self.selection_horizon_steps,
            self.stress_chunk_steps,
            self.stress_anchors_per_split,
            self.replay_windows_per_split,
        )
        if min(positive) < 1:
            errors.append(f"{self.name}: horizons, windows, chunks, and anchors must be positive")
        seeds = (
            self.pilot_validation_seed,
            self.selection_validation_seed,
            self.stress_anchor_seed,
            self.replay_window_seed,
        )
        if min(seeds) < 0:
            errors.append(f"{self.name}: deterministic evaluation seeds must be non-negative")
        if not self.final_training_seeds:
            errors.append(f"{self.name}: final training seeds must not be empty")
        if self.final_training_seeds != base.training_seeds:
            errors.append(f"{self.name}: final seeds drifted from base Phase 6")
        if self.pilot_validation_horizon_steps != base.pilot_validation_horizon_steps:
            errors.append(f"{self.name}: pilot validation horizon drifted from base Phase 6")
        if self.pilot_validation_seed != base.pilot_validation_seed:
            errors.append(f"{self.name}: pilot validation windows drifted from base Phase 6")
        if self.selection_horizon_steps != base.selection_horizon_steps:
            errors.append(f"{self.name}: final selection horizon drifted from base Phase 6")
        if self.selection_validation_seed != base.selection_validation_seed:
            errors.append(f"{self.name}: final validation windows drifted from base Phase 6")
        if tuple(sorted(self.stress_horizon_steps)) != self.stress_horizon_steps:
            errors.append(f"{self.name}: stress horizons must be increasing")
        if tuple(sorted(self.replay_horizon_steps)) != self.replay_horizon_steps:
            errors.append(f"{self.name}: replay horizons must be increasing")
        if any(horizon % self.stress_chunk_steps for horizon in self.stress_horizon_steps):
            errors.append(f"{self.name}: stress chunk must divide every stress horizon")
        if self.perturbation_epsilon_normalized <= 0.0:
            errors.append(f"{self.name}: perturbation epsilon must be positive")
        if self.adaptive_enabled:
            required = {
                self.adaptive_reference_latent,
                self.adaptive_boundary_latent,
            }
            if not required.issubset(self.candidate_latent_sizes):
                errors.append(f"{self.name}: adaptive comparison latents must be base candidates")
            if self.adaptive_candidate_latent <= max(self.candidate_latent_sizes):
                errors.append(f"{self.name}: adaptive candidate must extend the search boundary")
            if not (0.0 < self.adaptive_improvement_threshold < 1.0):
                errors.append(f"{self.name}: adaptive threshold must lie in (0, 1)")
        return errors


@dataclass(frozen=True)
class Phase6BSpec:
    """Executable source of truth for Phase 6B."""

    schema_version: int
    phase: dict[str, Any]
    search: dict[str, Any]
    stress: dict[str, Any]
    profiles: dict[str, Phase6BProfile]
    validation: dict[str, Any]

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
        *,
        phase2: Phase2Spec | None = None,
        phase4: Phase4Spec | None = None,
        phase6: Phase6Spec | None = None,
    ) -> Phase6BSpec:
        spec = cls(
            schema_version=int(raw["schema_version"]),
            phase=dict(raw["phase"]),
            search=dict(raw["search"]),
            stress=dict(raw["stress"]),
            profiles={
                name: Phase6BProfile.from_dict(name, values)
                for name, values in raw["profiles"].items()
            },
            validation=dict(raw["validation"]),
        )
        spec.validate(
            phase2=phase2 or load_phase2_spec(),
            phase4=phase4 or load_phase4_spec(),
            phase6=phase6 or load_phase6_spec(),
        )
        return spec

    @property
    def scenarios(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(value) for value in self.stress["scenarios"])

    def profile(self, name: str) -> Phase6BProfile:
        try:
            return self.profiles[name]
        except KeyError as error:
            choices = ", ".join(sorted(self.profiles))
            raise SpecValidationError(
                f"unknown Phase-6B profile {name!r}; choose one of: {choices}"
            ) from error

    def validate(
        self,
        *,
        phase2: Phase2Spec,
        phase4: Phase4Spec,
        phase6: Phase6Spec,
    ) -> None:
        errors: list[str] = []
        if self.schema_version != 1:
            errors.append("schema_version must be 1")
        if self.phase.get("number") != "6B" or self.phase.get("status") != "implemented":
            errors.append("Phase 6B must be named 6B and marked implemented")
        locked_search = {
            "fit_split": "train",
            "selection_split": "validation",
            "id_test_used_for_selection": False,
            "ood_test_used_for_selection": False,
            "aggregation": "median",
            "mode": "min",
            "near_tie_policy": "smallest_within_relative_tolerance",
            "reuse_phase6_checkpoint_when_selected_latent_matches": True,
        }
        for key, expected in locked_search.items():
            if self.search.get(key) != expected:
                errors.append(f"Phase-6B search rule changed: {key}")
        if tuple(self.stress.get("splits", ())) != STRESS_SPLITS:
            errors.append("stress splits must be validation, ID, and OOD")
        if self.stress.get("synthetic_stress_used_for_selection") is not False:
            errors.append("synthetic stress results must not select the architecture")
        if self.stress.get("held_out_replay_used_for_selection") is not False:
            errors.append("held-out replay results must not select the architecture")
        if self.stress.get("energy_definition") != "electromagnetic_plus_kinetic":
            errors.append("Phase-6B energy definition changed")
        if not math.isclose(
            float(self.stress.get("control_period_s", math.nan)),
            phase2.integration_settings.control_period_s,
        ):
            errors.append("stress control period drifted from the FULL plant")
        scenarios = self.scenarios
        if tuple(value.get("name") for value in scenarios) != STRESS_SCENARIOS:
            errors.append("stress scenario catalog or order changed")
        safe_voltage = float(phase4.domain["safe_voltage_limit_v"])
        for scenario in scenarios:
            errors.extend(_validate_scenario(scenario, safe_voltage))
        if set(self.profiles) != {"ci", "final"}:
            errors.append("Phase 6B must define exactly ci and final profiles")
        else:
            for profile in self.profiles.values():
                errors.extend(profile.validate(phase6))
            final = self.profiles["final"]
            if final.candidate_latent_sizes != (4, 6, 8, 10, 12, 16):
                errors.append("final Phase-6B latent catalog must be [4, 6, 8, 10, 12, 16]")
            if final.pilot_seeds != (1701, 2701, 3701):
                errors.append("final Phase-6B pilot seeds changed")
            if final.adaptive_candidate_latent != 24:
                errors.append("adaptive extension must test latent 24")
            if final.stress_horizon_steps != (10_000, 100_000, 1_000_000):
                errors.append("final stress horizons must be 10^4, 10^5, and 10^6 steps")
            if final.replay_horizon_steps != (1_000, 5_000, 10_000):
                errors.append("final replay horizons must be 1 s, 5 s, and 10 s")
        if set(self.validation.get("required_checks", ())) != REQUIRED_PHASE6B_CHECKS:
            errors.append("Phase-6B validation check catalog changed")
        if self.validation.get("phase7_gate_claimed") is not False:
            errors.append("Phase 6B must not claim the Phase-7 comparison Gate")
        thresholds = (
            "maximum_absolute_normalized_prediction",
            "maximum_latent_norm",
            "tail_fraction",
            "latent_tail_growth_ratio_limit",
            "unforced_energy_tail_growth_ratio_limit",
            "perturbation_tail_growth_ratio_limit",
            "perturbation_absolute_rms_limit_normalized",
        )
        if any(
            not math.isfinite(float(self.validation.get(key, math.nan)))
            or float(self.validation[key]) <= 0.0
            for key in thresholds
        ):
            errors.append("Phase-6B validation thresholds must be finite and positive")
        if not 0.0 < float(self.validation["tail_fraction"]) <= 0.5:
            errors.append("tail fraction must lie in (0, 0.5]")
        if errors:
            raise SpecValidationError("\n".join(f"- {error}" for error in errors))

    def summary(self, profile_name: str = "final") -> str:
        profile = self.profile(profile_name)
        adaptive = (
            f"; add {profile.adaptive_candidate_latent} only if "
            f"{profile.adaptive_boundary_latent} improves over "
            f"{profile.adaptive_reference_latent} by >"
            f"{100 * profile.adaptive_improvement_threshold:g}%"
            if profile.adaptive_enabled
            else ""
        )
        return "\n".join(
            [
                "PHASE 6B SPEC: PASS",
                f"profile: {profile.name}",
                f"base latent candidates: {list(profile.candidate_latent_sizes)}{adaptive}",
                f"pilot seeds: {list(profile.pilot_seeds)}",
                (
                    "architecture selection: median validation free-rollout NRMSE; "
                    f"prefer smaller within {100 * profile.tie_relative_tolerance:g}%"
                ),
                f"final seeds: {list(profile.final_training_seeds)}",
                f"stress horizons: {list(profile.stress_horizon_steps)} steps",
                f"stress splits: {list(STRESS_SPLITS)}",
                "ID/OOD, replay, and stress results do not select the model",
                "Phase-7 comparison Gate claimed: no",
            ]
        )


def _validate_scenario(scenario: dict[str, Any], safe_voltage: float) -> list[str]:
    errors: list[str] = []
    name = str(scenario.get("name", "<unnamed>"))
    kind = scenario.get("kind")
    if kind == "constant":
        maximum = abs(float(scenario.get("amplitude_v", math.nan)))
    elif kind == "prbs":
        maximum = abs(float(scenario.get("amplitude_v", math.nan)))
        if int(scenario.get("hold_steps", 0)) < 1 or int(scenario.get("seed", -1)) < 0:
            errors.append(f"{name}: PRBS hold and seed are invalid")
    elif kind == "sine":
        maximum = abs(float(scenario.get("amplitude_v", math.nan)))
        if float(scenario.get("frequency_hz", 0.0)) <= 0.0:
            errors.append(f"{name}: sine frequency must be positive")
    elif kind == "multisine":
        amplitudes = tuple(float(value) for value in scenario.get("amplitudes_v", ()))
        frequencies = tuple(float(value) for value in scenario.get("frequencies_hz", ()))
        phases = tuple(float(value) for value in scenario.get("phases_rad", ()))
        maximum = sum(abs(value) for value in amplitudes)
        if (
            not amplitudes
            or len(amplitudes) != len(frequencies)
            or len(amplitudes) != len(phases)
            or min(frequencies, default=0.0) <= 0.0
        ):
            errors.append(f"{name}: multisine vectors are inconsistent")
    else:
        return [f"{name}: unsupported stress scenario kind"]
    if not math.isfinite(maximum) or maximum > safe_voltage:
        errors.append(f"{name}: stress voltage exceeds the Phase-4 safe limit")
    return errors


def load_phase6b_spec(
    path: Path | str | None = None,
    *,
    phase2: Phase2Spec | None = None,
    phase4: Phase4Spec | None = None,
    phase6: Phase6Spec | None = None,
) -> Phase6BSpec:
    """Load and validate the canonical Phase-6B TOML contract."""

    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "phase6b.toml"
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    return Phase6BSpec.from_dict(
        raw,
        phase2=phase2,
        phase4=phase4,
        phase6=phase6,
    )
