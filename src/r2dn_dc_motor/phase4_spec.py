"""Typed Phase-4 dataset-generation specification."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from r2dn_dc_motor.phase2_spec import Phase2Spec, load_phase2_spec
from r2dn_dc_motor.spec import ExperimentSpec, SpecValidationError, load_phase0_spec

SPLIT_NAMES = ("train", "validation", "id_test", "ood_test")
EXCITATION_FAMILIES = (
    "prbs_voltage",
    "piecewise_constant_voltage",
    "step_ramp_voltage",
    "multisine_voltage",
    "pi_closed_loop",
    "pi_identification",
    "random_speed_reference",
    "heating_cooling_cycle",
)
OOD_AXES = (
    "higher_initial_temperature",
    "stronger_load",
    "novel_profile",
    "physical_parameter_shift",
)
REQUIRED_INTEGRITY_CHECKS = {
    "full_simulator_is_the_only_source",
    "whole_trajectory_splits_are_disjoint",
    "all_excitation_families_are_covered",
    "trajectories_are_complete_and_finite",
    "signals_stay_inside_declared_limits",
    "raw_temperature_is_retained",
    "model_view_excludes_temperature",
    "normalization_uses_train_only",
    "seeds_and_content_are_reproducible",
    "id_and_ood_domains_are_separated",
}


@dataclass(frozen=True)
class DatasetProfile:
    """One deterministic dataset-size profile."""

    name: str
    seed_base: int
    split_counts: dict[str, int]
    default_duration_s: float
    heating_cooling_duration_s: float
    minimum_total_transitions: int

    @property
    def trajectory_count(self) -> int:
        return sum(self.split_counts.values())

    def duration_for(self, excitation_family: str) -> float:
        if excitation_family == "heating_cooling_cycle":
            return self.heating_cooling_duration_s
        return self.default_duration_s


@dataclass(frozen=True)
class Phase4Spec:
    """Executable source of truth for the versioned FULL dataset."""

    schema_version: int
    phase: dict[str, Any]
    dataset: dict[str, Any]
    features: dict[str, Any]
    splits: dict[str, Any]
    normalization: dict[str, Any]
    domain: dict[str, Any]
    ood: dict[str, Any]
    excitation: dict[str, Any]
    profiles: dict[str, DatasetProfile]
    integrity: dict[str, Any]

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
        *,
        phase0: ExperimentSpec | None = None,
        phase2: Phase2Spec | None = None,
    ) -> Phase4Spec:
        profiles = {
            name: DatasetProfile(
                name=name,
                seed_base=int(values["seed_base"]),
                split_counts={
                    split: int(values[f"{split}_trajectories"])
                    for split in SPLIT_NAMES
                },
                default_duration_s=float(values["default_duration_s"]),
                heating_cooling_duration_s=float(
                    values["heating_cooling_duration_s"]
                ),
                minimum_total_transitions=int(
                    values["minimum_total_transitions"]
                ),
            )
            for name, values in raw["profiles"].items()
        }
        spec = cls(
            schema_version=int(raw["schema_version"]),
            phase=dict(raw["phase"]),
            dataset=dict(raw["dataset"]),
            features=dict(raw["features"]),
            splits=dict(raw["splits"]),
            normalization=dict(raw["normalization"]),
            domain=dict(raw["domain"]),
            ood=dict(raw["ood"]),
            excitation=dict(raw["excitation"]),
            profiles=profiles,
            integrity=dict(raw["integrity"]),
        )
        spec.validate(
            phase0=phase0 or load_phase0_spec(),
            phase2=phase2 or load_phase2_spec(),
        )
        return spec

    def profile(self, name: str) -> DatasetProfile:
        try:
            return self.profiles[name]
        except KeyError as error:
            choices = ", ".join(sorted(self.profiles))
            raise SpecValidationError(
                f"unknown Phase-4 profile {name!r}; choose one of: {choices}"
            ) from error

    def steps(self, duration_s: float, phase2: Phase2Spec) -> int:
        raw = duration_s / phase2.integration_settings.control_period_s
        steps = int(round(raw))
        if not math.isclose(raw, steps, rel_tol=0.0, abs_tol=1e-12):
            raise SpecValidationError(
                "trajectory duration must contain an integer number of control periods"
            )
        return steps

    def validate(self, *, phase0: ExperimentSpec, phase2: Phase2Spec) -> None:
        errors: list[str] = []
        if self.schema_version != 1:
            errors.append("schema_version must be 1")
        if self.phase.get("number") != 4:
            errors.append("phase.number must be 4")
        if self.phase.get("status") != "implemented":
            errors.append("Phase 4 status must be implemented")
        if self.dataset.get("simulator") != "FULL":
            errors.append("Phase-4 data may only come from the FULL simulator")
        if self.dataset.get("storage_format") != "npz_per_trajectory":
            errors.append("Phase-4 storage must preserve one NPZ per whole trajectory")
        if self.dataset.get("dtype") != "float32":
            errors.append("Phase-4 persisted arrays must use float32")
        if self.dataset.get("version") != "1.0.0":
            errors.append("Phase-4 dataset version must be 1.0.0")
        if not math.isclose(
            float(self.dataset.get("control_period_s", -1.0)),
            phase2.integration_settings.control_period_s,
        ):
            errors.append("dataset control period drifted from the FULL simulator")

        expected_state = tuple(phase0.signals.state)
        if tuple(self.features.get("state", ())) != expected_state:
            errors.append("raw state feature order drifted from Phase 0")
        if tuple(self.features.get("model_observation", ())) != tuple(
            phase0.signals.plant_output
        ):
            errors.append("model observation feature order drifted from Phase 0")
        if tuple(self.features.get("model_control", ())) != (
            phase0.signals.control,
        ):
            errors.append("model control feature drifted from Phase 0")
        model_features = set(self.features.get("model_observation", ())) | set(
            self.features.get("model_control", ())
        )
        if phase0.signals.hidden_state[0] in model_features:
            errors.append("hidden temperature leaked into the Phase-4 model view")
        if phase0.signals.known_disturbance in model_features:
            errors.append("load torque leaked into the frozen R2DN interface")
        if phase0.signals.hidden_state[0] not in set(
            self.features.get("evaluation_only", ())
        ):
            errors.append("raw temperature must remain available for evaluation")

        if tuple(self.splits.get("names", ())) != SPLIT_NAMES:
            errors.append("dataset split catalog or order changed")
        if self.splits.get("policy") != "disjoint_whole_trajectories":
            errors.append("Phase-4 splitting must operate on whole trajectories")
        if tuple(self.splits.get("ood_axes", ())) != OOD_AXES:
            errors.append("OOD axis catalog changed")
        if self.normalization.get("fit_split") != "train":
            errors.append("normalization must be fit on train only")
        if self.normalization.get("temperature_used") is not False:
            errors.append("temperature must not enter R2DN normalization")
        if tuple(self.normalization.get("observation_features", ())) != tuple(
            self.features.get("model_observation", ())
        ):
            errors.append("normalization observation features changed")
        if tuple(self.normalization.get("control_features", ())) != tuple(
            self.features.get("model_control", ())
        ):
            errors.append("normalization control features changed")
        if float(self.normalization.get("standard_deviation_floor", 0.0)) <= 0.0:
            errors.append("normalization standard-deviation floor must be positive")

        errors.extend(self._validate_ranges(phase0, phase2))
        if tuple(self.excitation.get("families", ())) != EXCITATION_FAMILIES:
            errors.append("excitation family catalog changed")
        if self.excitation.get("anti_windup") != "conditional_integration":
            errors.append("closed-loop dataset trajectories require conditional anti-windup")

        if set(self.profiles) != {"ci", "final"}:
            errors.append("Phase 4 must define exactly ci and final profiles")
        else:
            errors.extend(self._validate_profiles(phase2))

        if set(self.integrity.get("required_checks", ())) != REQUIRED_INTEGRITY_CHECKS:
            errors.append("Phase-4 integrity check catalog changed")
        if int(self.integrity.get("minimum_train_temperature_bands", 0)) != 3:
            errors.append("training must retain cold, medium, and hot temperature bands")
        final = self.profiles.get("final")
        if final is not None:
            if final.trajectory_count < int(
                self.integrity.get("minimum_final_trajectories", 0)
            ):
                errors.append("final profile has too few trajectories")
            if final.minimum_total_transitions < int(
                self.integrity.get("minimum_final_transitions", 0)
            ):
                errors.append("final profile has too few transitions")

        if errors:
            raise SpecValidationError("\n".join(f"- {error}" for error in errors))

    def _validate_ranges(
        self,
        phase0: ExperimentSpec,
        phase2: Phase2Spec,
    ) -> list[str]:
        errors: list[str] = []
        pairs = (
            ("initial_current_a", phase2.limits["armature_current_a"]),
            ("initial_speed_rad_s", phase2.limits["angular_speed_rad_s"]),
            ("speed_reference_rad_s", phase0.limits["angular_speed_reference_rad_s"]),
        )
        for name, limit in pairs:
            try:
                low, high = _pair(self.domain[name])
            except (KeyError, TypeError, ValueError) as error:
                errors.append(f"invalid domain range {name}: {error}")
                continue
            if low >= high or low < limit.minimum or high > limit.maximum:
                errors.append(f"domain range {name} must stay inside declared limits")

        voltage_limit = float(self.domain.get("safe_voltage_limit_v", 0.0))
        voltage = phase2.limits["armature_voltage_v"]
        if not (0.0 < voltage_limit <= min(abs(voltage.minimum), voltage.maximum)):
            errors.append("safe voltage limit must be positive and inside actuator limits")
        current_guard = float(self.domain.get("current_guard_a", 0.0))
        current = phase2.limits["armature_current_a"]
        if not (
            0.0
            < current_guard
            < min(abs(current.minimum), current.maximum)
        ):
            errors.append("current guard must be positive and strictly inside current limits")
        if not math.isclose(
            float(self.domain.get("nominal_load_torque_n_m", math.nan)),
            phase2.motor_parameters.default_load_torque_n_m,
        ):
            errors.append("core train/validation/ID load must equal the nominal FULL load")

        bands = self.domain.get("temperature_bands_c", ())
        if len(bands) != 3:
            errors.append("training domain must define exactly three temperature bands")
        for band in bands:
            low, high = _pair(band)
            limits = phase2.limits["winding_temperature_c"]
            if low >= high or low < limits.minimum or high > limits.maximum:
                errors.append("temperature bands must be ordered and inside plant limits")

        ood_temperature = _pair(self.ood["higher_initial_temperature_c"])
        training_maximum = max(float(band[1]) for band in bands)
        if ood_temperature[0] <= training_maximum:
            errors.append("OOD temperature must start above the training domain")
        if ood_temperature[1] > phase2.limits["winding_temperature_c"].maximum:
            errors.append("OOD temperature exceeds the FULL plant domain")
        ood_load = _pair(self.ood["stronger_load_torque_n_m"])
        if ood_load[0] <= float(self.domain["nominal_load_torque_n_m"]):
            errors.append("OOD load must be stronger than the nominal training load")
        for name in (
            "reference_resistance_scale",
            "inertia_scale",
            "thermal_capacitance_scale",
        ):
            low, high = _pair(self.ood[name])
            if low <= 0.0 or not (low < 1.0 < high):
                errors.append(f"{name} must be positive and bracket the nominal value")
        return errors

    def _validate_profiles(self, phase2: Phase2Spec) -> list[str]:
        errors: list[str] = []
        used_seeds: set[int] = set()
        for profile in self.profiles.values():
            if profile.seed_base < 0:
                errors.append(f"{profile.name}: seed base must be non-negative")
            if profile.default_duration_s <= 0.0:
                errors.append(f"{profile.name}: default duration must be positive")
            if profile.heating_cooling_duration_s <= profile.default_duration_s:
                errors.append(
                    f"{profile.name}: heating/cooling duration must exceed default duration"
                )
            expected_transitions = 0
            seed_cursor = profile.seed_base
            for split in SPLIT_NAMES:
                count = profile.split_counts.get(split, 0)
                if count < len(EXCITATION_FAMILIES):
                    errors.append(
                        f"{profile.name}.{split}: every excitation family needs a trajectory"
                    )
                if count % len(EXCITATION_FAMILIES) != 0:
                    errors.append(
                        f"{profile.name}.{split}: trajectory count must be divisible by "
                        f"{len(EXCITATION_FAMILIES)}"
                    )
                repeats = count // len(EXCITATION_FAMILIES)
                default_steps = self.steps(profile.default_duration_s, phase2)
                thermal_steps = self.steps(
                    profile.heating_cooling_duration_s,
                    phase2,
                )
                expected_transitions += repeats * (
                    default_steps * (len(EXCITATION_FAMILIES) - 1)
                    + thermal_steps
                )
                split_seeds = set(range(seed_cursor, seed_cursor + count))
                if used_seeds & split_seeds:
                    errors.append(f"{profile.name}: seed ranges overlap")
                used_seeds |= split_seeds
                seed_cursor += count
            if expected_transitions != profile.minimum_total_transitions:
                errors.append(
                    f"{profile.name}: declared transition count does not match durations"
                )
        return errors

    def summary(self, profile_name: str = "final") -> str:
        profile = self.profile(profile_name)
        counts = ", ".join(
            f"{split}={profile.split_counts[split]}" for split in SPLIT_NAMES
        )
        return "\n".join(
            [
                "PHASE 4 SPEC: PASS",
                f"dataset: {self.dataset['dataset_id']} v{self.dataset['version']}",
                "source simulator: FULL",
                f"profile: {profile.name} ({profile.trajectory_count} trajectories)",
                f"splits: {counts}",
                f"transitions: {profile.minimum_total_transitions}",
                "split policy: disjoint whole trajectories",
                "normalization fit: train only",
                "R2DN view: current, speed, applied voltage",
                "evaluation only: temperature, load, reference, commanded voltage",
            ]
        )


def _pair(value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("expected a two-element range")
    return float(value[0]), float(value[1])


def load_phase4_spec(
    path: Path | str | None = None,
    *,
    phase0: ExperimentSpec | None = None,
    phase2: Phase2Spec | None = None,
) -> Phase4Spec:
    """Load and validate the canonical Phase-4 TOML contract."""

    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "phase4.toml"
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return Phase4Spec.from_dict(raw, phase0=phase0, phase2=phase2)
