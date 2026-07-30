"""Integrity and coverage validation for the versioned Phase-4 dataset."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from r2dn_dc_motor.data.phase4_dataset import Phase4Dataset
from r2dn_dc_motor.data.phase4_generation import (
    _StreamingMoments,
    build_trajectory_plans,
)
from r2dn_dc_motor.phase2_spec import Phase2Spec, load_phase2_spec
from r2dn_dc_motor.phase4_spec import (
    EXCITATION_FAMILIES,
    OOD_AXES,
    REQUIRED_INTEGRITY_CHECKS,
    SPLIT_NAMES,
    Phase4Spec,
    load_phase4_spec,
)


@dataclass(frozen=True)
class DatasetIntegrityCheck:
    """One named Phase-4 integrity result."""

    name: str
    passed: bool
    metrics: dict[str, Any]
    criterion: str


@dataclass(frozen=True)
class Phase4ValidationReport:
    """Complete dataset integrity and coverage report."""

    passed: bool
    dataset_fingerprint: str
    profile: str
    trajectory_count: int
    transition_count: int
    split_counts: dict[str, int]
    excitation_counts: dict[str, dict[str, int]]
    signal_ranges: dict[str, list[float]]
    checks: tuple[DatasetIntegrityCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": 4,
            "validation": "dataset_integrity",
            "passed": self.passed,
            "dataset_fingerprint": self.dataset_fingerprint,
            "profile": self.profile,
            "trajectory_count": self.trajectory_count,
            "transition_count": self.transition_count,
            "split_counts": self.split_counts,
            "excitation_counts": self.excitation_counts,
            "signal_ranges": self.signal_ranges,
            "checks": [asdict(check) for check in self.checks],
        }

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"PHASE 4 DATASET INTEGRITY: {status}"]
        for check in self.checks:
            marker = "PASS" if check.passed else "FAIL"
            lines.append(f"[{marker}] {check.name}")
        lines.extend(
            [
                f"profile: {self.profile}",
                f"trajectories: {self.trajectory_count}",
                f"transitions: {self.transition_count}",
                f"fingerprint: {self.dataset_fingerprint}",
            ]
        )
        return "\n".join(lines)


def run_phase4_validation(
    dataset_root: Path | str,
    *,
    spec: Phase4Spec | None = None,
    phase2: Phase2Spec | None = None,
) -> Phase4ValidationReport:
    """Validate every persisted trajectory and the train-only normalizer."""

    phase2 = phase2 or load_phase2_spec()
    spec = spec or load_phase4_spec(phase2=phase2)
    dataset = Phase4Dataset(dataset_root)
    profile = spec.profile(str(dataset.manifest["profile"]))
    expected_plans = {
        plan.trajectory_id: plan
        for plan in build_trajectory_plans(spec, profile)
    }
    records = dataset.manifest["trajectories"]

    trajectory_data: dict[str, Any] = {}
    load_errors: list[str] = []
    for record in records:
        trajectory_id = str(record["trajectory_id"])
        try:
            trajectory_data[trajectory_id] = dataset.load_trajectory(trajectory_id)
        except (OSError, KeyError, TypeError, ValueError) as error:
            load_errors.append(f"{trajectory_id}: {error}")

    split_ids = {
        split: set(dataset.trajectory_ids(split))
        for split in SPLIT_NAMES
    }
    split_counts = {split: len(values) for split, values in split_ids.items()}
    excitation_counts = {
        split: {
            family: sum(
                record["split"] == split
                and record["excitation_family"] == family
                for record in records
            )
            for family in EXCITATION_FAMILIES
        }
        for split in SPLIT_NAMES
    }
    signal_ranges = _global_signal_ranges(trajectory_data)

    checks = (
        _check_full_source(records),
        _check_split_integrity(records, split_ids, profile.split_counts),
        _check_excitation_coverage(excitation_counts),
        _check_trajectory_completeness(
            records,
            trajectory_data,
            load_errors,
            profile.minimum_total_transitions,
        ),
        _check_signal_limits(signal_ranges, phase2),
        _check_temperature_retention(trajectory_data),
        _check_model_views(trajectory_data, spec),
        _check_normalization(dataset, trajectory_data, spec),
        _check_reproducible_identity(records, expected_plans),
        _check_id_ood_separation(records, spec),
    )
    passed = (
        {check.name for check in checks} == REQUIRED_INTEGRITY_CHECKS
        and all(check.passed for check in checks)
    )
    return Phase4ValidationReport(
        passed=passed,
        dataset_fingerprint=dataset.fingerprint,
        profile=profile.name,
        trajectory_count=len(records),
        transition_count=int(dataset.manifest["transition_count"]),
        split_counts=split_counts,
        excitation_counts=excitation_counts,
        signal_ranges=signal_ranges,
        checks=checks,
    )


def generate_phase4_validation_artifacts(
    dataset_root: Path | str,
    output_directory: Path | str,
    *,
    spec: Phase4Spec | None = None,
    phase2: Phase2Spec | None = None,
) -> tuple[Phase4ValidationReport, Path, Path]:
    """Write a JSON integrity report and compact dataset coverage figure."""

    report = run_phase4_validation(dataset_root, spec=spec, phase2=phase2)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "phase4_dataset_integrity.json"
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    figure_path = output / "phase4_dataset_coverage.png"
    _plot_coverage(report, figure_path)
    return report, report_path, figure_path


def _check_full_source(records: list[dict[str, Any]]) -> DatasetIntegrityCheck:
    sources = sorted({str(record.get("simulator")) for record in records})
    passed = bool(records) and sources == ["FULL"]
    return DatasetIntegrityCheck(
        name="full_simulator_is_the_only_source",
        passed=passed,
        metrics={"sources": sources},
        criterion="every trajectory must declare the Phase-2 FULL simulator",
    )


def _check_split_integrity(
    records: list[dict[str, Any]],
    split_ids: dict[str, set[str]],
    expected_counts: dict[str, int],
) -> DatasetIntegrityCheck:
    union = set().union(*split_ids.values())
    pairwise_disjoint = all(
        split_ids[first].isdisjoint(split_ids[second])
        for first_index, first in enumerate(SPLIT_NAMES)
        for second in SPLIT_NAMES[first_index + 1 :]
    )
    paths = [str(record["path"]) for record in records]
    actual_counts = {split: len(split_ids[split]) for split in SPLIT_NAMES}
    passed = (
        pairwise_disjoint
        and len(union) == len(records)
        and len(paths) == len(set(paths))
        and actual_counts == expected_counts
    )
    return DatasetIntegrityCheck(
        name="whole_trajectory_splits_are_disjoint",
        passed=passed,
        metrics={
            "pairwise_disjoint": pairwise_disjoint,
            "unique_ids": len(union),
            "unique_paths": len(set(paths)),
            "split_counts": actual_counts,
        },
        criterion="IDs and files are unique, disjoint, and match the locked split counts",
    )


def _check_excitation_coverage(
    counts: dict[str, dict[str, int]],
) -> DatasetIntegrityCheck:
    missing = {
        split: [family for family, count in families.items() if count < 1]
        for split, families in counts.items()
    }
    passed = all(not families for families in missing.values())
    return DatasetIntegrityCheck(
        name="all_excitation_families_are_covered",
        passed=passed,
        metrics={"counts": counts, "missing": missing},
        criterion="all eight excitation families occur in every split",
    )


def _check_trajectory_completeness(
    records: list[dict[str, Any]],
    trajectories: dict[str, Any],
    load_errors: list[str],
    expected_transitions: int,
) -> DatasetIntegrityCheck:
    actual_transitions = sum(
        trajectory.transitions for trajectory in trajectories.values()
    )
    declared_transitions = sum(int(record["transitions"]) for record in records)
    terminated = [
        record["trajectory_id"] for record in records if record.get("terminated")
    ]
    lengths_match = all(
        trajectory_id in trajectories
        and trajectories[trajectory_id].transitions == int(record["transitions"])
        for trajectory_id, record in (
            (str(record["trajectory_id"]), record) for record in records
        )
    )
    passed = (
        not load_errors
        and not terminated
        and lengths_match
        and actual_transitions == declared_transitions == expected_transitions
    )
    return DatasetIntegrityCheck(
        name="trajectories_are_complete_and_finite",
        passed=passed,
        metrics={
            "load_errors": load_errors,
            "terminated_trajectories": terminated,
            "lengths_match": lengths_match,
            "actual_transitions": actual_transitions,
            "expected_transitions": expected_transitions,
        },
        criterion="all hashed NPZ files are finite, complete, and have the locked length",
    )


def _check_signal_limits(
    ranges: dict[str, list[float]],
    phase2: Phase2Spec,
) -> DatasetIntegrityCheck:
    mapping = {
        "armature_current_a": phase2.limits["armature_current_a"],
        "angular_speed_rad_s": phase2.limits["angular_speed_rad_s"],
        "winding_temperature_c": phase2.limits["winding_temperature_c"],
        "armature_voltage_v": phase2.limits["armature_voltage_v"],
    }
    violations = {
        name: values
        for name, values in ranges.items()
        if name in mapping
        and (
            values[0] < mapping[name].minimum
            or values[1] > mapping[name].maximum
        )
    }
    passed = len(ranges) == 4 and not violations
    return DatasetIntegrityCheck(
        name="signals_stay_inside_declared_limits",
        passed=passed,
        metrics={"ranges": ranges, "violations": violations},
        criterion="current, speed, temperature, and applied voltage stay in Phase-2 limits",
    )


def _check_temperature_retention(
    trajectories: dict[str, Any],
) -> DatasetIntegrityCheck:
    shapes_valid = all(
        trajectory.temperature.shape == (trajectory.transitions + 1, 1)
        for trajectory in trajectories.values()
    )
    temperature_span = _span(
        [
            trajectory.temperature
            for trajectory in trajectories.values()
        ]
    )
    passed = bool(trajectories) and shapes_valid and temperature_span > 0.0
    return DatasetIntegrityCheck(
        name="raw_temperature_is_retained",
        passed=passed,
        metrics={
            "temperature_shapes_valid": shapes_valid,
            "temperature_span_c": temperature_span,
        },
        criterion="every raw trajectory retains a non-degenerate evaluation temperature",
    )


def _check_model_views(
    trajectories: dict[str, Any],
    spec: Phase4Spec,
) -> DatasetIntegrityCheck:
    valid = True
    for trajectory in trajectories.values():
        view = trajectory.model_view()
        valid = valid and view.observations.shape[-1] == 2
        valid = valid and view.controls.shape[-1] == 1
        valid = valid and np.array_equal(
            view.observations[:, 0, :],
            trajectory.states[:, :2],
        )
        valid = valid and np.array_equal(
            view.controls[:, 0, :],
            trajectory.applied_voltages,
        )
    model_names = list(spec.features["model_observation"]) + list(
        spec.features["model_control"]
    )
    forbidden = [
        name
        for name in model_names
        if "temperature" in name or name == "load_torque_n_m"
    ]
    passed = bool(trajectories) and valid and not forbidden
    return DatasetIntegrityCheck(
        name="model_view_excludes_temperature",
        passed=passed,
        metrics={
            "model_features": model_names,
            "forbidden_features": forbidden,
            "array_projection_valid": valid,
        },
        criterion="R2DN view is exactly [current, speed, applied voltage]",
    )


def _check_normalization(
    dataset: Phase4Dataset,
    trajectories: dict[str, Any],
    spec: Phase4Spec,
) -> DatasetIntegrityCheck:
    observation = _StreamingMoments(2)
    control = _StreamingMoments(1)
    for trajectory_id in dataset.trajectory_ids("train"):
        trajectory = trajectories.get(trajectory_id)
        if trajectory is None:
            continue
        observation.update(trajectory.states[:, :2])
        control.update(trajectory.applied_voltages)
    floor = float(spec.normalization["standard_deviation_floor"])
    stored = dataset.normalization
    comparisons = {
        "observation_mean": np.allclose(
            stored.observation_mean, observation.mean, rtol=1e-12, atol=1e-12
        ),
        "observation_std": np.allclose(
            stored.observation_std,
            observation.standard_deviation(floor),
            rtol=1e-12,
            atol=1e-12,
        ),
        "control_mean": np.allclose(
            stored.control_mean, control.mean, rtol=1e-12, atol=1e-12
        ),
        "control_std": np.allclose(
            stored.control_std,
            control.standard_deviation(floor),
            rtol=1e-12,
            atol=1e-12,
        ),
        "observation_count": stored.observation_count == observation.count,
        "control_count": stored.control_count == control.count,
        "fit_split": stored.fit_split == "train",
        "temperature_used": dataset.manifest["normalization"]["temperature_used"]
        is False,
    }
    return DatasetIntegrityCheck(
        name="normalization_uses_train_only",
        passed=all(comparisons.values()),
        metrics=comparisons,
        criterion="stored moments reproduce a fresh train-only temperature-free fit",
    )


def _check_reproducible_identity(
    records: list[dict[str, Any]],
    expected_plans: dict[str, Any],
) -> DatasetIntegrityCheck:
    seeds = [int(record["seed"]) for record in records]
    content_hashes_valid = all(
        len(str(record.get("content_sha256", ""))) == 64
        and all(
            character in "0123456789abcdef"
            for character in str(record["content_sha256"])
        )
        for record in records
    )
    plan_mismatches: list[str] = []
    for record in records:
        trajectory_id = str(record["trajectory_id"])
        plan = expected_plans.get(trajectory_id)
        if plan is None:
            plan_mismatches.append(trajectory_id)
            continue
        exact = (
            record["split"] == plan.split
            and record["excitation_family"] == plan.excitation_family
            and int(record["seed"]) == plan.seed
            and math.isclose(float(record["duration_s"]), plan.duration_s)
            and record.get("ood_axis") == plan.ood_axis
            and bool(record.get("novel_profile")) == plan.novel_profile
            and math.isclose(
                float(record["initial_state"]["armature_current_a"]),
                plan.initial_current_a,
            )
            and math.isclose(
                float(record["initial_state"]["angular_speed_rad_s"]),
                plan.initial_speed_rad_s,
            )
            and math.isclose(
                float(record["initial_state"]["winding_temperature_c"]),
                plan.initial_temperature_c,
            )
            and math.isclose(
                float(record["load_torque_n_m"]),
                plan.load_torque_n_m,
            )
            and all(
                math.isclose(
                    float(record["parameter_scales"][name]),
                    value,
                )
                for name, value in plan.parameter_scales.items()
            )
        )
        if not exact:
            plan_mismatches.append(trajectory_id)
    passed = (
        len(records) == len(expected_plans)
        and len(seeds) == len(set(seeds))
        and content_hashes_valid
        and not plan_mismatches
    )
    return DatasetIntegrityCheck(
        name="seeds_and_content_are_reproducible",
        passed=passed,
        metrics={
            "unique_seeds": len(set(seeds)),
            "expected_seeds": len(records),
            "content_hashes_valid": content_hashes_valid,
            "plan_mismatches": plan_mismatches,
        },
        criterion="manifest choices equal the seed-derived plans and content is hash-addressed",
    )


def _check_id_ood_separation(
    records: list[dict[str, Any]],
    spec: Phase4Spec,
) -> DatasetIntegrityCheck:
    id_records = [record for record in records if record["split"] == "id_test"]
    ood_records = [record for record in records if record["split"] == "ood_test"]
    training_maximum = max(
        float(band[1]) for band in spec.domain["temperature_bands_c"]
    )
    nominal_load = float(spec.domain["nominal_load_torque_n_m"])
    id_inside = all(
        record["ood_axis"] is None
        and float(record["initial_state"]["winding_temperature_c"])
        <= training_maximum
        and math.isclose(float(record["load_torque_n_m"]), nominal_load)
        and all(
            math.isclose(float(scale), 1.0)
            for scale in record["parameter_scales"].values()
        )
        and not bool(record["novel_profile"])
        for record in id_records
    )
    axes = {str(record["ood_axis"]) for record in ood_records}
    ood_valid = all(_record_matches_ood_axis(record, spec) for record in ood_records)
    passed = bool(id_records) and bool(ood_records) and id_inside and ood_valid and axes == set(
        OOD_AXES
    )
    return DatasetIntegrityCheck(
        name="id_and_ood_domains_are_separated",
        passed=passed,
        metrics={
            "id_inside_training_domain": id_inside,
            "ood_axes": sorted(axes),
            "ood_records_valid": ood_valid,
        },
        criterion="ID uses new in-domain seeds; OOD covers four declared out-of-domain axes",
    )


def _record_matches_ood_axis(
    record: dict[str, Any],
    spec: Phase4Spec,
) -> bool:
    axis = record["ood_axis"]
    if axis == "higher_initial_temperature":
        return float(record["initial_state"]["winding_temperature_c"]) >= float(
            spec.ood["higher_initial_temperature_c"][0]
        )
    if axis == "stronger_load":
        return float(record["load_torque_n_m"]) >= float(
            spec.ood["stronger_load_torque_n_m"][0]
        )
    if axis == "novel_profile":
        return bool(record["novel_profile"])
    if axis == "physical_parameter_shift":
        return all(
            not math.isclose(float(value), 1.0)
            for value in record["parameter_scales"].values()
        )
    return False


def _global_signal_ranges(
    trajectories: dict[str, Any],
) -> dict[str, list[float]]:
    if not trajectories:
        return {}
    ranges: dict[str, list[float]] = {
        "armature_current_a": [math.inf, -math.inf],
        "angular_speed_rad_s": [math.inf, -math.inf],
        "winding_temperature_c": [math.inf, -math.inf],
        "armature_voltage_v": [math.inf, -math.inf],
    }
    for trajectory in trajectories.values():
        arrays = {
            "armature_current_a": trajectory.states[:, 0],
            "angular_speed_rad_s": trajectory.states[:, 1],
            "winding_temperature_c": trajectory.states[:, 2],
            "armature_voltage_v": trajectory.applied_voltages[:, 0],
        }
        for name, values in arrays.items():
            ranges[name][0] = min(ranges[name][0], float(np.min(values)))
            ranges[name][1] = max(ranges[name][1], float(np.max(values)))
    return ranges


def _span(values: list[Any]) -> float:
    if not values:
        return 0.0
    minimum = min(float(np.min(value)) for value in values)
    maximum = max(float(np.max(value)) for value in values)
    return maximum - minimum


def _plot_coverage(report: Phase4ValidationReport, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    split_counts = [report.split_counts[split] for split in SPLIT_NAMES]
    axes[0].bar(SPLIT_NAMES, split_counts, color="#3B82F6")
    axes[0].set_title("Whole-trajectory split counts")
    axes[0].set_ylabel("Trajectories")
    axes[0].tick_params(axis="x", rotation=20)

    names = [
        "current [A]",
        "speed [rad/s]",
        "temperature [°C]",
        "voltage [V]",
    ]
    keys = list(report.signal_ranges)
    lows = [report.signal_ranges[key][0] for key in keys]
    highs = [report.signal_ranges[key][1] for key in keys]
    centers = [(low + high) / 2.0 for low, high in zip(lows, highs, strict=True)]
    half_widths = [
        (high - low) / 2.0 for low, high in zip(lows, highs, strict=True)
    ]
    axes[1].errorbar(
        centers,
        np.arange(len(names)),
        xerr=half_widths,
        fmt="o",
        capsize=5,
        color="#DC2626",
    )
    axes[1].set_yticks(np.arange(len(names)), names)
    axes[1].set_title("Observed signal ranges")
    axes[1].grid(axis="x", alpha=0.25)
    figure.suptitle(
        f"Phase 4 dataset coverage — {report.profile} — "
        f"{report.transition_count:,} transitions"
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)
