"""Validation and artifact generation for a trained Phase-6 R2DN checkpoint."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from r2dn_dc_motor.data import Phase4Dataset
from r2dn_dc_motor.models.r2dn_training import LoadedPhase6Checkpoint
from r2dn_dc_motor.phase6_spec import REQUIRED_PHASE6_CHECKS, Phase6Spec, load_phase6_spec


@dataclass(frozen=True)
class Phase6Check:
    """One auditable Phase-6 checkpoint condition."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class Phase6ValidationReport:
    """Training-only evidence; the clean model-comparison gate remains Phase 7."""

    passed: bool
    profile: str
    dataset_fingerprint: str
    selected_latent_size: int
    selected_seed: int
    validation_free_rollout_nrmse: float
    contractivity_margin: float
    checks: tuple[Phase6Check, ...]
    pilot_runs: tuple[dict[str, Any], ...]
    final_runs: tuple[dict[str, Any], ...]
    phase7_gate_claimed: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checks"] = [asdict(check) for check in self.checks]
        return payload

    def summary(self) -> str:
        lines = [
            "PHASE 6 R2DN TRAINING: PASS" if self.passed else "PHASE 6 R2DN TRAINING: FAIL",
            f"profile: {self.profile}",
            f"dataset: {self.dataset_fingerprint}",
            f"selected latent size: {self.selected_latent_size}",
            f"selected seed: {self.selected_seed}",
            (
                "validation free-rollout NRMSE: "
                f"{self.validation_free_rollout_nrmse:.6g}"
            ),
            f"contractivity margin: {self.contractivity_margin:.6g}",
            "Phase-7 comparison Gate claimed: no",
        ]
        lines.extend(
            f"[{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}"
            for check in self.checks
        )
        return "\n".join(lines)


def run_phase6_validation(
    dataset: Phase4Dataset,
    checkpoint: LoadedPhase6Checkpoint,
    *,
    spec: Phase6Spec | None = None,
) -> Phase6ValidationReport:
    """Verify data isolation, selection, hashes, finite training, and contraction."""

    spec = spec or load_phase6_spec()
    manifest = checkpoint.manifest
    manifest.validate(dataset=dataset, spec=spec)
    history = checkpoint.training_history
    profile = spec.profile(manifest.training_profile)
    pilot_runs = tuple(history.get("pilot_runs", ()))
    final_runs = tuple(history.get("final_runs", ()))
    selected_history = tuple(history.get("selected_run_history", ()))

    allowed = (
        *manifest.observation_features,
        *manifest.control_features,
    )
    model_view_passed = (
        allowed
        == (
            "armature_current_a",
            "angular_speed_rad_s",
            "armature_voltage_v",
        )
        and manifest.forbidden_training_features
        == tuple(spec.interface["forbidden_training_features"])
        and not set(allowed) & set(manifest.forbidden_training_features)
    )

    train_ids = set(manifest.train_trajectory_ids)
    validation_ids = set(manifest.validation_trajectory_ids)
    split_isolation_passed = (
        not train_ids & validation_ids
        and train_ids == set(dataset.trajectory_ids("train"))
        and validation_ids == set(dataset.trajectory_ids("validation"))
        and bool(spec.selection["id_test_used_for_selection"]) is False
        and bool(spec.selection["ood_test_used_for_selection"]) is False
    )

    normalization_passed = _normalization_equal(
        checkpoint.normalization,
        dataset.normalization,
    )
    checkpoint_binding_passed = (
        manifest.dataset_fingerprint == dataset.fingerprint
        and len(manifest.parameter_sha256) == 64
        and len(manifest.normalization_sha256) == 64
        and len(manifest.training_history_sha256) == 64
    )
    metric_names = (
        "total_loss",
        "one_step_loss",
        "rollout_loss",
        "reconstruction_loss",
        "gradient_norm",
    )
    recorded_values = [
        float(record[name])
        for record in selected_history
        for name in metric_names
    ]
    validation_values = [
        float(run["validation"]["free_rollout_nrmse"])
        for run in final_runs
    ]
    prediction_maxima = [
        float(run["validation"]["maximum_absolute_normalized_prediction"])
        for run in final_runs
    ]
    prediction_limit = float(
        spec.validation["maximum_absolute_normalized_prediction"]
    )
    finite_passed = (
        bool(recorded_values)
        and bool(validation_values)
        and all(
            math.isfinite(value)
            for value in (*recorded_values, *validation_values, *prediction_maxima)
        )
        and all(bool(run["validation"]["finite"]) for run in final_runs)
        and all(value <= prediction_limit for value in prediction_maxima)
    )
    contractivity_passed = (
        math.isfinite(manifest.contractivity_margin)
        and manifest.contractivity_margin > 0.0
    )
    minimum_score = min(validation_values, default=math.inf)
    selection_passed = (
        spec.selection["metric"] == "validation_free_rollout_nrmse"
        and spec.selection["selection_split"] == "validation"
        and math.isclose(
            manifest.validation_free_rollout_nrmse,
            minimum_score,
            rel_tol=1e-6,
            abs_tol=1e-8,
        )
        and int(history.get("selected_seed", -1)) == manifest.seed
    )
    recorded_seeds = tuple(int(run["seed"]) for run in final_runs)
    multi_seed_passed = (
        recorded_seeds == profile.training_seeds
        and len(set(recorded_seeds)) == len(recorded_seeds)
        and (
            profile.name == "ci"
            or len(recorded_seeds) >= 3
        )
    )

    checks = (
        Phase6Check(
            "training_uses_only_temperature_free_model_view",
            model_view_passed,
            f"allowed={list(allowed)}; forbidden={list(manifest.forbidden_training_features)}",
        ),
        Phase6Check(
            "training_and_selection_splits_are_isolated",
            split_isolation_passed,
            f"train={len(train_ids)}, validation={len(validation_ids)}, ID/OOD selection=false",
        ),
        Phase6Check(
            "normalization_matches_phase4_train_statistics",
            normalization_passed,
            f"fit_split={checkpoint.normalization.fit_split}",
        ),
        Phase6Check(
            "checkpoint_is_dataset_and_upstream_bound",
            checkpoint_binding_passed,
            f"dataset={manifest.dataset_fingerprint}; upstream={manifest.upstream_commit}",
        ),
        Phase6Check(
            "training_losses_and_gradients_are_finite",
            finite_passed,
            (
                f"history_records={len(selected_history)}, final_runs={len(final_runs)}, "
                f"max|normalized prediction|={max(prediction_maxima, default=math.inf):.6g}"
                f" <= {prediction_limit:g}"
            ),
        ),
        Phase6Check(
            "contractivity_certificate_is_positive",
            contractivity_passed,
            f"margin={manifest.contractivity_margin:.6g}",
        ),
        Phase6Check(
            "selection_uses_validation_long_rollout",
            selection_passed,
            (
                f"horizon={manifest.selection_horizon_steps}, "
                f"selected_NRMSE={manifest.validation_free_rollout_nrmse:.6g}"
            ),
        ),
        Phase6Check(
            "multiple_final_training_seeds_are_recorded",
            multi_seed_passed,
            f"profile={profile.name}, seeds={list(recorded_seeds)}",
        ),
    )
    if {check.name for check in checks} != REQUIRED_PHASE6_CHECKS:
        raise RuntimeError("Phase-6 runtime check catalog drifted from the specification")
    return Phase6ValidationReport(
        passed=all(check.passed for check in checks),
        profile=manifest.training_profile,
        dataset_fingerprint=dataset.fingerprint,
        selected_latent_size=manifest.latent_size,
        selected_seed=manifest.seed,
        validation_free_rollout_nrmse=manifest.validation_free_rollout_nrmse,
        contractivity_margin=manifest.contractivity_margin,
        checks=checks,
        pilot_runs=pilot_runs,
        final_runs=final_runs,
    )


def generate_phase6_artifacts(
    report: Phase6ValidationReport,
    history: dict[str, Any],
    output_directory: Path | str,
) -> tuple[Path, Path]:
    """Write a JSON audit report and compact training/selection diagnostic figure."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            'Phase-6 artifacts require: python -m pip install -e ".[phase6]"'
        ) from error

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "phase6_r2dn_training.json"
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    selected_history = history["selected_run_history"]
    updates = np.asarray([record["global_update"] for record in selected_history])
    total_loss = np.asarray([record["total_loss"] for record in selected_history])
    rollout_loss = np.asarray([record["rollout_loss"] for record in selected_history])
    final_runs = history["final_runs"]
    seeds = [str(run["seed"]) for run in final_runs]
    scores = [run["validation"]["free_rollout_nrmse"] for run in final_runs]

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    axes[0].plot(updates, total_loss, marker="o", markersize=2.5, label="total")
    axes[0].plot(updates, rollout_loss, marker=".", markersize=2, label="rollout")
    axes[0].set_yscale("symlog", linthresh=1e-8)
    axes[0].set_xlabel("Optimizer update")
    axes[0].set_ylabel("Normalized loss")
    axes[0].set_title("Selected-run curriculum")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    colors = [
        "#2b8cbe" if int(seed) == report.selected_seed else "#a6bddb"
        for seed in seeds
    ]
    axes[1].bar(seeds, scores, color=colors)
    axes[1].set_xlabel("Final training seed")
    axes[1].set_ylabel("Validation free-rollout NRMSE")
    axes[1].set_title("Checkpoint selection")
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle(
        f"Phase 6 R2DN — latent {report.selected_latent_size}, "
        f"selected seed {report.selected_seed}"
    )
    figure.tight_layout()
    figure_path = output / "phase6_r2dn_training.png"
    figure.savefig(figure_path, dpi=170)
    plt.close(figure)
    return report_path, figure_path


def _normalization_equal(left: Any, right: Any) -> bool:
    return (
        left.fit_split == right.fit_split == "train"
        and left.observation_count == right.observation_count
        and left.control_count == right.control_count
        and np.array_equal(left.observation_mean, right.observation_mean)
        and np.array_equal(left.observation_std, right.observation_std)
        and np.array_equal(left.control_mean, right.control_mean)
        and np.array_equal(left.control_std, right.control_std)
    )
