"""Phase-6D screening, protocol audit, and accuracy-ablation artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from r2dn_dc_motor.data import Phase4Dataset, R2DNWindowSampler
from r2dn_dc_motor.models.r2dn_phase6d import (
    Phase6DStudy,
    aligned_validation_window,
    select_variant_from_medians,
)
from r2dn_dc_motor.models.r2dn_training import evaluate_validation_rollout
from r2dn_dc_motor.phase6d_spec import (
    REQUIRED_PHASE6D_CHECKS,
    Phase6DSpec,
)


@dataclass(frozen=True)
class ScreenMetric:
    """One existing checkpoint evaluated with one aligned history length."""

    checkpoint_name: str
    checkpoint_phase: str
    latent_size: int
    seed: int
    burn_in_steps: int
    horizon_steps: int
    windows: int
    combined_nrmse: float
    current_nrmse: float
    speed_nrmse: float
    one_step_nrmse: float
    maximum_absolute_normalized_prediction: float
    contractivity_margin: float
    finite: bool


@dataclass(frozen=True)
class AccuracyScreenReport:
    """Cheap pre-training checkpoint and burn-in comparison."""

    schema_version: int
    phase: str
    passed: bool
    dataset_fingerprint: str
    validation_window_seed: int
    horizon_steps: int
    windows: int
    metrics: tuple[ScreenMetric, ...]
    selected_checkpoint_name: str
    selected_burn_in_steps: int
    selected_combined_nrmse: float

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "metrics": [asdict(value) for value in self.metrics],
        }

    def summary(self) -> str:
        lines = [
            "PHASE 6D SCREEN: PASS" if self.passed else "PHASE 6D SCREEN: FAIL",
            f"horizon: {self.horizon_steps} steps x {self.windows} windows",
            (
                f"best existing model: {self.selected_checkpoint_name}, "
                f"burn-in={self.selected_burn_in_steps}, "
                f"NRMSE={self.selected_combined_nrmse:.6g}"
            ),
        ]
        lines.extend(
            (
                f"{value.checkpoint_name} burn-in={value.burn_in_steps}: "
                f"combined/current/speed={value.combined_nrmse:.6g}/"
                f"{value.current_nrmse:.6g}/{value.speed_nrmse:.6g}"
            )
            for value in self.metrics
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class Phase6DCheck:
    """One auditable Phase-6D protocol condition."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class Phase6DValidationReport:
    """Serializable Phase-6D training-ablation result."""

    schema_version: int
    phase: str
    passed: bool
    profile: str
    dataset_fingerprint: str
    selected_variant: str
    selected_latent_size: int
    selected_burn_in_steps: int
    selected_seed: int
    selected_validation_nrmse: float
    selected_variant_median_nrmse: float
    baseline_median_nrmse: float
    relative_improvement_over_baseline: float
    target_combined_nrmse: float
    target_met: bool
    contractivity_margin: float
    checks: tuple[Phase6DCheck, ...]
    aggregates: tuple[dict[str, Any], ...]
    runs: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "checks": [asdict(value) for value in self.checks],
            "aggregates": list(self.aggregates),
            "runs": list(self.runs),
        }

    def summary(self) -> str:
        return "\n".join(
            (
                "PHASE 6D: PASS" if self.passed else "PHASE 6D: FAIL",
                f"profile: {self.profile}",
                (
                    f"selected: {self.selected_variant}, "
                    f"latent={self.selected_latent_size}, "
                    f"burn-in={self.selected_burn_in_steps}, seed={self.selected_seed}"
                ),
                (
                    "validation NRMSE selected / variant median / baseline median: "
                    f"{self.selected_validation_nrmse:.6g} / "
                    f"{self.selected_variant_median_nrmse:.6g} / "
                    f"{self.baseline_median_nrmse:.6g}"
                ),
                (
                    "relative median improvement over control: "
                    f"{100 * self.relative_improvement_over_baseline:.2f}%"
                ),
                (
                    f"target NRMSE <= {self.target_combined_nrmse:g}: "
                    f"{'YES' if self.target_met else 'NO'}"
                ),
                f"contractivity margin: {self.contractivity_margin:.6g}",
                *(
                    f"[{'PASS' if value.passed else 'FAIL'}] "
                    f"{value.name}: {value.detail}"
                    for value in self.checks
                ),
            )
        )


def screen_existing_checkpoints(
    dataset: Phase4Dataset,
    checkpoints: tuple[tuple[str, str, Any], ...] | list[tuple[str, str, Any]],
    *,
    phase6d: Phase6DSpec,
) -> AccuracyScreenReport:
    """Evaluate existing checkpoints on identical 10-second validation targets."""

    burn_ins = tuple(int(value) for value in phase6d.screen["burn_in_steps"])
    maximum_burn_in = max(burn_ins)
    horizon = int(phase6d.screen["validation_horizon_steps"])
    windows = int(phase6d.screen["validation_windows"])
    seed = int(phase6d.screen["validation_window_seed"])
    sampler = R2DNWindowSampler(dataset, split="validation", seed=0)
    maximum_window = sampler.fixed_validation_windows(
        count=windows,
        burn_in_steps=maximum_burn_in,
        rollout_steps=horizon,
        seed=seed,
    )
    metrics: list[ScreenMetric] = []
    for name, checkpoint_phase, checkpoint in checkpoints:
        adapter = checkpoint.adapter
        for burn_in in burn_ins:
            window = aligned_validation_window(
                maximum_window,
                maximum_burn_in_steps=maximum_burn_in,
                burn_in_steps=burn_in,
                rollout_steps=horizon,
            )
            result = evaluate_validation_rollout(
                adapter,
                checkpoint.parameters,
                window,
                burn_in_steps=burn_in,
            )
            metrics.append(
                ScreenMetric(
                    checkpoint_name=name,
                    checkpoint_phase=checkpoint_phase,
                    latent_size=checkpoint.manifest.latent_size,
                    seed=checkpoint.manifest.seed,
                    burn_in_steps=burn_in,
                    horizon_steps=horizon,
                    windows=windows,
                    combined_nrmse=result.free_rollout_nrmse,
                    current_nrmse=result.current_free_rollout_nrmse,
                    speed_nrmse=result.speed_free_rollout_nrmse,
                    one_step_nrmse=result.one_step_nrmse,
                    maximum_absolute_normalized_prediction=(
                        result.maximum_absolute_normalized_prediction
                    ),
                    contractivity_margin=checkpoint.manifest.contractivity_margin,
                    finite=result.finite,
                )
            )
    if not metrics:
        raise ValueError("at least one existing checkpoint is required for screening")
    selected = min(metrics, key=lambda value: value.combined_nrmse)
    passed = all(
        value.finite
        and math.isfinite(value.combined_nrmse)
        and value.contractivity_margin > 0.0
        for value in metrics
    )
    return AccuracyScreenReport(
        schema_version=1,
        phase="6D-screen",
        passed=passed,
        dataset_fingerprint=dataset.fingerprint,
        validation_window_seed=seed,
        horizon_steps=horizon,
        windows=windows,
        metrics=tuple(metrics),
        selected_checkpoint_name=selected.checkpoint_name,
        selected_burn_in_steps=selected.burn_in_steps,
        selected_combined_nrmse=selected.combined_nrmse,
    )


def validate_phase6d_study(
    study: Phase6DStudy,
    *,
    phase6d: Phase6DSpec,
) -> Phase6DValidationReport:
    """Audit completeness, finiteness, contraction, and locked selection."""

    profile = phase6d.profile(study.profile_name)
    expected = {
        (variant.name, seed)
        for variant in profile.variants
        for seed in profile.seeds
    }
    actual = {
        (run.role.removeprefix("phase6d_"), run.seed) for run in study.runs
    }
    complete = actual == expected and len(study.runs) == len(expected)
    finite = all(
        run.validation.finite
        and math.isfinite(run.selection_score)
        and run.validation.horizon_steps == profile.selection_horizon_steps
        and run.validation.windows == profile.validation_windows
        for run in study.runs
    )
    contraction = all(
        math.isfinite(run.contractivity_margin) and run.contractivity_margin > 0.0
        for run in study.runs
    )
    selection_isolated = (
        phase6d.selection["selection_split"] == "validation"
        and phase6d.selection["id_test_used_for_selection"] is False
        and phase6d.selection["ood_test_used_for_selection"] is False
        and phase6d.selection["stress_used_for_selection"] is False
    )
    expected_selected = select_variant_from_medians(
        list(study.aggregates),
        relative_tolerance=profile.tie_relative_tolerance,
    )
    selected_aggregate = next(
        value for value in study.aggregates if value.name == study.selected_variant.name
    )
    selected_runs = [
        value
        for value in study.runs
        if value.role == f"phase6d_{study.selected_variant.name}"
    ]
    selection_matches = (
        study.selected_variant.name == expected_selected
        and any(study.selected_run is value for value in selected_runs)
        and math.isclose(
            study.selected_run.selection_score,
            min(value.selection_score for value in selected_runs),
            rel_tol=1e-7,
            abs_tol=1e-9,
        )
    )
    checks = (
        Phase6DCheck(
            "all_declared_variants_and_seeds_are_evaluated",
            complete,
            f"completed={len(actual)}/{len(expected)}",
        ),
        Phase6DCheck(
            "all_runs_are_finite",
            finite,
            f"finite={sum(value.validation.finite for value in study.runs)}/{len(study.runs)}",
        ),
        Phase6DCheck(
            "all_contractivity_margins_are_positive",
            contraction,
            f"minimum margin={min(value.contractivity_margin for value in study.runs):.6g}",
        ),
        Phase6DCheck(
            "selection_uses_only_fixed_validation_windows",
            selection_isolated,
            (
                f"seed={profile.validation_window_seed}, "
                f"horizon={profile.selection_horizon_steps}, "
                f"windows={profile.validation_windows}"
            ),
        ),
        Phase6DCheck(
            "selected_checkpoint_matches_locked_median_rule",
            selection_matches,
            (
                f"variant={study.selected_variant.name}, seed={study.selected_run.seed}, "
                f"tie={100 * profile.tie_relative_tolerance:g}%"
            ),
        ),
    )
    if {value.name for value in checks} != REQUIRED_PHASE6D_CHECKS:
        raise RuntimeError("Phase-6D runtime check catalog drifted from specification")
    baseline = study.aggregates[0].median_validation_free_rollout_nrmse
    selected_median = selected_aggregate.median_validation_free_rollout_nrmse
    relative_improvement = (
        (baseline - selected_median) / baseline if baseline > 0.0 else 0.0
    )
    return Phase6DValidationReport(
        schema_version=1,
        phase="6D",
        passed=all(value.passed for value in checks),
        profile=study.profile_name,
        dataset_fingerprint=study.dataset_fingerprint,
        selected_variant=study.selected_variant.name,
        selected_latent_size=study.selected_variant.latent_size,
        selected_burn_in_steps=study.selected_variant.burn_in_steps,
        selected_seed=study.selected_run.seed,
        selected_validation_nrmse=study.selected_run.selection_score,
        selected_variant_median_nrmse=selected_median,
        baseline_median_nrmse=baseline,
        relative_improvement_over_baseline=relative_improvement,
        target_combined_nrmse=study.target_combined_nrmse,
        target_met=study.target_met,
        contractivity_margin=study.selected_run.contractivity_margin,
        checks=checks,
        aggregates=tuple(value.to_dict() for value in study.aggregates),
        runs=tuple(value.summary() for value in study.runs),
    )


def generate_screen_artifacts(
    report: AccuracyScreenReport,
    output_directory: Path | str,
) -> tuple[Path, Path]:
    """Write screening JSON and a checkpoint/burn-in comparison plot."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "phase6d_existing_checkpoint_screen.json"
    figure_path = output / "phase6d_existing_checkpoint_screen.png"
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot_screen(report, figure_path)
    return report_path, figure_path


def generate_phase6d_artifacts(
    report: Phase6DValidationReport,
    output_directory: Path | str,
) -> tuple[Path, Path]:
    """Write final ablation JSON and median/seed comparison plot."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "phase6d_accuracy_ablation.json"
    figure_path = output / "phase6d_accuracy_ablation.png"
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot_ablation(report, figure_path)
    return report_path, figure_path


def _plot_screen(report: AccuracyScreenReport, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("Phase-6D artifacts require matplotlib") from error
    names = tuple(dict.fromkeys(value.checkpoint_name for value in report.metrics))
    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    for name in names:
        values = [value for value in report.metrics if value.checkpoint_name == name]
        axis.plot(
            [value.burn_in_steps for value in values],
            [value.combined_nrmse for value in values],
            marker="o",
            linewidth=1.8,
            label=name,
        )
    axis.set_xlabel("Burn-in [control steps]")
    axis.set_ylabel("10 s validation NRMSE")
    axis.set_title("Phase 6D screening — identical prediction targets")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_ablation(report: Phase6DValidationReport, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("Phase-6D artifacts require matplotlib") from error
    names = [str(value["name"]) for value in report.aggregates]
    medians = [
        float(value["median_validation_free_rollout_nrmse"])
        for value in report.aggregates
    ]
    figure, axis = plt.subplots(figsize=(11.5, 6.2))
    bars = axis.bar(names, medians, color=("#9ecae1", "#6baed6", "#3182bd", "#08519c"))
    for index, aggregate in enumerate(report.aggregates):
        scores = np.asarray(aggregate["scores"], dtype=np.float64)
        axis.scatter(
            np.full(scores.shape, index),
            scores,
            color="#252525",
            s=28,
            zorder=3,
            label="seed score" if index == 0 else None,
        )
    axis.axhline(
        report.target_combined_nrmse,
        color="#de2d26",
        linestyle="--",
        linewidth=1.3,
        label=f"target {report.target_combined_nrmse:g}",
    )
    for bar, value in zip(bars, medians, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f" {value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axis.set_ylabel("10 s validation NRMSE")
    axis.set_title(
        f"Phase 6D accuracy ablation — selected {report.selected_variant}"
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
