"""Phase-6F protocol audit and optimizer-ablation artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from r2dn_dc_motor.models.r2dn_phase6f import (
    Phase6FStudy,
    _file_sha256,
    select_optimizer_variant,
)
from r2dn_dc_motor.phase6f_spec import REQUIRED_PHASE6F_CHECKS, Phase6FSpec


@dataclass(frozen=True)
class Phase6FCheck:
    """One auditable optimizer-ablation condition."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class Phase6FValidationReport:
    """Serializable Phase-6F result and protocol audit."""

    schema_version: int
    phase: str
    passed: bool
    profile: str
    dataset_fingerprint: str
    selected_variant: str
    selected_combined_nrmse: float
    selected_current_nrmse: float
    selected_speed_nrmse: float
    baseline_combined_nrmse: float
    relative_improvement_over_baseline: float
    target_combined_nrmse: float
    target_met: bool
    contractivity_margin: float
    checks: tuple[Phase6FCheck, ...]
    variants: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "checks": [asdict(value) for value in self.checks],
            "variants": list(self.variants),
        }

    def summary(self) -> str:
        return "\n".join(
            (
                "PHASE 6F: PASS" if self.passed else "PHASE 6F: FAIL",
                f"profile: {self.profile}",
                f"selected optimizer variant: {self.selected_variant}",
                (
                    "selected multisine NRMSE current / speed / combined: "
                    f"{self.selected_current_nrmse:.6g} / "
                    f"{self.selected_speed_nrmse:.6g} / "
                    f"{self.selected_combined_nrmse:.6g}"
                ),
                f"baseline combined NRMSE: {self.baseline_combined_nrmse:.6g}",
                (
                    "relative improvement over baseline: "
                    f"{100 * self.relative_improvement_over_baseline:.3f}%"
                ),
                (
                    f"target combined NRMSE <= {self.target_combined_nrmse:g}: "
                    f"{'YES' if self.target_met else 'NO'}"
                ),
                f"selected contractivity margin: {self.contractivity_margin:.6g}",
                *(
                    f"[{'PASS' if value.passed else 'FAIL'}] "
                    f"{value.name}: {value.detail}"
                    for value in self.checks
                ),
            )
        )


def validate_phase6f_study(
    study: Phase6FStudy,
    phase6b_report_path: Path | str,
    phase6e_checkpoint_directory: Path | str,
    *,
    phase6f: Phase6FSpec,
) -> Phase6FValidationReport:
    """Audit baseline provenance, finiteness, contraction, isolation, and selection."""

    profile = phase6f.profile(study.profile_name)
    expected_names = tuple(value.name for value in profile.variants)
    actual_names = tuple(value.variant.name for value in study.variants)
    baseline = study.baseline
    baseline_valid = (
        baseline.variant.source == "phase6e_checkpoint"
        and baseline.run.role == "baseline_phase6e"
        and baseline.run.architecture.state_size == profile.latent_size
        and baseline.run.seed == profile.seed
        and _file_sha256(Path(phase6e_checkpoint_directory) / "manifest.json")
        == study.phase6e_manifest_sha256
    )
    complete = actual_names == expected_names and len(study.variants) == len(expected_names)
    finite = all(
        value.run.validation.finite
        and math.isfinite(value.run.selection_score)
        and math.isfinite(value.selection.median_combined_nrmse)
        and math.isfinite(value.selection.median_current_nrmse)
        and math.isfinite(value.selection.median_speed_nrmse)
        and all(metric.finite for metric in value.selection.multisine_metrics)
        for value in study.variants
    )
    contraction = all(
        math.isfinite(value.run.contractivity_margin)
        and value.run.contractivity_margin > 0.0
        for value in study.variants
    )
    isolated = (
        phase6f.selection["canonical_phase6c_scenario_used_for_selection"] is False
        and phase6f.selection["phase6e_selection_scenarios_reused"] is False
        and tuple(
            metric.scenario_name
            for metric in study.variants[0].selection.multisine_metrics
        )
        == tuple(value.name for value in profile.multisine_scenarios)
    )
    expected_selected = select_optimizer_variant(
        {
            value.variant.name: value.selection.median_combined_nrmse
            for value in study.variants
        },
        expected_names,
        relative_tolerance=profile.tie_relative_tolerance,
    )
    selection_matches = study.selected_variant_name == expected_selected
    report_hash_matches = _file_sha256(Path(phase6b_report_path)) == (
        study.phase6b_report_sha256
    )
    checks = (
        Phase6FCheck(
            "baseline_checkpoint_is_phase6e_winner",
            baseline_valid,
            (
                f"latent={baseline.run.architecture.state_size}, "
                f"seed={baseline.run.seed}, manifest_hash={baseline_valid}"
            ),
        ),
        Phase6FCheck(
            "all_declared_optimizer_variants_are_evaluated",
            complete,
            f"completed={len(study.variants)}/{len(expected_names)}",
        ),
        Phase6FCheck(
            "all_training_and_selection_rollouts_are_finite",
            finite,
            f"finite variants={sum(value.run.validation.finite for value in study.variants)}"
            f"/{len(study.variants)}",
        ),
        Phase6FCheck(
            "all_contractivity_margins_are_positive",
            contraction,
            (
                "minimum margin="
                f"{min(value.run.contractivity_margin for value in study.variants):.6g}"
            ),
        ),
        Phase6FCheck(
            "selection_scenarios_are_new_and_canonical_is_held_out",
            isolated and report_hash_matches,
            (
                f"scenarios={[value.name for value in profile.multisine_scenarios]}, "
                f"report_hash={report_hash_matches}"
            ),
        ),
        Phase6FCheck(
            "selected_checkpoint_matches_locked_tie_rule",
            selection_matches,
            (
                f"selected={study.selected_variant_name}, "
                f"tie={100 * profile.tie_relative_tolerance:g}%"
            ),
        ),
    )
    if {value.name for value in checks} != REQUIRED_PHASE6F_CHECKS:
        raise RuntimeError("Phase-6F runtime check catalog drifted from specification")
    selected = study.selected
    return Phase6FValidationReport(
        schema_version=1,
        phase="6F",
        passed=all(value.passed for value in checks),
        profile=study.profile_name,
        dataset_fingerprint=study.dataset_fingerprint,
        selected_variant=study.selected_variant_name,
        selected_combined_nrmse=selected.selection.median_combined_nrmse,
        selected_current_nrmse=selected.selection.median_current_nrmse,
        selected_speed_nrmse=selected.selection.median_speed_nrmse,
        baseline_combined_nrmse=baseline.selection.median_combined_nrmse,
        relative_improvement_over_baseline=study.relative_improvement_over_baseline,
        target_combined_nrmse=study.target_combined_nrmse,
        target_met=study.target_met,
        contractivity_margin=selected.run.contractivity_margin,
        checks=checks,
        variants=tuple(value.summary() for value in study.variants),
    )


def generate_phase6f_artifacts(
    report: Phase6FValidationReport,
    output_directory: Path | str,
) -> tuple[Path, Path]:
    """Write the Phase-6F JSON report and optimizer comparison plot."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("Phase-6F artifacts require matplotlib") from error

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "phase6f_optimizer_floor_ablation.json"
    figure_path = output / "phase6f_optimizer_floor_ablation.png"
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    names = [value["name"] for value in report.variants]
    combined = np.asarray(
        [value["selection_multisine_median_combined_nrmse"] for value in report.variants]
    )
    current = np.asarray(
        [value["selection_multisine_median_current_nrmse"] for value in report.variants]
    )
    speed = np.asarray(
        [value["selection_multisine_median_speed_nrmse"] for value in report.variants]
    )
    x = np.arange(len(names), dtype=np.float64)
    width = 0.25
    figure, axis = plt.subplots(figsize=(10.5, 6.2))
    axis.bar(x - width, current, width, label="current")
    axis.bar(x, speed, width, label="speed")
    axis.bar(x + width, combined, width, label="combined")
    axis.axhline(
        report.target_combined_nrmse,
        color="#d7301f",
        linestyle="--",
        linewidth=1.2,
        label=f"target {report.target_combined_nrmse:g}",
    )
    axis.set_xticks(x, names)
    axis.set_ylabel("100 s held-out multisine NRMSE")
    axis.set_title("Phase 6F — latent-16 optimizer-floor ablation")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return report_path, figure_path
