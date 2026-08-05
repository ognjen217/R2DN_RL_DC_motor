"""Phase-6E protocol audit and larger-latent result artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from r2dn_dc_motor.models.r2dn_phase6b import select_latent_from_medians
from r2dn_dc_motor.models.r2dn_phase6e import Phase6EStudy, _file_sha256
from r2dn_dc_motor.phase6e_spec import REQUIRED_PHASE6E_CHECKS, Phase6ESpec


@dataclass(frozen=True)
class Phase6ECheck:
    """One auditable larger-latent experiment condition."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class Phase6EValidationReport:
    """Serializable Phase-6E result and protocol audit."""

    schema_version: int
    phase: str
    passed: bool
    profile: str
    dataset_fingerprint: str
    phase6b_report_sha256: str
    selected_latent_size: int
    selected_seed: int
    selected_run_combined_nrmse: float
    selected_run_current_nrmse: float
    selected_run_speed_nrmse: float
    selected_latent_median_combined_nrmse: float
    selected_latent_median_current_nrmse: float
    selected_latent_median_speed_nrmse: float
    target_combined_nrmse: float
    target_met: bool
    contractivity_margin: float
    checks: tuple[Phase6ECheck, ...]
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
                "PHASE 6E: PASS" if self.passed else "PHASE 6E: FAIL",
                f"profile: {self.profile}",
                f"selected latent / seed: {self.selected_latent_size} / {self.selected_seed}",
                (
                    "selected run multisine NRMSE current / speed / combined: "
                    f"{self.selected_run_current_nrmse:.6g} / "
                    f"{self.selected_run_speed_nrmse:.6g} / "
                    f"{self.selected_run_combined_nrmse:.6g}"
                ),
                (
                    "selected latent three-seed median current / speed / combined: "
                    f"{self.selected_latent_median_current_nrmse:.6g} / "
                    f"{self.selected_latent_median_speed_nrmse:.6g} / "
                    f"{self.selected_latent_median_combined_nrmse:.6g}"
                ),
                (
                    f"target median combined NRMSE <= {self.target_combined_nrmse:g}: "
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


def validate_phase6e_study(
    study: Phase6EStudy,
    phase6b_report_path: Path | str,
    *,
    phase6e: Phase6ESpec,
) -> Phase6EValidationReport:
    """Audit completeness, finiteness, contraction, isolation, and selection."""

    profile = phase6e.profile(study.profile_name)
    expected = {
        (latent, seed) for latent in profile.latent_sizes for seed in profile.seeds
    }
    actual = {
        (value.run.architecture.state_size, value.run.seed) for value in study.runs
    }
    complete = actual == expected and len(study.runs) == len(expected)
    expected_scenarios = tuple(value.name for value in profile.multisine_scenarios)
    expected_anchors = profile.selection_anchor_indices
    selection_rollouts_match = all(
        tuple(value.scenario_name for value in run.multisine_metrics)
        == expected_scenarios
        and tuple(value.anchor_index for value in run.multisine_metrics)
        == expected_anchors
        for run in study.runs
    )
    finite = all(
        run.run.validation.finite
        and math.isfinite(run.run.selection_score)
        and math.isfinite(run.median_combined_nrmse)
        and math.isfinite(run.median_current_nrmse)
        and math.isfinite(run.median_speed_nrmse)
        and all(value.finite for value in run.multisine_metrics)
        for run in study.runs
    )
    contraction = all(
        math.isfinite(value.run.contractivity_margin)
        and value.run.contractivity_margin > 0.0
        for value in study.runs
    )
    selection_isolated = (
        phase6e.selection["selection_split"] == "synthetic_validation_multisine"
        and phase6e.selection["selection_reference"] == "full_rk4"
        and phase6e.selection["id_test_used_for_selection"] is False
        and phase6e.selection["ood_test_used_for_selection"] is False
        and phase6e.selection["stress_used_for_selection"] is False
        and selection_rollouts_match
    )
    canonical_held_out = (
        phase6e.selection["canonical_phase6c_scenario_used_for_selection"] is False
        and all(value != "multisine" for value in expected_scenarios)
    )
    expected_selected = select_latent_from_medians(
        {
            value.latent_size: value.median_combined_nrmse
            for value in study.aggregates
        },
        relative_tolerance=profile.tie_relative_tolerance,
    )
    selected_candidates = [
        value
        for value in study.runs
        if value.run.architecture.state_size == study.selected_latent_size
    ]
    selection_matches = (
        study.selected_latent_size == expected_selected
        and any(study.selected_run is value for value in selected_candidates)
        and math.isclose(
            study.selected_run.median_combined_nrmse,
            min(value.median_combined_nrmse for value in selected_candidates),
            rel_tol=1e-7,
            abs_tol=1e-9,
        )
    )
    report_hash_matches = _file_sha256(Path(phase6b_report_path)) == (
        study.phase6b_report_sha256
    )
    checks = (
        Phase6ECheck(
            "all_declared_latents_and_seeds_are_evaluated",
            complete,
            f"completed={len(actual)}/{len(expected)}",
        ),
        Phase6ECheck(
            "all_training_and_selection_rollouts_are_finite",
            finite,
            (
                f"finite training runs={sum(value.run.validation.finite for value in study.runs)}"
                f"/{len(study.runs)}"
            ),
        ),
        Phase6ECheck(
            "all_contractivity_margins_are_positive",
            contraction,
            (
                "minimum margin="
                f"{min(value.run.contractivity_margin for value in study.runs):.6g}"
            ),
        ),
        Phase6ECheck(
            "selection_uses_only_locked_multisine_scenarios",
            selection_isolated and report_hash_matches,
            (
                f"scenarios={list(expected_scenarios)}, "
                f"anchors={list(expected_anchors)}, report_hash={report_hash_matches}"
            ),
        ),
        Phase6ECheck(
            "canonical_phase6c_scenario_is_held_out",
            canonical_held_out,
            "canonical 1000 s Phase-6C multisine is not a selection scenario",
        ),
        Phase6ECheck(
            "selected_checkpoint_matches_locked_median_rule",
            selection_matches,
            (
                f"latent={study.selected_latent_size}, seed={study.selected_run.run.seed}, "
                f"tie={100 * profile.tie_relative_tolerance:g}%"
            ),
        ),
    )
    if {value.name for value in checks} != REQUIRED_PHASE6E_CHECKS:
        raise RuntimeError("Phase-6E runtime check catalog drifted from specification")
    aggregate = study.selected_aggregate
    selected = study.selected_run
    return Phase6EValidationReport(
        schema_version=1,
        phase="6E",
        passed=all(value.passed for value in checks),
        profile=study.profile_name,
        dataset_fingerprint=study.dataset_fingerprint,
        phase6b_report_sha256=study.phase6b_report_sha256,
        selected_latent_size=study.selected_latent_size,
        selected_seed=selected.run.seed,
        selected_run_combined_nrmse=selected.median_combined_nrmse,
        selected_run_current_nrmse=selected.median_current_nrmse,
        selected_run_speed_nrmse=selected.median_speed_nrmse,
        selected_latent_median_combined_nrmse=aggregate.median_combined_nrmse,
        selected_latent_median_current_nrmse=aggregate.median_current_nrmse,
        selected_latent_median_speed_nrmse=aggregate.median_speed_nrmse,
        target_combined_nrmse=study.target_combined_nrmse,
        target_met=study.target_met,
        contractivity_margin=selected.run.contractivity_margin,
        checks=checks,
        aggregates=tuple(value.to_dict() for value in study.aggregates),
        runs=tuple(value.summary() for value in study.runs),
    )


def generate_phase6e_artifacts(
    report: Phase6EValidationReport,
    output_directory: Path | str,
) -> tuple[Path, Path]:
    """Write the Phase-6E JSON report and latent/seed NRMSE plot."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("Phase-6E artifacts require matplotlib") from error

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "phase6e_larger_latent_search.json"
    figure_path = output / "phase6e_larger_latent_search.png"
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    aggregates = sorted(report.aggregates, key=lambda value: value["latent_size"])
    latents = np.asarray([value["latent_size"] for value in aggregates])
    combined = np.asarray([value["median_combined_nrmse"] for value in aggregates])
    current = np.asarray([value["median_current_nrmse"] for value in aggregates])
    speed = np.asarray([value["median_speed_nrmse"] for value in aggregates])
    figure, axis = plt.subplots(figsize=(10.5, 6.2))
    axis.plot(latents, combined, marker="o", linewidth=2.2, label="combined median")
    axis.plot(latents, current, marker="s", linewidth=1.6, label="current median")
    axis.plot(latents, speed, marker="^", linewidth=1.6, label="speed median")
    for aggregate in aggregates:
        x = np.full(len(aggregate["combined_scores"]), aggregate["latent_size"])
        axis.scatter(
            x,
            aggregate["combined_scores"],
            color="#636363",
            alpha=0.55,
            s=28,
            zorder=4,
        )
    axis.axhline(
        report.target_combined_nrmse,
        color="#d7301f",
        linestyle="--",
        linewidth=1.2,
        label=f"target {report.target_combined_nrmse:g}",
    )
    axis.set_xticks(latents)
    axis.set_xlabel("R2DN latent dimension")
    axis.set_ylabel("100 s held-out multisine NRMSE")
    axis.set_title("Phase 6E — full-curriculum larger-latent comparison")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return report_path, figure_path
