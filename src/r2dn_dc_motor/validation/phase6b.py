"""Phase-6B audit, held-out error growth, and long autoregressive stress tests."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from r2dn_dc_motor.data import Phase4Dataset
from r2dn_dc_motor.models.r2dn_phase6b import (
    LoadedPhase6BCheckpoint,
    adaptive_candidate_required,
    select_latent_from_medians,
)
from r2dn_dc_motor.phase2_spec import Phase2Spec, load_phase2_spec
from r2dn_dc_motor.phase6_spec import Phase6Spec, load_phase6_spec
from r2dn_dc_motor.phase6b_spec import (
    REQUIRED_PHASE6B_CHECKS,
    Phase6BSpec,
    load_phase6b_spec,
)


@dataclass(frozen=True)
class Phase6BCheck:
    """One auditable Phase-6B condition."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ReplayErrorMetric:
    """Autoregressive error on fixed post-selection held-out windows."""

    split: str
    horizon_steps: int
    windows: int
    free_rollout_nrmse: float
    current_nrmse: float
    speed_nrmse: float
    maximum_absolute_normalized_prediction: float
    finite: bool
    used_for_selection: bool = False


@dataclass(frozen=True)
class StressRolloutMetric:
    """Boundedness, energy, and perturbation evidence at one milestone."""

    split: str
    scenario: str
    horizon_steps: int
    duration_s: float
    anchors: int
    finite: bool
    bounded: bool
    first_nonfinite_step: int | None
    first_current_limit_violation_step: int | None
    first_speed_limit_violation_step: int | None
    maximum_absolute_normalized_prediction: float
    maximum_absolute_current_a: float
    maximum_absolute_speed_rad_s: float
    maximum_latent_norm: float
    initial_mean_energy_j: float
    final_mean_energy_j: float
    maximum_energy_j: float
    cumulative_input_work_j: float
    tail_output_rms_growth_ratio: float
    tail_energy_mean_growth_ratio: float
    tail_latent_rms_growth_ratio: float
    tail_latent_second_half_rms: float
    maximum_perturbation_separation_normalized: float
    final_perturbation_separation_normalized: float
    tail_perturbation_rms_growth_ratio: float
    tail_perturbation_second_half_rms: float
    unforced_energy_ok: bool
    latent_ok: bool
    perturbation_ok: bool


@dataclass(frozen=True)
class StressAnchor:
    """Provenance for one measured burn-in anchor."""

    split: str
    trajectory_id: str
    start_step: int
    burn_in_steps: int


@dataclass(frozen=True)
class Phase6BValidationReport:
    """Search audit plus numerical evidence about the outer autoregressive loop."""

    passed: bool
    profile: str
    dataset_fingerprint: str
    selected_latent_size: int
    selected_seed: int
    selected_run_source: str
    pilot_median_validation_nrmse: float
    final_validation_free_rollout_nrmse: float
    contractivity_margin: float
    checks: tuple[Phase6BCheck, ...]
    pilot_aggregates: tuple[dict[str, Any], ...]
    replay_metrics: tuple[ReplayErrorMetric, ...]
    stress_metrics: tuple[StressRolloutMetric, ...]
    stress_anchors: tuple[StressAnchor, ...]
    phase7_gate_claimed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "checks": [asdict(value) for value in self.checks],
            "replay_metrics": [asdict(value) for value in self.replay_metrics],
            "stress_metrics": [asdict(value) for value in self.stress_metrics],
            "stress_anchors": [asdict(value) for value in self.stress_anchors],
        }

    def summary(self) -> str:
        longest = max((value.horizon_steps for value in self.stress_metrics), default=0)
        longest_metrics = [
            value for value in self.stress_metrics if value.horizon_steps == longest
        ]
        worst_prediction = max(
            (
                value.maximum_absolute_normalized_prediction
                for value in longest_metrics
            ),
            default=math.inf,
        )
        lines = [
            "PHASE 6B: PASS" if self.passed else "PHASE 6B: FAIL",
            f"profile: {self.profile}",
            f"dataset: {self.dataset_fingerprint}",
            f"selected latent / seed: {self.selected_latent_size} / {self.selected_seed}",
            f"selected source: {self.selected_run_source}",
            (
                "pilot median / final validation NRMSE: "
                f"{self.pilot_median_validation_nrmse:.6g} / "
                f"{self.final_validation_free_rollout_nrmse:.6g}"
            ),
            f"contractivity margin: {self.contractivity_margin:.6g}",
            (
                f"longest stress horizon: {longest} steps; "
                f"worst max|normalized prediction|={worst_prediction:.6g}"
            ),
            "Phase-7 comparison Gate claimed: no",
        ]
        lines.extend(
            f"[{'PASS' if value.passed else 'FAIL'}] {value.name}: {value.detail}"
            for value in self.checks
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class _AnchorBatch:
    observations: np.ndarray
    controls: np.ndarray
    initial_observations: np.ndarray
    provenance: tuple[StressAnchor, ...]


class _StressAccumulator:
    """Streaming metrics for one requested stress milestone."""

    def __init__(
        self,
        *,
        split: str,
        scenario: str,
        horizon_steps: int,
        anchors: int,
        initial_physical_observations: np.ndarray,
        phase2: Phase2Spec,
        phase6b: Phase6BSpec,
    ) -> None:
        self.split = split
        self.scenario = scenario
        self.horizon_steps = horizon_steps
        self.anchors = anchors
        self.phase2 = phase2
        self.phase6b = phase6b
        self.maximum_normalized = 0.0
        self.maximum_current = 0.0
        self.maximum_speed = 0.0
        self.maximum_latent = 0.0
        self.maximum_energy = 0.0
        self.maximum_perturbation = 0.0
        self.input_work = 0.0
        self.first_nonfinite: int | None = None
        self.first_current_violation: int | None = None
        self.first_speed_violation: int | None = None
        self.final_energy = math.nan
        self.final_perturbation = math.nan
        self.tail_output: list[np.ndarray] = []
        self.tail_energy: list[np.ndarray] = []
        self.tail_latent: list[np.ndarray] = []
        self.tail_perturbation: list[np.ndarray] = []
        parameters = phase2.motor_parameters
        self.initial_energy = float(
            np.mean(
                0.5
                * parameters.armature_inductance_h
                * initial_physical_observations[:, 0] ** 2
                + 0.5
                * parameters.inertia_kg_m2
                * initial_physical_observations[:, 1] ** 2
            )
        )

    def update(
        self,
        *,
        start_step: int,
        base_normalized: np.ndarray,
        perturbed_normalized: np.ndarray,
        latent_norms: np.ndarray,
        physical_voltage: np.ndarray,
        normalization: Any,
    ) -> None:
        if start_step >= self.horizon_steps:
            return
        count = min(base_normalized.shape[0], self.horizon_steps - start_step)
        base = np.asarray(base_normalized[:count], dtype=np.float64)
        perturbed = np.asarray(perturbed_normalized[:count], dtype=np.float64)
        latent = np.asarray(latent_norms[:count], dtype=np.float64)
        voltage = np.asarray(physical_voltage[:count], dtype=np.float64)
        finite_rows = (
            np.isfinite(base).all(axis=(1, 2))
            & np.isfinite(perturbed).all(axis=(1, 2))
            & np.isfinite(latent).all(axis=1)
        )
        if not finite_rows.all() and self.first_nonfinite is None:
            self.first_nonfinite = start_step + int(np.flatnonzero(~finite_rows)[0]) + 1
        finite_count = int(np.argmax(~finite_rows)) if not finite_rows.all() else count
        if finite_count < 1:
            return
        base = base[:finite_count]
        perturbed = perturbed[:finite_count]
        latent = latent[:finite_count]
        voltage = voltage[:finite_count]
        all_predictions = np.concatenate((base, perturbed), axis=1)
        self.maximum_normalized = max(
            self.maximum_normalized,
            float(np.max(np.abs(all_predictions))),
        )
        all_physical = (
            all_predictions * normalization.observation_std[None, None, :]
            + normalization.observation_mean[None, None, :]
        )
        physical = all_physical[:, : self.anchors]
        self.maximum_current = max(
            self.maximum_current,
            float(np.max(np.abs(all_physical[:, :, 0]))),
        )
        self.maximum_speed = max(
            self.maximum_speed,
            float(np.max(np.abs(all_physical[:, :, 1]))),
        )
        self.maximum_latent = max(
            self.maximum_latent,
            float(np.max(latent)),
        )
        current_limit = self.phase2.limits["armature_current_a"]
        speed_limit = self.phase2.limits["angular_speed_rad_s"]
        current_violation = (all_physical[:, :, 0] < current_limit.minimum) | (
            all_physical[:, :, 0] > current_limit.maximum
        )
        speed_violation = (all_physical[:, :, 1] < speed_limit.minimum) | (
            all_physical[:, :, 1] > speed_limit.maximum
        )
        if current_violation.any() and self.first_current_violation is None:
            first = int(np.argwhere(current_violation)[0, 0])
            self.first_current_violation = start_step + first + 1
        if speed_violation.any() and self.first_speed_violation is None:
            first = int(np.argwhere(speed_violation)[0, 0])
            self.first_speed_violation = start_step + first + 1
        parameters = self.phase2.motor_parameters
        energy = (
            0.5 * parameters.armature_inductance_h * physical[:, :, 0] ** 2
            + 0.5 * parameters.inertia_kg_m2 * physical[:, :, 1] ** 2
        )
        self.maximum_energy = max(self.maximum_energy, float(np.max(energy)))
        self.final_energy = float(np.mean(energy[-1]))
        dt = self.phase2.integration_settings.control_period_s
        self.input_work += float(
            np.sum(voltage[:, None] * physical[:, :, 0]) * dt / self.anchors
        )
        separation = np.linalg.norm(perturbed - base, axis=-1)
        self.maximum_perturbation = max(
            self.maximum_perturbation,
            float(np.max(separation)),
        )
        self.final_perturbation = float(np.sqrt(np.mean(separation[-1] ** 2)))
        tail_start = int(
            math.floor(
                self.horizon_steps
                * (1.0 - float(self.phase6b.validation["tail_fraction"]))
            )
        )
        overlap_start = max(start_step, tail_start)
        overlap_stop = min(start_step + finite_count, self.horizon_steps)
        if overlap_start < overlap_stop:
            local_start = overlap_start - start_step
            local_stop = overlap_stop - start_step
            self.tail_output.append(
                np.sqrt(np.mean(base[local_start:local_stop] ** 2, axis=-1))
            )
            self.tail_energy.append(energy[local_start:local_stop])
            self.tail_latent.append(latent[local_start:local_stop])
            self.tail_perturbation.append(separation[local_start:local_stop])

    def mark_nonfinite(self, step: int) -> None:
        if self.first_nonfinite is None and step <= self.horizon_steps:
            self.first_nonfinite = step

    def finish(self) -> StressRolloutMetric:
        output_ratio, _ = _tail_rms_growth(self.tail_output)
        energy_ratio, energy_second = _tail_mean_growth(self.tail_energy)
        latent_ratio, latent_second = _tail_rms_growth(self.tail_latent)
        perturbation_ratio, perturbation_second = _tail_rms_growth(
            self.tail_perturbation
        )
        finite = self.first_nonfinite is None
        prediction_limit = float(
            self.phase6b.validation["maximum_absolute_normalized_prediction"]
        )
        latent_ok = (
            self.maximum_latent <= float(self.phase6b.validation["maximum_latent_norm"])
            and (
                latent_second <= 1e-6
                or latent_ratio
                <= float(
                    self.phase6b.validation["latent_tail_growth_ratio_limit"]
                )
            )
        )
        bounded = (
            finite
            and self.first_current_violation is None
            and self.first_speed_violation is None
            and self.maximum_normalized <= prediction_limit
            and latent_ok
        )
        unforced_energy_ok = True
        if self.scenario == "zero_voltage":
            energy_limit = float(
                self.phase6b.validation[
                    "unforced_energy_tail_growth_ratio_limit"
                ]
            )
            unforced_energy_ok = energy_second <= 1e-6 or energy_ratio <= energy_limit
        perturbation_limit = float(
            self.phase6b.validation["perturbation_tail_growth_ratio_limit"]
        )
        perturbation_absolute = float(
            self.phase6b.validation[
                "perturbation_absolute_rms_limit_normalized"
            ]
        )
        perturbation_ok = (
            perturbation_second <= perturbation_absolute
            or perturbation_ratio <= perturbation_limit
        )
        return StressRolloutMetric(
            split=self.split,
            scenario=self.scenario,
            horizon_steps=self.horizon_steps,
            duration_s=(
                self.horizon_steps
                * self.phase2.integration_settings.control_period_s
            ),
            anchors=self.anchors,
            finite=finite,
            bounded=bounded,
            first_nonfinite_step=self.first_nonfinite,
            first_current_limit_violation_step=self.first_current_violation,
            first_speed_limit_violation_step=self.first_speed_violation,
            maximum_absolute_normalized_prediction=self.maximum_normalized,
            maximum_absolute_current_a=self.maximum_current,
            maximum_absolute_speed_rad_s=self.maximum_speed,
            maximum_latent_norm=self.maximum_latent,
            initial_mean_energy_j=self.initial_energy,
            final_mean_energy_j=self.final_energy,
            maximum_energy_j=self.maximum_energy,
            cumulative_input_work_j=self.input_work,
            tail_output_rms_growth_ratio=output_ratio,
            tail_energy_mean_growth_ratio=energy_ratio,
            tail_latent_rms_growth_ratio=latent_ratio,
            tail_latent_second_half_rms=latent_second,
            maximum_perturbation_separation_normalized=self.maximum_perturbation,
            final_perturbation_separation_normalized=self.final_perturbation,
            tail_perturbation_rms_growth_ratio=perturbation_ratio,
            tail_perturbation_second_half_rms=perturbation_second,
            unforced_energy_ok=unforced_energy_ok,
            latent_ok=latent_ok,
            perturbation_ok=perturbation_ok,
        )


def run_phase6b_validation(
    dataset: Phase4Dataset,
    checkpoint: LoadedPhase6BCheckpoint,
    *,
    phase6b: Phase6BSpec | None = None,
    phase6: Phase6Spec | None = None,
    phase2: Phase2Spec | None = None,
    progress: Any | None = None,
) -> Phase6BValidationReport:
    """Audit selection, then run held-out replay and long stress tests."""

    phase6b = phase6b or load_phase6b_spec()
    phase6 = phase6 or load_phase6_spec()
    phase2 = phase2 or load_phase2_spec()
    progress = progress or (lambda _: None)
    manifest = checkpoint.manifest
    manifest.validate(dataset=dataset, phase6=phase6, phase6b=phase6b)
    profile = phase6b.profile(manifest.training_profile)
    history = checkpoint.study_history
    pilot_runs = tuple(history["pilot_runs"])
    aggregates = tuple(history["pilot_aggregates"])
    median_map = {
        int(value["latent_size"]): float(
            value["median_validation_free_rollout_nrmse"]
        )
        for value in aggregates
    }
    recomputed_medians = {
        latent: float(
            np.median(
                [
                    float(run["validation"]["free_rollout_nrmse"])
                    for run in pilot_runs
                    if int(run["latent_size"]) == latent
                ]
            )
        )
        for latent in median_map
    }
    aggregates_match_runs = all(
        math.isclose(
            median_map[latent],
            recomputed_medians[latent],
            rel_tol=1e-7,
            abs_tol=1e-9,
        )
        for latent in median_map
    )
    expected_latents = list(profile.candidate_latent_sizes)
    if profile.adaptive_enabled and manifest.adaptive_candidate_triggered:
        expected_latents.append(profile.adaptive_candidate_latent)
    count_by_latent = {
        latent: sum(int(run["latent_size"]) == latent for run in pilot_runs)
        for latent in expected_latents
    }
    latent_catalog_passed = (
        tuple(expected_latents) == manifest.evaluated_latent_sizes
        and set(median_map) == set(expected_latents)
        and all(count == len(profile.pilot_seeds) for count in count_by_latent.values())
    )
    pilot_seed_passed = all(
        tuple(int(run["seed"]) for run in pilot_runs if int(run["latent_size"]) == latent)
        == profile.pilot_seeds
        for latent in expected_latents
    ) and all(
        int(run["validation"]["horizon_steps"])
        == profile.pilot_validation_horizon_steps
        for run in pilot_runs
    )
    if profile.adaptive_enabled:
        expected_trigger = adaptive_candidate_required(
            median_map,
            reference_latent=profile.adaptive_reference_latent,
            boundary_latent=profile.adaptive_boundary_latent,
            improvement_threshold=profile.adaptive_improvement_threshold,
        )
    else:
        expected_trigger = False
    expected_selected = select_latent_from_medians(
        median_map,
        relative_tolerance=profile.tie_relative_tolerance,
    )
    architecture_selection_passed = (
        expected_trigger == manifest.adaptive_candidate_triggered
        and expected_selected == manifest.latent_size
        and aggregates_match_runs
        and math.isclose(
            median_map[expected_selected],
            manifest.pilot_median_validation_nrmse,
            rel_tol=1e-7,
            abs_tol=1e-9,
        )
    )
    final_runs = tuple(history["final_runs"])
    final_scores = tuple(
        float(value["validation"]["free_rollout_nrmse"]) for value in final_runs
    )
    final_selection_passed = (
        tuple(int(value["seed"]) for value in final_runs)
        == profile.final_training_seeds
        and int(history["selected_seed"]) == manifest.seed
        and math.isclose(
            manifest.validation_free_rollout_nrmse,
            min(final_scores, default=math.inf),
            rel_tol=1e-6,
            abs_tol=1e-8,
        )
        and manifest.selection_horizon_steps == profile.selection_horizon_steps
    )
    selection_isolation_passed = (
        phase6b.search["selection_split"] == "validation"
        and phase6b.search["id_test_used_for_selection"] is False
        and phase6b.search["ood_test_used_for_selection"] is False
        and phase6b.stress["synthetic_stress_used_for_selection"] is False
        and phase6b.stress["held_out_replay_used_for_selection"] is False
    )

    replay_metrics = run_held_out_replay(
        dataset,
        checkpoint,
        phase6b=phase6b,
        phase6=phase6,
        progress=progress,
    )
    stress_metrics, anchors = run_autoregressive_stress(
        dataset,
        checkpoint,
        phase6b=phase6b,
        phase6=phase6,
        phase2=phase2,
        progress=progress,
    )
    longest = max(profile.stress_horizon_steps)
    longest_metrics = [
        value for value in stress_metrics if value.horizon_steps == longest
    ]
    replay_passed = bool(replay_metrics) and all(
        value.finite
        and value.maximum_absolute_normalized_prediction
        <= float(phase6b.validation["maximum_absolute_normalized_prediction"])
        for value in replay_metrics
    )
    bounded_passed = bool(longest_metrics) and all(
        value.finite and value.bounded for value in longest_metrics
    )
    unforced_passed = all(
        value.unforced_energy_ok
        for value in longest_metrics
        if value.scenario == "zero_voltage"
    )
    perturbation_passed = all(value.perturbation_ok for value in longest_metrics)
    checks = (
        Phase6BCheck(
            "latent_catalog_is_fully_evaluated",
            latent_catalog_passed,
            (
                f"latents={expected_latents}; "
                f"runs_per_latent={count_by_latent}"
            ),
        ),
        Phase6BCheck(
            "pilot_uses_three_seeds_and_fixed_validation_windows",
            pilot_seed_passed,
            (
                f"seeds={list(profile.pilot_seeds)}; "
                f"horizon={profile.pilot_validation_horizon_steps}; "
                f"window_seed={profile.pilot_validation_seed}"
            ),
        ),
        Phase6BCheck(
            "architecture_selection_uses_locked_median_and_tie_rule",
            architecture_selection_passed,
            (
                f"selected={manifest.latent_size}; "
                f"median={manifest.pilot_median_validation_nrmse:.6g}; "
                f"tie={100 * profile.tie_relative_tolerance:g}%"
            ),
        ),
        Phase6BCheck(
            "final_checkpoint_uses_locked_validation_selection",
            final_selection_passed,
            (
                f"seeds={list(profile.final_training_seeds)}; "
                f"selected={manifest.seed}; "
                f"NRMSE={manifest.validation_free_rollout_nrmse:.6g}"
            ),
        ),
        Phase6BCheck(
            "selection_excludes_id_ood_and_stress_results",
            selection_isolation_passed,
            "selection=validation only; ID/OOD/replay/stress=post-selection",
        ),
        Phase6BCheck(
            "held_out_replay_is_finite",
            replay_passed,
            (
                f"windows={len(replay_metrics)}; "
                f"finite={sum(value.finite for value in replay_metrics)}; "
                "used_for_selection=false"
            ),
        ),
        Phase6BCheck(
            "long_stress_rollouts_are_finite_and_physically_bounded",
            bounded_passed,
            (
                f"horizon={longest}; "
                f"bounded={sum(value.bounded for value in longest_metrics)}"
                f"/{len(longest_metrics)}"
            ),
        ),
        Phase6BCheck(
            "zero_input_has_no_sustained_tail_energy_growth",
            unforced_passed,
            (
                "tail ratio limit="
                f"{phase6b.validation['unforced_energy_tail_growth_ratio_limit']}"
            ),
        ),
        Phase6BCheck(
            "autoregressive_perturbations_have_no_sustained_tail_growth",
            perturbation_passed,
            (
                "tail ratio / absolute RMS limits="
                f"{phase6b.validation['perturbation_tail_growth_ratio_limit']} / "
                f"{phase6b.validation['perturbation_absolute_rms_limit_normalized']}"
            ),
        ),
    )
    if {value.name for value in checks} != REQUIRED_PHASE6B_CHECKS:
        raise RuntimeError("Phase-6B runtime check catalog drifted from specification")
    return Phase6BValidationReport(
        passed=all(value.passed for value in checks),
        profile=profile.name,
        dataset_fingerprint=dataset.fingerprint,
        selected_latent_size=manifest.latent_size,
        selected_seed=manifest.seed,
        selected_run_source=manifest.selected_run_source,
        pilot_median_validation_nrmse=manifest.pilot_median_validation_nrmse,
        final_validation_free_rollout_nrmse=(
            manifest.validation_free_rollout_nrmse
        ),
        contractivity_margin=manifest.contractivity_margin,
        checks=checks,
        pilot_aggregates=aggregates,
        replay_metrics=replay_metrics,
        stress_metrics=stress_metrics,
        stress_anchors=anchors,
    )


def run_held_out_replay(
    dataset: Phase4Dataset,
    checkpoint: LoadedPhase6BCheckpoint,
    *,
    phase6b: Phase6BSpec,
    phase6: Phase6Spec,
    progress: Any,
) -> tuple[ReplayErrorMetric, ...]:
    """Measure error growth on fixed validation, ID, and OOD windows."""

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as error:
        raise ImportError("Phase-6B replay requires Phase-6 JAX dependencies") from error

    profile = phase6b.profile(checkpoint.manifest.training_profile)
    base = phase6.profile(profile.base_phase6_profile)
    maximum_horizon = max(profile.replay_horizon_steps)
    metrics: list[ReplayErrorMetric] = []
    adapter = checkpoint.adapter

    @jax.jit
    def evaluate(observations: Any, controls: Any) -> Any:
        batch_size = observations.shape[1]
        initial_state = jnp.zeros(
            (batch_size, adapter.architecture.state_size),
            dtype=jnp.float32,
        )
        burned_state, _ = adapter.burn_in(
            checkpoint.parameters,
            initial_state,
            observations[: base.burn_in_steps],
            controls[: base.burn_in_steps],
        )
        _, predictions = adapter.free_rollout(
            checkpoint.parameters,
            burned_state,
            observations[base.burn_in_steps],
            controls[base.burn_in_steps :],
        )
        return predictions

    for split_index, split in enumerate(phase6b.stress["splits"]):
        observations, controls = _fixed_replay_windows(
            dataset,
            split=str(split),
            count=profile.replay_windows_per_split,
            burn_in_steps=base.burn_in_steps,
            rollout_steps=maximum_horizon,
            seed=profile.replay_window_seed + split_index,
        )
        observations_jax = jnp.asarray(observations, dtype=jnp.float32)
        controls_jax = jnp.asarray(controls, dtype=jnp.float32)
        batch = observations.shape[1]
        predictions = np.asarray(evaluate(observations_jax, controls_jax))
        targets = observations[base.burn_in_steps + 1 :]
        for horizon in profile.replay_horizon_steps:
            prediction = predictions[:horizon]
            target = targets[:horizon]
            finite = bool(np.isfinite(prediction).all())
            if finite:
                error = prediction - target
                feature = np.sqrt(np.mean(error**2, axis=(0, 1)))
                score = float(np.sqrt(np.mean(error**2)))
                maximum = float(np.max(np.abs(prediction)))
            else:
                feature = np.asarray((math.inf, math.inf))
                score = math.inf
                maximum = math.inf
            metrics.append(
                ReplayErrorMetric(
                    split=str(split),
                    horizon_steps=horizon,
                    windows=batch,
                    free_rollout_nrmse=score,
                    current_nrmse=float(feature[0]),
                    speed_nrmse=float(feature[1]),
                    maximum_absolute_normalized_prediction=maximum,
                    finite=finite,
                )
            )
        progress(
            f"held-out replay {split}: "
            f"{maximum_horizon} steps x {batch} windows complete"
        )
    return tuple(metrics)


def run_autoregressive_stress(
    dataset: Phase4Dataset,
    checkpoint: LoadedPhase6BCheckpoint,
    *,
    phase6b: Phase6BSpec,
    phase6: Phase6Spec,
    phase2: Phase2Spec,
    progress: Any,
) -> tuple[tuple[StressRolloutMetric, ...], tuple[StressAnchor, ...]]:
    """Run chunked long rollouts without allocating a million-step trace."""

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as error:
        raise ImportError("Phase-6B stress tests require Phase-6 JAX dependencies") from error

    profile = phase6b.profile(checkpoint.manifest.training_profile)
    base_profile = phase6.profile(profile.base_phase6_profile)
    adapter = checkpoint.adapter
    explicit = adapter.model.direct_to_explicit(checkpoint.parameters)
    maximum_horizon = max(profile.stress_horizon_steps)
    chunk_steps = profile.stress_chunk_steps
    all_metrics: list[StressRolloutMetric] = []
    all_anchors: list[StressAnchor] = []
    for split_index, split in enumerate(phase6b.stress["splits"]):
        anchors = _fixed_stress_anchors(
            dataset,
            split=str(split),
            count=profile.stress_anchors_per_split,
            burn_in_steps=base_profile.burn_in_steps,
            seed=profile.stress_anchor_seed + split_index,
        )
        all_anchors.extend(anchors.provenance)
        observations = jnp.asarray(anchors.observations, dtype=jnp.float32)
        controls = jnp.asarray(anchors.controls, dtype=jnp.float32)
        batch = anchors.initial_observations.shape[0]
        initial_state = jnp.zeros(
            (batch, adapter.architecture.state_size),
            dtype=jnp.float32,
        )
        burned_state, _ = adapter.burn_in(
            checkpoint.parameters,
            initial_state,
            observations,
            controls,
        )
        base_observation = jnp.asarray(
            anchors.initial_observations,
            dtype=jnp.float32,
        )
        epsilon = profile.perturbation_epsilon_normalized
        latent_direction = jnp.ones_like(burned_state) / math.sqrt(
            adapter.architecture.state_size
        )
        observation_direction = jnp.asarray(
            (1.0, -1.0),
            dtype=jnp.float32,
        ) / math.sqrt(2.0)
        perturbed_state = burned_state + epsilon * latent_direction
        perturbed_observation = base_observation + epsilon * observation_direction
        paired_initial_state = jnp.concatenate((burned_state, perturbed_state), axis=0)
        paired_initial_observation = jnp.concatenate(
            (base_observation, perturbed_observation),
            axis=0,
        )
        initial_physical = (
            anchors.initial_observations
            * checkpoint.normalization.observation_std[None, :]
            + checkpoint.normalization.observation_mean[None, :]
        )

        @jax.jit
        def run_chunk(
            state: Any,
            observation: Any,
            chunk_controls: Any,
        ) -> tuple[Any, Any, Any, Any]:
            def step(carry: tuple[Any, Any], control: Any) -> tuple[Any, Any]:
                latent_state, predicted_observation = carry
                regressor = jnp.concatenate(
                    (predicted_observation, control),
                    axis=-1,
                )
                next_state, next_observation = adapter.model.explicit_call(
                    checkpoint.parameters,
                    latent_state,
                    regressor,
                    explicit,
                )
                latent_norm = jnp.linalg.norm(next_state, axis=-1)
                return (next_state, next_observation), (
                    next_observation,
                    latent_norm,
                )

            (final_state, final_observation), trace = jax.lax.scan(
                step,
                (state, observation),
                chunk_controls,
            )
            predictions, latent_norms = trace
            return final_state, final_observation, predictions, latent_norms

        for scenario in phase6b.scenarios:
            scenario_name = str(scenario["name"])
            accumulators = [
                _StressAccumulator(
                    split=str(split),
                    scenario=scenario_name,
                    horizon_steps=horizon,
                    anchors=batch,
                    initial_physical_observations=initial_physical,
                    phase2=phase2,
                    phase6b=phase6b,
                )
                for horizon in profile.stress_horizon_steps
            ]
            state = paired_initial_state
            observation = paired_initial_observation
            for start in range(0, maximum_horizon, chunk_steps):
                physical_voltage = scenario_voltage(
                    scenario,
                    start_step=start,
                    steps=chunk_steps,
                    control_period_s=phase2.integration_settings.control_period_s,
                )
                normalized_control = (
                    physical_voltage - checkpoint.normalization.control_mean[0]
                ) / checkpoint.normalization.control_std[0]
                chunk_control = np.repeat(
                    normalized_control[:, None, None],
                    repeats=2 * batch,
                    axis=1,
                ).astype(np.float32)
                state, observation, predictions, latent_norms = run_chunk(
                    state,
                    observation,
                    jnp.asarray(chunk_control),
                )
                predictions_np = np.asarray(predictions)
                latent_np = np.asarray(latent_norms)
                base_predictions = predictions_np[:, :batch]
                perturbed_predictions = predictions_np[:, batch:]
                for accumulator in accumulators:
                    accumulator.update(
                        start_step=start,
                        base_normalized=base_predictions,
                        perturbed_normalized=perturbed_predictions,
                        latent_norms=latent_np,
                        physical_voltage=physical_voltage,
                        normalization=checkpoint.normalization,
                    )
                finite = bool(
                    np.isfinite(predictions_np).all()
                    and np.isfinite(latent_np).all()
                )
                if not finite:
                    bad = (
                        ~np.isfinite(predictions_np).all(axis=(1, 2))
                        | ~np.isfinite(latent_np).all(axis=1)
                    )
                    first = start + int(np.flatnonzero(bad)[0]) + 1
                    for accumulator in accumulators:
                        accumulator.mark_nonfinite(first)
                    break
            all_metrics.extend(accumulator.finish() for accumulator in accumulators)
            longest_metric = all_metrics[-1]
            progress(
                f"stress {split}/{scenario_name}: "
                f"{maximum_horizon} steps; "
                f"bounded={'yes' if longest_metric.bounded else 'no'}"
            )
    return tuple(all_metrics), tuple(all_anchors)


def scenario_voltage(
    scenario: dict[str, Any],
    *,
    start_step: int,
    steps: int,
    control_period_s: float,
) -> np.ndarray:
    """Generate a deterministic physical-voltage chunk by absolute step index."""

    indices = start_step + np.arange(steps, dtype=np.int64)
    kind = str(scenario["kind"])
    if kind == "constant":
        return np.full(steps, float(scenario["amplitude_v"]), dtype=np.float64)
    if kind == "prbs":
        blocks = indices // int(scenario["hold_steps"])
        values = (
            blocks * np.int64(1_103_515_245)
            + np.int64(int(scenario["seed"]) * 12_345)
        )
        signs = np.where(((values >> 16) & 1) == 0, -1.0, 1.0)
        return float(scenario["amplitude_v"]) * signs
    time = indices.astype(np.float64) * control_period_s
    if kind == "sine":
        return float(scenario["amplitude_v"]) * np.sin(
            2.0 * math.pi * float(scenario["frequency_hz"]) * time
            + float(scenario.get("phase_rad", 0.0))
        )
    if kind == "multisine":
        result = np.zeros(steps, dtype=np.float64)
        for amplitude, frequency, phase in zip(
            scenario["amplitudes_v"],
            scenario["frequencies_hz"],
            scenario["phases_rad"],
            strict=True,
        ):
            result += float(amplitude) * np.sin(
                2.0 * math.pi * float(frequency) * time + float(phase)
            )
        return result
    raise ValueError(f"unsupported stress scenario kind: {kind}")


def generate_phase6b_artifacts(
    report: Phase6BValidationReport,
    output_directory: Path | str,
) -> tuple[Path, Path]:
    """Write the complete JSON report and a compact three-panel diagnostic."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("Phase-6B artifacts require matplotlib") from error

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "phase6b_latent_and_stability.json"
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    figure, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))
    latents = np.asarray(
        [int(value["latent_size"]) for value in report.pilot_aggregates]
    )
    medians = np.asarray(
        [
            float(value["median_validation_free_rollout_nrmse"])
            for value in report.pilot_aggregates
        ]
    )
    axes[0].plot(latents, medians, marker="o", color="#2b8cbe", label="median")
    for aggregate in report.pilot_aggregates:
        latent = int(aggregate["latent_size"])
        scores = np.asarray(aggregate["scores"], dtype=float)
        axes[0].scatter(
            np.full(scores.shape, latent),
            scores,
            color="#a6bddb",
            s=22,
            zorder=3,
        )
    selected_index = int(np.flatnonzero(latents == report.selected_latent_size)[0])
    axes[0].scatter(
        [report.selected_latent_size],
        [medians[selected_index]],
        color="#d7301f",
        s=65,
        zorder=4,
        label="selected",
    )
    axes[0].set_xlabel("Latent dimension")
    axes[0].set_ylabel("Pilot validation rollout NRMSE")
    axes[0].set_title("Repeated latent search")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    split_colors = {
        "validation": "#2b8cbe",
        "id_test": "#31a354",
        "ood_test": "#de2d26",
    }
    for split, color in split_colors.items():
        values = [value for value in report.replay_metrics if value.split == split]
        axes[1].plot(
            [value.horizon_steps / 1000.0 for value in values],
            [value.free_rollout_nrmse for value in values],
            marker="o",
            color=color,
            label=split,
        )
    axes[1].set_xlabel("Held-out horizon [s]")
    axes[1].set_ylabel("Free-rollout NRMSE")
    axes[1].set_title("Error accumulation")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    longest = max(value.horizon_steps for value in report.stress_metrics)
    longest_metrics = [
        value for value in report.stress_metrics if value.horizon_steps == longest
    ]
    scenario_names = tuple(
        dict.fromkeys(value.scenario for value in longest_metrics)
    )
    phase2 = load_phase2_spec()
    current_limit = phase2.limits["armature_current_a"].maximum
    speed_limit = phase2.limits["angular_speed_rad_s"].maximum
    utilization = []
    for scenario in scenario_names:
        values = [value for value in longest_metrics if value.scenario == scenario]
        utilization.append(
            max(
                max(
                    value.maximum_absolute_current_a / current_limit,
                    value.maximum_absolute_speed_rad_s / speed_limit,
                )
                for value in values
            )
        )
    colors = ["#de2d26" if value > 1.0 else "#74c476" for value in utilization]
    axes[2].barh(scenario_names, utilization, color=colors)
    axes[2].axvline(1.0, color="black", linestyle="--", linewidth=1)
    axes[2].set_xlabel("Worst physical-bound utilization")
    axes[2].set_title(f"Boundedness at {longest / 1000:g} s")
    axes[2].grid(axis="x", alpha=0.25)
    figure.suptitle(
        f"Phase 6B R2DN — latent {report.selected_latent_size}, "
        f"seed {report.selected_seed}"
    )
    figure.tight_layout()
    figure_path = output / "phase6b_latent_and_stability.png"
    figure.savefig(figure_path, dpi=170)
    plt.close(figure)
    return report_path, figure_path


def _fixed_stress_anchors(
    dataset: Phase4Dataset,
    *,
    split: str,
    count: int,
    burn_in_steps: int,
    seed: int,
) -> _AnchorBatch:
    rng = np.random.default_rng(seed)
    eligible = [
        trajectory_id
        for trajectory_id in dataset.trajectory_ids(split)
        if int(dataset.record(trajectory_id)["transitions"]) >= burn_in_steps
    ]
    if not eligible:
        raise ValueError(f"no {split} trajectory can provide Phase-6B burn-in")
    observation_windows = []
    control_windows = []
    initial_observations = []
    provenance = []
    normalization = dataset.normalization
    for _ in range(count):
        trajectory_id = eligible[int(rng.integers(0, len(eligible)))]
        trajectory = dataset.load_trajectory(trajectory_id)
        latest = trajectory.transitions - burn_in_steps
        start = int(rng.integers(0, latest + 1))
        stop = start + burn_in_steps
        observation_windows.append(trajectory.states[start:stop, :2])
        control_windows.append(trajectory.applied_voltages[start:stop])
        initial_observations.append(trajectory.states[stop, :2])
        provenance.append(
            StressAnchor(
                split=split,
                trajectory_id=trajectory_id,
                start_step=start,
                burn_in_steps=burn_in_steps,
            )
        )
    observations = np.stack(observation_windows, axis=1)
    controls = np.stack(control_windows, axis=1)
    initial = np.stack(initial_observations, axis=0)
    observations = (
        observations - normalization.observation_mean[None, None, :]
    ) / normalization.observation_std[None, None, :]
    controls = (
        controls - normalization.control_mean[None, None, :]
    ) / normalization.control_std[None, None, :]
    initial = (
        initial - normalization.observation_mean[None, :]
    ) / normalization.observation_std[None, :]
    return _AnchorBatch(
        observations=observations.astype(np.float32),
        controls=controls.astype(np.float32),
        initial_observations=initial.astype(np.float32),
        provenance=tuple(provenance),
    )


def _fixed_replay_windows(
    dataset: Phase4Dataset,
    *,
    split: str,
    count: int,
    burn_in_steps: int,
    rollout_steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    required = burn_in_steps + rollout_steps
    eligible = [
        trajectory_id
        for trajectory_id in dataset.trajectory_ids(split)
        if int(dataset.record(trajectory_id)["transitions"]) >= required
    ]
    if not eligible:
        raise ValueError(
            f"no {split} trajectory contains {required} transitions for replay"
        )
    observation_windows = []
    control_windows = []
    normalization = dataset.normalization
    for _ in range(count):
        trajectory_id = eligible[int(rng.integers(0, len(eligible)))]
        trajectory = dataset.load_trajectory(trajectory_id)
        latest = trajectory.transitions - required
        start = int(rng.integers(0, latest + 1))
        stop = start + required
        observation_windows.append(trajectory.states[start : stop + 1, :2])
        control_windows.append(trajectory.applied_voltages[start:stop])
    observations = np.stack(observation_windows, axis=1)
    controls = np.stack(control_windows, axis=1)
    observations = (
        observations - normalization.observation_mean[None, None, :]
    ) / normalization.observation_std[None, None, :]
    controls = (
        controls - normalization.control_mean[None, None, :]
    ) / normalization.control_std[None, None, :]
    return observations.astype(np.float32), controls.astype(np.float32)


def _tail_rms_growth(chunks: list[np.ndarray]) -> tuple[float, float]:
    if not chunks:
        return math.inf, math.inf
    values = np.concatenate(chunks, axis=0)
    midpoint = max(1, values.shape[0] // 2)
    first = float(np.sqrt(np.mean(values[:midpoint] ** 2)))
    second_values = values[midpoint:] if midpoint < values.shape[0] else values[-1:]
    second = float(np.sqrt(np.mean(second_values**2)))
    return _safe_ratio(second, first), second


def _tail_mean_growth(chunks: list[np.ndarray]) -> tuple[float, float]:
    if not chunks:
        return math.inf, math.inf
    values = np.concatenate(chunks, axis=0)
    midpoint = max(1, values.shape[0] // 2)
    first = float(np.mean(values[:midpoint]))
    second_values = values[midpoint:] if midpoint < values.shape[0] else values[-1:]
    second = float(np.mean(second_values))
    return _safe_ratio(second, first), second


def _tail_max(chunks: list[np.ndarray]) -> float:
    if not chunks:
        return math.inf
    return float(max(np.max(value) for value in chunks))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 1e-12:
        return 1.0 if numerator <= 1e-12 else math.inf
    return numerator / denominator
