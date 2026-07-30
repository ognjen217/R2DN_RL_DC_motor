"""Calibration checks and predictive evaluation for Phase-5 baselines."""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from r2dn_dc_motor.data import Phase4Dataset
from r2dn_dc_motor.models.isothermal_calibration import (
    IsothermalCalibrationCheckpoint,
    nominal_isothermal_parameters,
)
from r2dn_dc_motor.phase2_spec import Phase2Spec
from r2dn_dc_motor.phase5_spec import (
    EVALUATION_SPLITS,
    REQUIRED_PHASE5_CHECKS,
    Phase5Spec,
)
from r2dn_dc_motor.plants import IsothermalWorldModel


@dataclass(frozen=True)
class Phase5Check:
    """One auditable baseline-validation assertion."""

    name: str
    passed: bool
    metrics: dict[str, Any]
    criterion: str


@dataclass(frozen=True)
class BaselinePredictiveMetrics:
    """One model's teacher-forced, rollout, regime, and runtime metrics."""

    model_name: str
    one_step_nrmse: dict[str, float]
    one_step_channel_nrmse: dict[str, dict[str, float]]
    rollout_nrmse: dict[str, dict[str, float]]
    long_rollout_regime_nrmse: dict[str, float]
    runtime_transitions_per_s: float


@dataclass(frozen=True)
class Phase5ValidationReport:
    """Complete Phase-5 output for ISO-NOM and one fitted ISO-CAL."""

    phase: int
    validation: str
    passed: bool
    dataset_fingerprint: str
    calibrated_parameters: dict[str, float]
    metrics: dict[str, BaselinePredictiveMetrics]
    checks: tuple[Phase5Check, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_isothermal_models(
    checkpoint: IsothermalCalibrationCheckpoint,
    phase2: Phase2Spec,
) -> dict[str, IsothermalWorldModel]:
    return {
        "ISO-NOM": IsothermalWorldModel(
            nominal_isothermal_parameters(phase2),
            phase2.motor_limits,
            phase2.integration_settings,
            name="ISO-NOM",
        ),
        "ISO-CAL": IsothermalWorldModel(
            checkpoint.parameters,
            phase2.motor_limits,
            phase2.integration_settings,
            name="ISO-CAL",
        ),
    }


def evaluate_isothermal_model(
    model: IsothermalWorldModel,
    dataset: Phase4Dataset,
    spec: Phase5Spec,
) -> BaselinePredictiveMetrics:
    """Evaluate one model without exposing evaluation-only fields to prediction."""

    observation_std = dataset.normalization.observation_std
    one_step: dict[str, float] = {}
    one_step_channel: dict[str, dict[str, float]] = {}
    rollout: dict[str, dict[str, float]] = {}

    for split in EVALUATION_SPLITS:
        one_step_sse = np.zeros(2, dtype=np.float64)
        one_step_count = 0
        for trajectory_id in dataset.trajectory_ids(split):
            view = dataset.model_view(trajectory_id)
            observations = np.asarray(view.observations[:, 0, :], dtype=np.float64)
            controls = np.asarray(view.controls[:, 0, :], dtype=np.float64)
            predicted = model.predict_next_batch(observations[:-1], controls)
            normalized_error = (predicted - observations[1:]) / observation_std
            one_step_sse += np.sum(normalized_error**2, axis=0)
            one_step_count += normalized_error.shape[0]
        channel = np.sqrt(one_step_sse / one_step_count)
        one_step_channel[split] = {
            "armature_current_a": float(channel[0]),
            "angular_speed_rad_s": float(channel[1]),
        }
        one_step[split] = float(np.sqrt(np.sum(one_step_sse) / (2 * one_step_count)))

        rollout[split] = {}
        for horizon_name, horizon_s in spec.rollout_horizons_s.items():
            sse, count = _rollout_error(
                model,
                dataset,
                dataset.trajectory_ids(split),
                horizon_s=horizon_s,
            )
            rollout[split][horizon_name] = float(np.sqrt(sse / count))

    cold_ids: list[str] = []
    hot_ids: list[str] = []
    cold_limit = float(spec.evaluation["cold_initial_temperature_max_c"])
    hot_limit = float(spec.evaluation["hot_initial_temperature_min_c"])
    for split in EVALUATION_SPLITS:
        for trajectory_id in dataset.trajectory_ids(split):
            initial_temperature = float(
                dataset.record(trajectory_id)["initial_state"][
                    "winding_temperature_c"
                ]
            )
            if initial_temperature <= cold_limit:
                cold_ids.append(trajectory_id)
            if initial_temperature >= hot_limit:
                hot_ids.append(trajectory_id)
    regime: dict[str, float] = {}
    for name, trajectory_ids in (("cold", cold_ids), ("hot", hot_ids)):
        if not trajectory_ids:
            regime[name] = math.nan
            continue
        sse, count = _rollout_error(
            model,
            dataset,
            tuple(trajectory_ids),
            horizon_s=spec.rollout_horizons_s["long"],
        )
        regime[name] = float(np.sqrt(sse / count))

    transitions_per_s = _benchmark_model(model, dataset, spec)
    return BaselinePredictiveMetrics(
        model_name=model.name,
        one_step_nrmse=one_step,
        one_step_channel_nrmse=one_step_channel,
        rollout_nrmse=rollout,
        long_rollout_regime_nrmse=regime,
        runtime_transitions_per_s=transitions_per_s,
    )


def run_phase5_validation(
    dataset: Phase4Dataset,
    checkpoint: IsothermalCalibrationCheckpoint,
    *,
    spec: Phase5Spec,
    phase2: Phase2Spec,
) -> Phase5ValidationReport:
    """Validate the global checkpoint and evaluate both physical baselines."""

    checkpoint.validate(dataset)
    models = build_isothermal_models(checkpoint, phase2)
    metrics = {
        name: evaluate_isothermal_model(model, dataset, spec)
        for name, model in models.items()
    }
    nominal = nominal_isothermal_parameters(phase2)
    checks = (
        Phase5Check(
            name="models_share_temperature_free_interface",
            passed=all(
                model.observation_names
                == ("armature_current_a", "angular_speed_rad_s")
                and model.control_names == ("armature_voltage_v",)
                for model in models.values()
            ),
            metrics={
                "observations": list(models["ISO-NOM"].observation_names),
                "controls": list(models["ISO-NOM"].control_names),
                "thermal_state_present": False,
            },
            criterion="both models expose exactly [current, speed] plus applied voltage",
        ),
        Phase5Check(
            name="iso_nom_matches_phase2_nominal_parameters",
            passed=models["ISO-NOM"].parameters == nominal,
            metrics=asdict(nominal),
            criterion="ISO-NOM is an unchanged projection of Phase-2 nominal parameters",
        ),
        Phase5Check(
            name="calibration_uses_train_only_model_view",
            passed=(
                checkpoint.fit_split == "train"
                and checkpoint.fit_trajectory_ids == dataset.trajectory_ids("train")
                and not checkpoint.temperature_used
                and not checkpoint.load_torque_used
            ),
            metrics={
                "fit_split": checkpoint.fit_split,
                "fit_trajectory_count": len(checkpoint.fit_trajectory_ids),
                "temperature_used": checkpoint.temperature_used,
                "load_torque_used": checkpoint.load_torque_used,
                "allowed_fit_features": list(checkpoint.allowed_fit_features),
            },
            criterion="fit uses every whole train trajectory through [i, speed, voltage]",
        ),
        Phase5Check(
            name="checkpoint_is_single_global_and_dataset_bound",
            passed=(
                checkpoint.model_name == "ISO-CAL"
                and checkpoint.dataset_fingerprint == dataset.fingerprint
                and checkpoint.sufficient_statistics.trajectory_count
                == len(dataset.trajectory_ids("train"))
            ),
            metrics={
                "model_name": checkpoint.model_name,
                "dataset_fingerprint": checkpoint.dataset_fingerprint,
                "global_parameter_sets": 1,
            },
            criterion="one global ISO-CAL checkpoint is bound to this dataset fingerprint",
        ),
        Phase5Check(
            name="evaluation_splits_never_select_parameters",
            passed=(
                checkpoint.selection_policy
                == "single_locked_fit_no_validation_tuning"
                and not set(checkpoint.fit_trajectory_ids)
                & set(
                    trajectory_id
                    for split in EVALUATION_SPLITS
                    for trajectory_id in dataset.trajectory_ids(split)
                )
            ),
            metrics={
                "selection_policy": checkpoint.selection_policy,
                "evaluation_splits": list(EVALUATION_SPLITS),
            },
            criterion="validation, ID, and OOD trajectories never alter ISO-CAL",
        ),
        Phase5Check(
            name="required_metrics_are_finite",
            passed=_all_metrics_finite(metrics),
            metrics={
                name: {
                    "one_step_splits": sorted(result.one_step_nrmse),
                    "rollout_splits": sorted(result.rollout_nrmse),
                    "runtime_transitions_per_s": result.runtime_transitions_per_s,
                }
                for name, result in metrics.items()
            },
            criterion="one-step, short/medium/long, cold/hot, and runtime metrics are finite",
        ),
    )
    if {check.name for check in checks} != REQUIRED_PHASE5_CHECKS:
        raise RuntimeError("Phase-5 implementation check catalog drifted from the spec")
    return Phase5ValidationReport(
        phase=5,
        validation="physical_isothermal_baselines",
        passed=all(check.passed for check in checks),
        dataset_fingerprint=dataset.fingerprint,
        calibrated_parameters={
            "effective_resistance_ohm": (
                checkpoint.parameters.effective_resistance_ohm
            ),
            "viscous_friction_n_m_s_per_rad": (
                checkpoint.parameters.viscous_friction_n_m_s_per_rad
            ),
        },
        metrics=metrics,
        checks=checks,
    )


def generate_phase5_artifacts(
    report: Phase5ValidationReport,
    output_dir: Path | str,
) -> tuple[Path, Path]:
    """Write the complete JSON report and a compact baseline-comparison figure."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "phase5_baseline_evaluation.json"
    figure_path = destination / "phase5_baseline_comparison.png"
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    models = ("ISO-NOM", "ISO-CAL")
    horizons = ("one-step", "short", "medium", "long")
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    for axis, split in zip(axes.flat[:3], EVALUATION_SPLITS, strict=True):
        x = np.arange(len(horizons))
        width = 0.36
        for offset, model_name in zip((-0.5, 0.5), models, strict=True):
            metric = report.metrics[model_name]
            values = [
                metric.one_step_nrmse[split],
                metric.rollout_nrmse[split]["short"],
                metric.rollout_nrmse[split]["medium"],
                metric.rollout_nrmse[split]["long"],
            ]
            axis.bar(x + offset * width, values, width, label=model_name)
        axis.set_title(split.replace("_", " ").upper())
        axis.set_xticks(x, horizons)
        axis.set_ylabel("Train-normalized RMSE")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()

    axis = axes.flat[3]
    x = np.arange(2)
    width = 0.36
    for offset, model_name in zip((-0.5, 0.5), models, strict=True):
        values = [
            report.metrics[model_name].long_rollout_regime_nrmse["cold"],
            report.metrics[model_name].long_rollout_regime_nrmse["hot"],
        ]
        axis.bar(x + offset * width, values, width, label=model_name)
    axis.set_title("LONG ROLLOUT BY INITIAL THERMAL REGIME")
    axis.set_xticks(x, ("cold", "hot"))
    axis.set_ylabel("Train-normalized RMSE")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.suptitle("Phase 5 — physical baseline prediction errors")
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return report_path, figure_path


def _rollout_error(
    model: IsothermalWorldModel,
    dataset: Phase4Dataset,
    trajectory_ids: tuple[str, ...],
    *,
    horizon_s: float,
) -> tuple[float, int]:
    dt = model.integration.control_period_s
    requested_steps = max(1, int(round(horizon_s / dt)))
    observation_std = dataset.normalization.observation_std
    sse = 0.0
    count = 0
    for trajectory_id in trajectory_ids:
        view = dataset.model_view(trajectory_id)
        observations = np.asarray(view.observations[:, 0, :], dtype=np.float64)
        controls = np.asarray(view.controls[:, 0, :], dtype=np.float64)
        steps = min(requested_steps, controls.shape[0])
        predicted = model.free_rollout(observations[0], controls[:steps])
        error = (predicted[1:] - observations[1 : steps + 1]) / observation_std
        sse += float(np.sum(error**2))
        count += int(error.size)
    if count < 1:
        raise ValueError("rollout evaluation received no transitions")
    return sse, count


def _benchmark_model(
    model: IsothermalWorldModel,
    dataset: Phase4Dataset,
    spec: Phase5Spec,
) -> float:
    ids = tuple(
        trajectory_id
        for split in EVALUATION_SPLITS
        for trajectory_id in dataset.trajectory_ids(split)
    )
    maximum_steps = int(
        round(spec.rollout_horizons_s["long"] / model.integration.control_period_s)
    )
    prepared = []
    transitions = 0
    for trajectory_id in ids:
        view = dataset.model_view(trajectory_id)
        observations = np.asarray(view.observations[:, 0, :], dtype=np.float64)
        controls = np.asarray(view.controls[:, 0, :], dtype=np.float64)
        steps = min(maximum_steps, controls.shape[0])
        prepared.append((observations[0], controls[:steps]))
        transitions += steps
    rates: list[float] = []
    for _ in range(int(spec.evaluation["runtime_repeats"])):
        started = time.perf_counter()
        for initial, controls in prepared:
            model.free_rollout(initial, controls)
        elapsed = time.perf_counter() - started
        rates.append(transitions / elapsed)
    return float(statistics.median(rates))


def _all_metrics_finite(
    metrics: dict[str, BaselinePredictiveMetrics],
) -> bool:
    values: list[float] = []
    for result in metrics.values():
        values.extend(result.one_step_nrmse.values())
        for channels in result.one_step_channel_nrmse.values():
            values.extend(channels.values())
        for split in result.rollout_nrmse.values():
            values.extend(split.values())
        values.extend(result.long_rollout_regime_nrmse.values())
        values.append(result.runtime_transitions_per_s)
    return bool(values) and all(math.isfinite(value) and value >= 0.0 for value in values)
