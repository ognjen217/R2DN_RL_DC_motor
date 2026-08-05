"""Matched comparison of temperature-blind models against FULL/RK4."""

from __future__ import annotations

import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from r2dn_dc_motor.data import Phase4Dataset
from r2dn_dc_motor.models.isothermal_calibration import (
    IsothermalCalibrationCheckpoint,
)
from r2dn_dc_motor.models.jax_runtime import JAXRuntime
from r2dn_dc_motor.phase2_spec import Phase2Spec
from r2dn_dc_motor.plants import IsothermalWorldModel
from r2dn_dc_motor.validation.phase5 import build_isothermal_models
from r2dn_dc_motor.validation.r2dn_rk4_benchmark import (
    AccuracyMetrics,
    BenchmarkAnchorData,
    R2DNTrace,
    RK4Trace,
    calculate_accuracy,
)

MODEL_ORDER = ("R2DN", "ISO-NOM", "ISO-CAL")


@dataclass(frozen=True)
class IsothermalTiming:
    """End-to-end timing for one materialized isothermal rollout."""

    wall_time_s: float
    simulated_seconds_per_wall_second: float
    control_steps_completed: int


@dataclass(frozen=True)
class IsothermalTrace:
    """Physical current/speed predictions from a temperature-free model."""

    observations: np.ndarray
    timing: IsothermalTiming


@dataclass(frozen=True)
class HiddenThermalBenchmarkReport:
    """Serializable four-way hidden-thermal benchmark result."""

    schema_version: int
    passed: bool
    benchmark: str
    dataset_fingerprint: str
    scenario: dict[str, Any]
    duration_s: float
    control_period_s: float
    control_steps: int
    horizons_s: tuple[float, ...]
    anchor: dict[str, Any]
    reference: dict[str, Any]
    models: dict[str, dict[str, Any]]
    horizon_ranking: dict[str, list[dict[str, Any]]]
    interface_audit: dict[str, Any]
    hardware: dict[str, Any]
    interpretation: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            (
                "HIDDEN-THERMAL BENCHMARK: PASS"
                if self.passed
                else "HIDDEN-THERMAL BENCHMARK: FAIL"
            ),
            f"horizon: {self.duration_s:g} s ({self.control_steps} control steps)",
            f"scenario: {self.scenario['name']}",
            "combined NRMSE by cumulative horizon:",
        ]
        for horizon in self.horizons_s:
            key = _horizon_key(horizon)
            ranking = self.horizon_ranking[key]
            values = ", ".join(
                f"{row['model']}={row['combined_nrmse']:.6g}" for row in ranking
            )
            lines.append(f"  {horizon:g} s: {values}")
        return "\n".join(lines)


def run_isothermal_trace(
    model: IsothermalWorldModel,
    initial_observation: np.ndarray,
    physical_voltages_v: np.ndarray,
    *,
    duration_s: float,
) -> IsothermalTrace:
    """Run one fully autoregressive ISO-NOM or ISO-CAL trajectory."""

    voltages = np.asarray(physical_voltages_v, dtype=np.float64)
    initial = np.asarray(initial_observation, dtype=np.float64)
    if voltages.ndim != 1 or not np.isfinite(voltages).all():
        raise ValueError("benchmark voltages must be a finite one-dimensional array")
    if initial.shape != (2,) or not np.isfinite(initial).all():
        raise ValueError("initial observation must contain finite current and speed")

    started = time.perf_counter()
    observations = model.free_rollout(initial, voltages[:, None])[1:]
    elapsed = time.perf_counter() - started
    if not np.isfinite(observations).all():
        raise FloatingPointError(f"{model.name} produced NaN or infinite predictions")
    return IsothermalTrace(
        observations=np.asarray(observations, dtype=np.float64),
        timing=IsothermalTiming(
            wall_time_s=elapsed,
            simulated_seconds_per_wall_second=_safe_rate(duration_s, elapsed),
            control_steps_completed=observations.shape[0],
        ),
    )


def build_hidden_thermal_report(
    *,
    dataset: Phase4Dataset,
    checkpoint: Any,
    calibration: IsothermalCalibrationCheckpoint,
    phase2: Phase2Spec,
    scenario: dict[str, Any],
    duration_s: float,
    horizons_s: tuple[float, ...],
    anchor: BenchmarkAnchorData,
    r2dn: R2DNTrace,
    full_rk4: RK4Trace,
    isothermal: dict[str, IsothermalTrace],
    runtime: JAXRuntime,
) -> HiddenThermalBenchmarkReport:
    """Build metrics, rankings, provenance, and the no-temperature audit."""

    if tuple(isothermal) != ("ISO-NOM", "ISO-CAL"):
        raise ValueError("isothermal traces must be ordered as ISO-NOM then ISO-CAL")
    horizons = _validated_horizons(horizons_s, duration_s)
    period = phase2.integration_settings.control_period_s
    requested_steps = int(round(duration_s / period))
    observation_std = checkpoint.normalization.observation_std
    traces = {
        "R2DN": r2dn.observations,
        "ISO-NOM": isothermal["ISO-NOM"].observations,
        "ISO-CAL": isothermal["ISO-CAL"].observations,
    }
    accuracy = {
        name: calculate_accuracy(
            trace,
            full_rk4.observations,
            observation_std,
            control_period_s=period,
            requested_horizons_s=horizons,
        )
        for name, trace in traces.items()
    }

    iso_models = build_isothermal_models(calibration, phase2)
    interface_audit = {
        "candidate_temperature_input": False,
        "candidate_observations": [
            "armature_current_a",
            "angular_speed_rad_s",
        ],
        "candidate_control": ["armature_voltage_v"],
        "r2dn_burn_in_uses_temperature": False,
        "iso_nom_thermal_state": False,
        "iso_cal_thermal_state": False,
        "iso_cal_temperature_used_during_fit": calibration.temperature_used,
        "iso_cal_load_torque_used_during_fit": calibration.load_torque_used,
        "full_reference_evolves_internal_temperature_state": True,
    }
    interface_passed = bool(
        not calibration.temperature_used
        and not calibration.load_torque_used
        and all(
            model.observation_names
            == ("armature_current_a", "angular_speed_rad_s")
            and model.control_names == ("armature_voltage_v",)
            for model in iso_models.values()
        )
    )
    complete = bool(
        not full_rk4.terminated
        and full_rk4.timing.control_steps_completed == requested_steps
        and all(metrics.compared_steps == requested_steps for metrics in accuracy.values())
        and all(np.isfinite(trace).all() for trace in traces.values())
        and interface_passed
    )

    models = {
        "R2DN": {
            "role": "temperature_blind_data_driven_candidate",
            "temperature_input": False,
            "checkpoint_phase": str(checkpoint.manifest.phase),
            "selected_variant": getattr(checkpoint.manifest, "selected_variant", None),
            "latent_size": checkpoint.manifest.latent_size,
            "seed": checkpoint.manifest.seed,
            "parameter_sha256": checkpoint.manifest.parameter_sha256,
            "contractivity_margin": checkpoint.manifest.contractivity_margin,
            "timing": asdict(r2dn.timing),
            "accuracy": _accuracy_payload(accuracy["R2DN"]),
        },
        "ISO-NOM": {
            "role": "temperature_blind_nominal_physical_candidate",
            "temperature_input": False,
            "parameters": asdict(iso_models["ISO-NOM"].parameters),
            "timing": asdict(isothermal["ISO-NOM"].timing),
            "accuracy": _accuracy_payload(accuracy["ISO-NOM"]),
        },
        "ISO-CAL": {
            "role": "temperature_blind_calibrated_physical_candidate",
            "temperature_input": False,
            "parameters": asdict(iso_models["ISO-CAL"].parameters),
            "calibration": {
                "fit_split": calibration.fit_split,
                "fit_trajectory_count": len(calibration.fit_trajectory_ids),
                "temperature_used": calibration.temperature_used,
                "load_torque_used": calibration.load_torque_used,
                "selection_policy": calibration.selection_policy,
            },
            "timing": asdict(isothermal["ISO-CAL"].timing),
            "accuracy": _accuracy_payload(accuracy["ISO-CAL"]),
        },
    }
    return HiddenThermalBenchmarkReport(
        schema_version=1,
        passed=complete,
        benchmark="r2dn_iso_nom_iso_cal_vs_full_rk4",
        dataset_fingerprint=dataset.fingerprint,
        scenario=dict(scenario),
        duration_s=duration_s,
        control_period_s=period,
        control_steps=requested_steps,
        horizons_s=horizons,
        anchor=asdict(anchor.provenance),
        reference={
            "name": "FULL/RK4",
            "role": "ground_truth_baseline",
            "temperature_input": False,
            "internal_temperature_state": True,
            "true_temperature_dependence": True,
            "rk4_substeps_per_control_step": (
                full_rk4.timing.rk4_substeps_per_control_step
            ),
            "timing": asdict(full_rk4.timing),
            "terminated": full_rk4.terminated,
            "termination_reason": full_rk4.termination_reason,
            "initial_temperature_c": anchor.provenance.initial_temperature_c,
            "final_temperature_c": (
                float(full_rk4.temperatures_c[-1])
                if full_rk4.temperatures_c.size
                else math.nan
            ),
        },
        models=models,
        horizon_ranking=build_horizon_ranking(accuracy, horizons),
        interface_audit={**interface_audit, "passed": interface_passed},
        hardware={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "jax": runtime.to_dict(),
        },
        interpretation={
            "reference": (
                "FULL/RK4 is the ground-truth baseline. It evolves winding temperature "
                "internally and applies the true resistance-temperature law."
            ),
            "fair_input_scope": (
                "R2DN, ISO-NOM, and ISO-CAL receive only current, angular speed, and "
                "applied voltage; none receives temperature."
            ),
            "horizon_metric": (
                "Every reported horizon is cumulative from the shared anchor, not an "
                "independently reinitialized rollout."
            ),
            "claim_scope": (
                "The result tests whether the temperature-blind learned model captures "
                "hidden thermal effects better than incomplete isothermal physics on this "
                "held-out excitation."
            ),
        },
    )


def build_horizon_ranking(
    accuracy: dict[str, AccuracyMetrics],
    horizons_s: tuple[float, ...],
) -> dict[str, list[dict[str, Any]]]:
    """Rank all temperature-blind candidates at every cumulative horizon."""

    ranking: dict[str, list[dict[str, Any]]] = {}
    for horizon in horizons_s:
        rows = []
        for model_name in MODEL_ORDER:
            metric = _metric_at_horizon(accuracy[model_name], horizon)
            rows.append(
                {
                    "model": model_name,
                    "combined_nrmse": metric.combined_nrmse,
                    "current_nrmse": metric.current_nrmse,
                    "speed_nrmse": metric.speed_nrmse,
                }
            )
        rows.sort(key=lambda row: (row["combined_nrmse"], MODEL_ORDER.index(row["model"])))
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        ranking[_horizon_key(horizon)] = rows
    return ranking


def generate_hidden_thermal_artifacts(
    report: HiddenThermalBenchmarkReport,
    *,
    r2dn_observations: np.ndarray,
    isothermal_observations: dict[str, np.ndarray],
    full_rk4: RK4Trace,
    physical_voltages_v: np.ndarray,
    output_directory: Path | str,
    maximum_plot_points: int,
) -> tuple[Path, Path]:
    """Write JSON plus a trajectory, horizon, and hidden-temperature figure."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("hidden-thermal artifacts require matplotlib") from error

    if maximum_plot_points < 100:
        raise ValueError("maximum_plot_points must be at least 100")
    compared = report.control_steps
    traces = {
        "FULL/RK4": np.asarray(full_rk4.observations[:compared], dtype=np.float64),
        "R2DN": np.asarray(r2dn_observations[:compared], dtype=np.float64),
        "ISO-NOM": np.asarray(
            isothermal_observations["ISO-NOM"][:compared], dtype=np.float64
        ),
        "ISO-CAL": np.asarray(
            isothermal_observations["ISO-CAL"][:compared], dtype=np.float64
        ),
    }
    if any(trace.shape != (compared, 2) for trace in traces.values()):
        raise ValueError("every plotted observation trace must cover the complete horizon")

    stride = max(1, math.ceil(compared / maximum_plot_points))
    indices = np.arange(0, compared, stride, dtype=np.int64)
    if indices[-1] != compared - 1:
        indices = np.append(indices, compared - 1)
    time_s = (indices + 1) * report.control_period_s
    voltages = np.asarray(physical_voltages_v[:compared], dtype=np.float64)
    temperatures = np.asarray(full_rk4.temperatures_c[:compared], dtype=np.float64)

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"hidden_thermal_comparison_{report.duration_s:g}s"
    report_path = output / f"{stem}.json"
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    styles = {
        "FULL/RK4": {"color": "#202020", "linewidth": 1.45, "linestyle": "-"},
        "R2DN": {"color": "#2b6cb0", "linewidth": 1.0, "linestyle": "-"},
        "ISO-NOM": {"color": "#dd6b20", "linewidth": 0.95, "linestyle": "--"},
        "ISO-CAL": {"color": "#2f855a", "linewidth": 0.95, "linestyle": ":"},
    }
    figure, axes = plt.subplots(2, 2, figsize=(16.0, 9.2))
    for name, trace in traces.items():
        axes[0, 0].plot(time_s, trace[indices, 0], label=name, **styles[name])
        axes[0, 1].plot(time_s, trace[indices, 1], label=name, **styles[name])
    axes[0, 0].set_title("Armature-current rollout")
    axes[0, 0].set_ylabel("Current [A]")
    axes[0, 1].set_title("Angular-speed rollout")
    axes[0, 1].set_ylabel("Speed [rad/s]")
    axes[0, 0].legend(ncol=2)
    axes[0, 1].legend(ncol=2)

    for name in MODEL_ORDER:
        horizon_values = [
            _ranking_value(report, horizon, name, "combined_nrmse")
            for horizon in report.horizons_s
        ]
        axes[1, 0].plot(
            report.horizons_s,
            horizon_values,
            marker="o",
            markersize=5,
            label=name,
            **styles[name],
        )
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xticks(report.horizons_s)
    axes[1, 0].set_xticklabels([f"{value:g}" for value in report.horizons_s])
    axes[1, 0].set_title("Cumulative error against FULL/RK4")
    axes[1, 0].set_xlabel("Rollout horizon [s]")
    axes[1, 0].set_ylabel("Combined NRMSE")
    axes[1, 0].legend()

    axes[1, 1].plot(
        time_s,
        temperatures[indices],
        color="#805ad5",
        linewidth=1.1,
        label="FULL winding temperature",
    )
    axes[1, 1].set_title("Hidden thermal state and shared excitation")
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("Temperature [°C]")
    voltage_axis = axes[1, 1].twinx()
    voltage_axis.plot(
        time_s,
        voltages[indices],
        color="#718096",
        linewidth=0.65,
        alpha=0.55,
        label="Applied voltage",
    )
    voltage_axis.set_ylabel("Voltage [V]")
    handles, labels = axes[1, 1].get_legend_handles_labels()
    voltage_handles, voltage_labels = voltage_axis.get_legend_handles_labels()
    axes[1, 1].legend(handles + voltage_handles, labels + voltage_labels)

    for axis in axes.flat:
        axis.grid(True, alpha=0.22)
    axes[0, 0].set_xlabel("Time [s]")
    axes[0, 1].set_xlabel("Time [s]")
    figure.suptitle(
        "Temperature-blind model comparison with FULL electrothermal RK4 baseline"
    )
    figure.tight_layout()
    figure_path = output / f"{stem}.png"
    figure.savefig(figure_path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return report_path, figure_path


def _accuracy_payload(metrics: AccuracyMetrics) -> dict[str, Any]:
    payload = asdict(metrics)
    payload["horizons"] = [asdict(value) for value in metrics.horizons]
    return payload


def _metric_at_horizon(metrics: AccuracyMetrics, horizon_s: float) -> Any:
    for value in metrics.horizons:
        if math.isclose(value.duration_s, horizon_s, rel_tol=0.0, abs_tol=1e-9):
            return value
    raise ValueError(f"accuracy payload does not contain the {horizon_s:g} s horizon")


def _ranking_value(
    report: HiddenThermalBenchmarkReport,
    horizon_s: float,
    model_name: str,
    metric_name: str,
) -> float:
    for row in report.horizon_ranking[_horizon_key(horizon_s)]:
        if row["model"] == model_name:
            return float(row[metric_name])
    raise KeyError(model_name)


def _validated_horizons(
    horizons_s: tuple[float, ...],
    duration_s: float,
) -> tuple[float, ...]:
    values = tuple(float(value) for value in horizons_s)
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("horizons must contain positive finite durations")
    if tuple(sorted(set(values))) != values:
        raise ValueError("horizons must be strictly increasing and unique")
    if not math.isclose(values[-1], duration_s, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("the final requested horizon must equal duration_s")
    return values


def _horizon_key(duration_s: float) -> str:
    return f"{duration_s:g}s"


def _safe_rate(simulated_s: float, wall_s: float) -> float:
    return float(simulated_s / wall_s) if wall_s > 0.0 else math.inf
