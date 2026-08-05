"""Matched-horizon performance comparison between R2DN and FULL/RK4."""

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
from r2dn_dc_motor.models.jax_runtime import JAXRuntime
from r2dn_dc_motor.phase2_spec import Phase2Spec
from r2dn_dc_motor.phase6b_spec import Phase6BSpec
from r2dn_dc_motor.plants import MotorState
from r2dn_dc_motor.validation.phase6b import scenario_voltage


@dataclass(frozen=True)
class BenchmarkAnchor:
    """Exact measured-history anchor shared by the two model rollouts."""

    split: str
    trajectory_id: str
    start_step: int
    burn_in_steps: int
    initial_current_a: float
    initial_speed_rad_s: float
    initial_temperature_c: float


@dataclass(frozen=True)
class BenchmarkAnchorData:
    """Raw and normalized values required to initialize R2DN and FULL/RK4."""

    provenance: BenchmarkAnchor
    burn_in_observations_normalized: np.ndarray
    burn_in_controls_normalized: np.ndarray
    initial_observation_normalized: np.ndarray
    initial_full_state: MotorState


@dataclass(frozen=True)
class R2DNTiming:
    """Synchronized end-to-end R2DN timings, including trace materialization."""

    cold_wall_time_s: float
    warm_wall_time_s: float
    warm_burn_in_wall_time_s: float
    cold_simulated_seconds_per_wall_second: float
    warm_simulated_seconds_per_wall_second: float


@dataclass(frozen=True)
class RK4Timing:
    """Canonical NumPy/Python FULL-plant RK4 timing."""

    wall_time_s: float
    simulated_seconds_per_wall_second: float
    control_steps_completed: int
    rk4_substeps_per_control_step: int
    rk4_substeps_completed: int


@dataclass(frozen=True)
class HorizonAccuracy:
    """R2DN error against FULL/RK4 through one cumulative horizon."""

    horizon_steps: int
    duration_s: float
    combined_nrmse: float
    current_nrmse: float
    speed_nrmse: float
    current_rmse_a: float
    speed_rmse_rad_s: float


@dataclass(frozen=True)
class AccuracyMetrics:
    """Full-horizon physical and train-normalized R2DN errors."""

    compared_steps: int
    combined_nrmse: float
    current_nrmse: float
    speed_nrmse: float
    current_rmse_a: float
    speed_rmse_rad_s: float
    maximum_absolute_current_error_a: float
    maximum_absolute_speed_error_rad_s: float
    final_current_error_a: float
    final_speed_error_rad_s: float
    horizons: tuple[HorizonAccuracy, ...]


@dataclass(frozen=True)
class RK4Trace:
    """Materialized FULL/RK4 reference trace and termination evidence."""

    observations: np.ndarray
    temperatures_c: np.ndarray
    timing: RK4Timing
    terminated: bool
    termination_reason: str | None


@dataclass(frozen=True)
class R2DNTrace:
    """Materialized physical-unit R2DN predictions and synchronized timings."""

    observations: np.ndarray
    timing: R2DNTiming


@dataclass(frozen=True)
class R2DNRK4BenchmarkReport:
    """Serializable scientific result for one matched 1000-second comparison."""

    schema_version: int
    passed: bool
    benchmark: str
    dataset_fingerprint: str
    phase6b_report_passed: bool
    scenario: dict[str, Any]
    duration_s: float
    control_period_s: float
    control_steps: int
    chunk_steps: int
    model: dict[str, Any]
    anchor: BenchmarkAnchor
    runtime: dict[str, Any]
    accuracy: AccuracyMetrics
    rk4_terminated: bool
    rk4_termination_reason: str | None
    reference_final_temperature_c: float
    hardware: dict[str, Any]
    interpretation: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "anchor": asdict(self.anchor),
            "accuracy": {
                **asdict(self.accuracy),
                "horizons": [
                    asdict(value) for value in self.accuracy.horizons
                ],
            },
        }

    def summary(self) -> str:
        warm_speedup = float(self.runtime["speedup_rk4_over_r2dn_warm"])
        return "\n".join(
            (
                "R2DN VS FULL/RK4: PASS" if self.passed else "R2DN VS FULL/RK4: FAIL",
                (
                    f"horizon: {self.duration_s:g} s "
                    f"({self.control_steps} control steps)"
                ),
                f"scenario: {self.scenario['name']}",
                (
                    "R2DN latent / seed: "
                    f"{self.model['latent_size']} / {self.model['seed']}"
                ),
                (
                    "wall time R2DN warm / cold / FULL-RK4: "
                    f"{self.runtime['r2dn']['warm_wall_time_s']:.6g} / "
                    f"{self.runtime['r2dn']['cold_wall_time_s']:.6g} / "
                    f"{self.runtime['rk4']['wall_time_s']:.6g} s"
                ),
                f"measured warm speedup: {warm_speedup:.6g}x",
                (
                    "full-horizon NRMSE current / speed / combined: "
                    f"{self.accuracy.current_nrmse:.6g} / "
                    f"{self.accuracy.speed_nrmse:.6g} / "
                    f"{self.accuracy.combined_nrmse:.6g}"
                ),
                (
                    "timing scope: current implementations on recorded devices; "
                    "not a hardware-independent algorithmic speedup claim"
                ),
            )
        )


def load_benchmark_anchor(
    dataset: Phase4Dataset,
    phase6b_report_path: Path | str,
    *,
    split: str,
    anchor_index: int,
    checkpoint: Any,
    require_checkpoint_match: bool = True,
) -> tuple[BenchmarkAnchorData, dict[str, Any]]:
    """Load one exact Phase-6B stress anchor and verify all provenance bindings."""

    report = json.loads(Path(phase6b_report_path).read_text(encoding="utf-8"))
    if report.get("dataset_fingerprint") != dataset.fingerprint:
        raise ValueError("Phase-6B report belongs to another dataset")
    if require_checkpoint_match:
        if int(report.get("selected_latent_size", -1)) != checkpoint.manifest.latent_size:
            raise ValueError("Phase-6B report latent does not match the checkpoint")
        if int(report.get("selected_seed", -1)) != checkpoint.manifest.seed:
            raise ValueError("Phase-6B report seed does not match the checkpoint")
    anchors = [
        value
        for value in report.get("stress_anchors", ())
        if value.get("split") == split
    ]
    if not anchors:
        raise ValueError(f"Phase-6B report contains no stress anchor for {split!r}")
    if anchor_index < 0 or anchor_index >= len(anchors):
        raise ValueError(
            f"anchor index {anchor_index} is outside [0, {len(anchors) - 1}]"
        )
    raw_anchor = anchors[anchor_index]
    trajectory_id = str(raw_anchor["trajectory_id"])
    original_start = int(raw_anchor["start_step"])
    original_burn_in = int(raw_anchor["burn_in_steps"])
    stop = original_start + original_burn_in
    burn_in_steps = int(
        getattr(checkpoint.manifest, "burn_in_steps", original_burn_in)
    )
    start = stop - burn_in_steps
    trajectory = dataset.load_trajectory(trajectory_id)
    if start < 0 or stop > trajectory.transitions:
        raise ValueError(
            "checkpoint burn-in cannot end at the selected Phase-6B anchor state"
        )

    normalization = checkpoint.normalization
    observations = trajectory.states[start:stop, :2].astype(np.float64)
    controls = trajectory.applied_voltages[start:stop].astype(np.float64)
    initial_full = MotorState.from_array(
        trajectory.states[stop].astype(np.float64)
    )
    initial_observation = trajectory.states[stop, :2].astype(np.float64)
    observations = (
        observations - normalization.observation_mean[None, :]
    ) / normalization.observation_std[None, :]
    controls = (
        controls - normalization.control_mean[None, :]
    ) / normalization.control_std[None, :]
    initial_observation = (
        initial_observation - normalization.observation_mean
    ) / normalization.observation_std
    provenance = BenchmarkAnchor(
        split=split,
        trajectory_id=trajectory_id,
        start_step=start,
        burn_in_steps=burn_in_steps,
        initial_current_a=initial_full.current_a,
        initial_speed_rad_s=initial_full.speed_rad_s,
        initial_temperature_c=initial_full.temperature_c,
    )
    return (
        BenchmarkAnchorData(
            provenance=provenance,
            burn_in_observations_normalized=observations[:, None, :].astype(
                np.float32
            ),
            burn_in_controls_normalized=controls[:, None, :].astype(np.float32),
            initial_observation_normalized=initial_observation[None, :].astype(
                np.float32
            ),
            initial_full_state=initial_full,
        ),
        report,
    )


def run_r2dn_trace(
    checkpoint: Any,
    anchor: BenchmarkAnchorData,
    physical_voltages_v: np.ndarray,
    *,
    duration_s: float,
    chunk_steps: int,
) -> R2DNTrace:
    """Run one cold and one warm synchronized R2DN rollout."""

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as error:
        raise ImportError("R2DN/RK4 benchmark requires Phase-6 JAX dependencies") from error

    voltages = np.asarray(physical_voltages_v, dtype=np.float64)
    if voltages.ndim != 1 or not np.isfinite(voltages).all():
        raise ValueError("benchmark voltages must be a finite one-dimensional array")
    if chunk_steps < 1 or voltages.size % chunk_steps:
        raise ValueError("chunk_steps must be positive and divide the benchmark horizon")
    normalization = checkpoint.normalization
    normalized_controls = (
        voltages - normalization.control_mean[0]
    ) / normalization.control_std[0]
    normalized_controls = normalized_controls.astype(np.float32)
    adapter = checkpoint.adapter
    explicit = adapter.model.direct_to_explicit(checkpoint.parameters)

    @jax.jit
    def burn_in(observations: Any, controls: Any) -> Any:
        initial_state = jnp.zeros(
            (1, adapter.architecture.state_size),
            dtype=jnp.float32,
        )
        state, _ = adapter.burn_in(
            checkpoint.parameters,
            initial_state,
            observations,
            controls,
        )
        return state

    @jax.jit
    def run_chunk(
        state: Any,
        observation: Any,
        controls: Any,
    ) -> tuple[Any, Any, Any]:
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
            return (next_state, next_observation), next_observation

        (final_state, final_observation), predictions = jax.lax.scan(
            step,
            (state, observation),
            controls,
        )
        return final_state, final_observation, predictions

    burn_observations = jnp.asarray(
        anchor.burn_in_observations_normalized,
        dtype=jnp.float32,
    )
    burn_controls = jnp.asarray(
        anchor.burn_in_controls_normalized,
        dtype=jnp.float32,
    )
    initial_observation = jnp.asarray(
        anchor.initial_observation_normalized,
        dtype=jnp.float32,
    )

    def execute_once() -> tuple[np.ndarray, float]:
        started = time.perf_counter()
        state = burn_in(burn_observations, burn_controls)
        state.block_until_ready()
        observation = initial_observation
        chunks: list[np.ndarray] = []
        for start in range(0, normalized_controls.size, chunk_steps):
            control = jnp.asarray(
                normalized_controls[start : start + chunk_steps, None, None],
                dtype=jnp.float32,
            )
            state, observation, predictions = run_chunk(
                state,
                observation,
                control,
            )
            chunks.append(np.asarray(predictions)[:, 0, :])
        state.block_until_ready()
        predictions_normalized = np.concatenate(chunks, axis=0)
        elapsed = time.perf_counter() - started
        return predictions_normalized, elapsed

    _, cold_time = execute_once()
    predictions_normalized, warm_time = execute_once()

    burn_started = time.perf_counter()
    warmed_state = burn_in(burn_observations, burn_controls)
    warmed_state.block_until_ready()
    warm_burn_time = time.perf_counter() - burn_started
    predictions = (
        predictions_normalized
        * normalization.observation_std[None, :]
        + normalization.observation_mean[None, :]
    )
    if not np.isfinite(predictions).all():
        raise FloatingPointError("R2DN produced NaN or infinite benchmark predictions")
    return R2DNTrace(
        observations=np.asarray(predictions, dtype=np.float64),
        timing=R2DNTiming(
            cold_wall_time_s=cold_time,
            warm_wall_time_s=warm_time,
            warm_burn_in_wall_time_s=warm_burn_time,
            cold_simulated_seconds_per_wall_second=_safe_rate(
                duration_s,
                cold_time,
            ),
            warm_simulated_seconds_per_wall_second=_safe_rate(
                duration_s,
                warm_time,
            ),
        ),
    )


def run_rk4_trace(
    phase2: Phase2Spec,
    initial_state: MotorState,
    physical_voltages_v: np.ndarray,
    *,
    duration_s: float,
) -> RK4Trace:
    """Run the canonical FULL plant with its locked RK4 substepping."""

    voltages = np.asarray(physical_voltages_v, dtype=np.float64)
    if voltages.ndim != 1 or not np.isfinite(voltages).all():
        raise ValueError("benchmark voltages must be a finite one-dimensional array")
    plant = phase2.build_plant()
    plant.reset(initial_state)
    states = np.empty((voltages.size, 3), dtype=np.float64)
    completed = 0
    started = time.perf_counter()
    for index, voltage in enumerate(voltages):
        result = plant.step(float(voltage))
        states[index] = result.state.as_array()
        completed = index + 1
        if result.terminated:
            break
    elapsed = time.perf_counter() - started
    states = states[:completed]
    substeps = phase2.integration_settings.substeps_per_control
    return RK4Trace(
        observations=states[:, :2],
        temperatures_c=states[:, 2],
        timing=RK4Timing(
            wall_time_s=elapsed,
            simulated_seconds_per_wall_second=_safe_rate(
                completed * phase2.integration_settings.control_period_s,
                elapsed,
            ),
            control_steps_completed=completed,
            rk4_substeps_per_control_step=substeps,
            rk4_substeps_completed=completed * substeps,
        ),
        terminated=plant.terminated,
        termination_reason=(
            plant.termination_reason.value
            if plant.termination_reason is not None
            else None
        ),
    )


def calculate_accuracy(
    predictions: np.ndarray,
    reference: np.ndarray,
    observation_std: np.ndarray,
    *,
    control_period_s: float,
    requested_horizons_s: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0),
) -> AccuracyMetrics:
    """Calculate physical RMSE and the project's train-normalized NRMSE."""

    prediction = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(reference, dtype=np.float64)
    scale = np.asarray(observation_std, dtype=np.float64)
    if (
        prediction.ndim != 2
        or target.ndim != 2
        or prediction.shape[1:] != (2,)
        or target.shape[1:] != (2,)
    ):
        raise ValueError("prediction and reference traces must have shape (T, 2)")
    compared = min(prediction.shape[0], target.shape[0])
    if compared < 1:
        raise ValueError("at least one matched benchmark step is required")
    if scale.shape != (2,) or np.any(scale <= 0.0):
        raise ValueError("observation_std must contain two positive values")
    prediction = prediction[:compared]
    target = target[:compared]
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("prediction and reference traces must be finite")
    error = prediction - target
    normalized_error = error / scale[None, :]
    physical_rmse = np.sqrt(np.mean(error**2, axis=0))
    feature_nrmse = np.sqrt(np.mean(normalized_error**2, axis=0))
    horizons = []
    horizon_steps = {
        min(
            compared,
            max(1, int(round(float(horizon) / control_period_s))),
        )
        for horizon in requested_horizons_s
        if horizon > 0.0
    }
    horizon_steps.add(compared)
    for steps in sorted(horizon_steps):
        partial_error = error[:steps]
        partial_normalized = normalized_error[:steps]
        partial_physical = np.sqrt(np.mean(partial_error**2, axis=0))
        partial_feature = np.sqrt(np.mean(partial_normalized**2, axis=0))
        horizons.append(
            HorizonAccuracy(
                horizon_steps=steps,
                duration_s=steps * control_period_s,
                combined_nrmse=float(np.sqrt(np.mean(partial_normalized**2))),
                current_nrmse=float(partial_feature[0]),
                speed_nrmse=float(partial_feature[1]),
                current_rmse_a=float(partial_physical[0]),
                speed_rmse_rad_s=float(partial_physical[1]),
            )
        )
    return AccuracyMetrics(
        compared_steps=compared,
        combined_nrmse=float(np.sqrt(np.mean(normalized_error**2))),
        current_nrmse=float(feature_nrmse[0]),
        speed_nrmse=float(feature_nrmse[1]),
        current_rmse_a=float(physical_rmse[0]),
        speed_rmse_rad_s=float(physical_rmse[1]),
        maximum_absolute_current_error_a=float(np.max(np.abs(error[:, 0]))),
        maximum_absolute_speed_error_rad_s=float(np.max(np.abs(error[:, 1]))),
        final_current_error_a=float(error[-1, 0]),
        final_speed_error_rad_s=float(error[-1, 1]),
        horizons=tuple(horizons),
    )


def build_benchmark_report(
    *,
    dataset: Phase4Dataset,
    checkpoint: Any,
    phase6b_report: dict[str, Any],
    phase2: Phase2Spec,
    scenario: dict[str, Any],
    duration_s: float,
    chunk_steps: int,
    anchor: BenchmarkAnchorData,
    r2dn: R2DNTrace,
    rk4: RK4Trace,
    runtime: JAXRuntime,
) -> R2DNRK4BenchmarkReport:
    """Combine trace, timing, provenance, and scope into one report."""

    control_period = phase2.integration_settings.control_period_s
    requested_steps = int(round(duration_s / control_period))
    accuracy = calculate_accuracy(
        r2dn.observations,
        rk4.observations,
        checkpoint.normalization.observation_std,
        control_period_s=control_period,
        requested_horizons_s=(1.0, 10.0, 100.0, duration_s),
    )
    rk4_time = rk4.timing.wall_time_s
    r2dn_warm = r2dn.timing.warm_wall_time_s
    r2dn_cold = r2dn.timing.cold_wall_time_s
    runtime_payload = {
        "r2dn": asdict(r2dn.timing),
        "rk4": asdict(rk4.timing),
        "speedup_rk4_over_r2dn_warm": _safe_ratio(rk4_time, r2dn_warm),
        "speedup_rk4_over_r2dn_cold": _safe_ratio(rk4_time, r2dn_cold),
        "materialized_trace_bytes": {
            "r2dn_current_and_speed": int(r2dn.observations.nbytes),
            "rk4_current_speed_and_temperature": int(
                rk4.observations.nbytes + rk4.temperatures_c.nbytes
            ),
            "scope": (
                "Array payload only; this is not host-process or JAX allocator "
                "peak memory."
            ),
        },
        "timed_scope": (
            "Both methods materialize the complete current/speed trace. "
            "R2DN cold time includes JIT compilation and burn-in; warm time includes "
            "burn-in, chunk dispatch, device execution, and device-to-host trace transfer."
        ),
    }
    complete = bool(
        not rk4.terminated
        and rk4.timing.control_steps_completed == requested_steps
        and accuracy.compared_steps == requested_steps
        and np.isfinite(r2dn.observations).all()
        and np.isfinite(rk4.observations).all()
    )
    final_temperature = (
        float(rk4.temperatures_c[-1]) if rk4.temperatures_c.size else math.nan
    )
    return R2DNRK4BenchmarkReport(
        schema_version=1,
        passed=complete,
        benchmark="phase6c_r2dn_vs_full_rk4",
        dataset_fingerprint=dataset.fingerprint,
        phase6b_report_passed=bool(phase6b_report.get("passed", False)),
        scenario=dict(scenario),
        duration_s=duration_s,
        control_period_s=control_period,
        control_steps=requested_steps,
        chunk_steps=chunk_steps,
        model={
            "type": checkpoint.manifest.model_type,
            "checkpoint_phase": str(checkpoint.manifest.phase),
            "selected_variant": getattr(
                checkpoint.manifest,
                "selected_variant",
                None,
            ),
            "latent_size": checkpoint.manifest.latent_size,
            "seed": checkpoint.manifest.seed,
            "parameter_sha256": checkpoint.manifest.parameter_sha256,
            "observation_std": (
                checkpoint.normalization.observation_std.tolist()
            ),
            "selection_validation_nrmse": (
                checkpoint.manifest.validation_free_rollout_nrmse
            ),
            "contractivity_margin": checkpoint.manifest.contractivity_margin,
        },
        anchor=anchor.provenance,
        runtime=runtime_payload,
        accuracy=accuracy,
        rk4_terminated=rk4.terminated,
        rk4_termination_reason=rk4.termination_reason,
        reference_final_temperature_c=final_temperature,
        hardware={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "jax": runtime.to_dict(),
        },
        interpretation={
            "accuracy_reference": (
                "FULL electrothermal plant integrated by the locked classical RK4 "
                "solver with ten 0.1 ms substeps per 1 ms control interval."
            ),
            "compared_outputs": (
                "Armature current and angular speed only; R2DN does not predict "
                "the hidden winding temperature."
            ),
            "load_torque": (
                "The future FULL/RK4 rollout uses the declared default constant load. "
                "Load is intentionally unavailable to the frozen R2DN interface."
            ),
            "speed_claim": (
                "The measured ratio compares the current NumPy/Python FULL-RK4 "
                "implementation on CPU with the current JAX R2DN implementation on "
                "the recorded device. It is not hardware-independent algorithmic speedup."
            ),
            "stability_claim": (
                "A finite 1000-second matched rollout is empirical evidence, not a "
                "formal infinite-horizon stability or passivity proof."
            ),
        },
    )


def generate_benchmark_artifacts(
    report: R2DNRK4BenchmarkReport,
    *,
    r2dn_observations: np.ndarray,
    rk4_observations: np.ndarray,
    physical_voltages_v: np.ndarray,
    output_directory: Path | str,
    maximum_plot_points: int,
) -> tuple[Path, Path]:
    """Write the JSON report and a four-panel 1000-second comparison plot."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("R2DN/RK4 benchmark artifacts require matplotlib") from error

    if maximum_plot_points < 100:
        raise ValueError("maximum_plot_points must be at least 100")
    compared = report.accuracy.compared_steps
    prediction = np.asarray(r2dn_observations[:compared], dtype=np.float64)
    reference = np.asarray(rk4_observations[:compared], dtype=np.float64)
    voltages = np.asarray(physical_voltages_v[:compared], dtype=np.float64)
    stride = max(1, math.ceil(compared / maximum_plot_points))
    indices = np.arange(0, compared, stride, dtype=np.int64)
    if indices[-1] != compared - 1:
        indices = np.append(indices, compared - 1)
    time_s = (indices + 1) * report.control_period_s
    normalized_error = (
        prediction[indices] - reference[indices]
    ) / np.asarray(report.model["observation_std"], dtype=np.float64)[None, :]
    instantaneous_error = np.sqrt(np.mean(normalized_error**2, axis=1))

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"phase6c_r2dn_vs_rk4_{report.duration_s:g}s"
    report_path = output / f"{stem}.json"
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    figure, axes = plt.subplots(2, 2, figsize=(15.5, 9.0))
    axes[0, 0].plot(
        time_s,
        reference[indices, 0],
        color="#252525",
        linewidth=1.2,
        label="FULL/RK4",
    )
    axes[0, 0].plot(
        time_s,
        prediction[indices, 0],
        color="#2b8cbe",
        linewidth=0.9,
        alpha=0.85,
        label="R2DN",
    )
    axes[0, 0].set_ylabel("Armature current [A]")
    axes[0, 0].set_title("Current trajectory")
    axes[0, 0].legend()

    axes[0, 1].plot(
        time_s,
        reference[indices, 1],
        color="#252525",
        linewidth=1.2,
        label="FULL/RK4",
    )
    axes[0, 1].plot(
        time_s,
        prediction[indices, 1],
        color="#31a354",
        linewidth=0.9,
        alpha=0.85,
        label="R2DN",
    )
    axes[0, 1].set_ylabel("Angular speed [rad/s]")
    axes[0, 1].set_title("Speed trajectory")
    axes[0, 1].legend()

    axes[1, 0].plot(
        time_s,
        instantaneous_error,
        color="#de2d26",
        linewidth=0.9,
        label="instantaneous normalized error",
    )
    voltage_axis = axes[1, 0].twinx()
    voltage_axis.plot(
        time_s,
        voltages[indices],
        color="#969696",
        linewidth=0.6,
        alpha=0.35,
        label="voltage",
    )
    axes[1, 0].set_xlabel("Simulated time [s]")
    axes[1, 0].set_ylabel("Normalized output error")
    voltage_axis.set_ylabel("Voltage [V]", color="#737373")
    axes[1, 0].set_title(
        f"Error — full-horizon NRMSE={report.accuracy.combined_nrmse:.4f}"
    )

    timing_names = ("R2DN warm", "R2DN cold", "FULL/RK4")
    timing_values = (
        float(report.runtime["r2dn"]["warm_wall_time_s"]),
        float(report.runtime["r2dn"]["cold_wall_time_s"]),
        float(report.runtime["rk4"]["wall_time_s"]),
    )
    bars = axes[1, 1].bar(
        timing_names,
        timing_values,
        color=("#2b8cbe", "#9ecae1", "#525252"),
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel("Wall time [s], logarithmic scale")
    axes[1, 1].set_title(
        "Measured runtime — "
        f"{report.runtime['speedup_rk4_over_r2dn_warm']:.2f}x warm ratio"
    )
    for bar, value in zip(bars, timing_values, strict=True):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f" {value:.3g} s",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for axis in axes.flat:
        axis.grid(alpha=0.22)
    axes[0, 0].set_xlabel("Simulated time [s]")
    axes[0, 1].set_xlabel("Simulated time [s]")
    figure.suptitle(
        f"R2DN vs FULL/RK4 — {report.duration_s:g} s, "
        f"{report.scenario['name']}, latent {report.model['latent_size']}"
    )
    figure.tight_layout()
    figure_path = output / f"{stem}.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    return report_path, figure_path


def resolve_scenario(phase6b: Phase6BSpec, name: str) -> dict[str, Any]:
    """Return one locked Phase-6B voltage scenario by name."""

    matches = [value for value in phase6b.scenarios if value["name"] == name]
    if len(matches) != 1:
        choices = ", ".join(value["name"] for value in phase6b.scenarios)
        raise ValueError(f"unknown scenario {name!r}; choose one of: {choices}")
    return matches[0]


def build_voltage_trace(
    scenario: dict[str, Any],
    *,
    duration_s: float,
    control_period_s: float,
) -> np.ndarray:
    """Build the complete deterministic voltage trace for the benchmark."""

    raw_steps = duration_s / control_period_s
    steps = int(round(raw_steps))
    if (
        not math.isfinite(duration_s)
        or duration_s <= 0.0
        or not math.isclose(raw_steps, steps, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise ValueError("duration must contain a positive integer number of control steps")
    return scenario_voltage(
        scenario,
        start_step=0,
        steps=steps,
        control_period_s=control_period_s,
    )


def _safe_rate(simulated_seconds: float, wall_seconds: float) -> float:
    return _safe_ratio(simulated_seconds, wall_seconds)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return math.inf
    return numerator / denominator
