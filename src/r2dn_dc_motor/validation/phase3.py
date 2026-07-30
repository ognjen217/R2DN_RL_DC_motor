"""Gate-1 hidden-temperature observability and identifiability validation."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from r2dn_dc_motor.models.temperature_probe import (
    HISTORY_FEATURE_NAMES,
    INSTANTANEOUS_FEATURE_NAMES,
    ProbeTrajectory,
    StandardizedRidge,
    build_probe_samples,
)
from r2dn_dc_motor.phase2_spec import Phase2Spec, load_phase2_spec
from r2dn_dc_motor.phase3_spec import Phase3Spec, load_phase3_spec
from r2dn_dc_motor.plants import IntegrationSettings, MotorState, Rollout
from r2dn_dc_motor.validation.phase2 import ValidationCheck

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class ProbeWindowResult:
    """Held-out temperature-estimation result for one history duration."""

    history_s: float
    train_samples: int
    test_samples: int
    history_mae_c: float
    history_rmse_c: float
    history_r2: float
    instantaneous_mae_c: float
    mean_baseline_mae_c: float


@dataclass(frozen=True)
class Phase3ValidationReport:
    """Complete Gate-1 result and its quantitative evidence."""

    passed: bool
    checks: tuple[ValidationCheck, ...]
    physical_metrics: dict[str, float | bool]
    control_metrics: dict[str, float | bool]
    probe_results: tuple[ProbeWindowResult, ...]
    selected_history_s: float

    def to_dict(self) -> dict[str, object]:
        return {
            "phase": 3,
            "gate": "gate_1_hidden_temperature_observability",
            "passed": self.passed,
            "selected_history_s": self.selected_history_s,
            "physical_metrics": self.physical_metrics,
            "control_metrics": self.control_metrics,
            "probe_feature_names": {
                "history": list(HISTORY_FEATURE_NAMES),
                "instantaneous": list(INSTANTANEOUS_FEATURE_NAMES),
            },
            "probe_results": [asdict(result) for result in self.probe_results],
            "checks": [asdict(check) for check in self.checks],
        }

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"PHASE 3 GATE 1: {status}"]
        for check in self.checks:
            marker = "PASS" if check.passed else "FAIL"
            lines.append(f"[{marker}] {check.name}")
        selected = next(
            result
            for result in self.probe_results
            if math.isclose(result.history_s, self.selected_history_s)
        )
        lines.extend(
            [
                (
                    "thermal / solver ratio: "
                    f"{self.physical_metrics['signal_to_solver_ratio']:.6g}"
                ),
                (
                    "paired max differences: "
                    f"current={self.physical_metrics['maximum_current_difference_a']:.6g} A, "
                    "speed="
                    f"{self.physical_metrics['maximum_speed_difference_rad_s']:.6g} rad/s"
                ),
                (
                    f"history probe ({selected.history_s:g} s): "
                    f"MAE={selected.history_mae_c:.6g} C, "
                    f"instantaneous MAE={selected.instantaneous_mae_c:.6g} C"
                ),
            ]
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class _ControlTrace:
    times_s: FloatArray
    states: FloatArray
    voltages_v: FloatArray
    errors_rad_s: FloatArray
    tracking_iae: float
    control_effort_v_s: float
    terminated: bool


@dataclass(frozen=True)
class _Phase3Diagnostics:
    report: Phase3ValidationReport
    cold_open_loop: Rollout
    hot_open_loop: Rollout
    cold_resistance_ohm: FloatArray
    hot_resistance_ohm: FloatArray
    cold_acceleration_rad_s2: FloatArray
    hot_acceleration_rad_s2: FloatArray
    cold_control: _ControlTrace
    hot_control: _ControlTrace
    control_reference_rad_s: float
    selected_true_temperature_c: FloatArray
    selected_predicted_temperature_c: FloatArray


def run_phase3_validation(
    spec: Phase3Spec | None = None,
    *,
    phase2: Phase2Spec | None = None,
) -> Phase3ValidationReport:
    """Execute the full paired-trajectory and diagnostic-probe Gate 1."""

    return _run_phase3(spec=spec, phase2=phase2).report


def generate_phase3_artifacts(
    output_directory: Path | str,
    *,
    spec: Phase3Spec | None = None,
    phase2: Phase2Spec | None = None,
) -> tuple[Phase3ValidationReport, Path, Path]:
    """Run Gate 1 once and write its JSON report and diagnostic figure."""

    diagnostics = _run_phase3(spec=spec, phase2=phase2)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    report_path = output / "phase3_observability.json"
    report_path.write_text(
        json.dumps(diagnostics.report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    figure_path = output / "phase3_observability.png"
    _plot_phase3_diagnostics(diagnostics, figure_path)
    return diagnostics.report, report_path, figure_path


def _run_phase3(
    *,
    spec: Phase3Spec | None,
    phase2: Phase2Spec | None,
) -> _Phase3Diagnostics:
    phase2 = phase2 or load_phase2_spec()
    spec = spec or load_phase3_spec(phase2=phase2)

    (
        cold_open,
        hot_open,
        cold_resistance,
        hot_resistance,
        cold_acceleration,
        hot_acceleration,
        physical_metrics,
    ) = _paired_open_loop(spec, phase2)
    cold_control = _diagnostic_pi_rollout(
        spec,
        phase2,
        float(spec.paired_experiment["cold_temperature_c"]),
    )
    hot_control = _diagnostic_pi_rollout(
        spec,
        phase2,
        float(spec.paired_experiment["hot_temperature_c"]),
    )
    control_metrics = _control_metrics(cold_control, hot_control)
    (
        probe_results,
        selected_true,
        selected_predicted,
        split_is_valid,
        pilot_terminated,
    ) = _run_temperature_probe(spec, phase2)

    checks = _gate_checks(
        spec,
        physical_metrics,
        control_metrics,
        probe_results,
        split_is_valid=split_is_valid,
        pilot_terminated=pilot_terminated,
    )
    actual_names = {check.name for check in checks}
    passed = (
        actual_names == set(spec.gate["required_checks"])
        and all(check.passed for check in checks)
    )
    report = Phase3ValidationReport(
        passed=passed,
        checks=checks,
        physical_metrics=physical_metrics,
        control_metrics=control_metrics,
        probe_results=probe_results,
        selected_history_s=float(spec.probe["selected_history_s"]),
    )
    return _Phase3Diagnostics(
        report=report,
        cold_open_loop=cold_open,
        hot_open_loop=hot_open,
        cold_resistance_ohm=cold_resistance,
        hot_resistance_ohm=hot_resistance,
        cold_acceleration_rad_s2=cold_acceleration,
        hot_acceleration_rad_s2=hot_acceleration,
        cold_control=cold_control,
        hot_control=hot_control,
        control_reference_rad_s=float(spec.closed_loop["reference_rad_s"]),
        selected_true_temperature_c=selected_true,
        selected_predicted_temperature_c=selected_predicted,
    )


def _paired_open_loop(
    spec: Phase3Spec,
    phase2: Phase2Spec,
) -> tuple[
    Rollout,
    Rollout,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    dict[str, float | bool],
]:
    pair = spec.paired_experiment
    steps = spec.steps(float(pair["duration_s"]), phase2)
    hold_steps = spec.steps(float(pair["voltage_hold_s"]), phase2)
    pattern = np.asarray(pair["voltage_pattern_v"], dtype=np.float64)
    repeats = math.ceil(steps / (hold_steps * pattern.size))
    voltages = np.tile(np.repeat(pattern, hold_steps), repeats)[:steps]
    initial_current = float(pair["initial_current_a"])
    initial_speed = float(pair["initial_speed_rad_s"])
    load = float(pair["load_torque_n_m"])

    cold_state = MotorState(
        initial_current,
        initial_speed,
        float(pair["cold_temperature_c"]),
    )
    hot_state = MotorState(
        initial_current,
        initial_speed,
        float(pair["hot_temperature_c"]),
    )
    cold = phase2.build_plant().rollout(
        voltages,
        initial_state=cold_state,
        load_torques_n_m=load,
    )
    hot = phase2.build_plant().rollout(
        voltages,
        initial_state=hot_state,
        load_torques_n_m=load,
    )

    divisor = int(spec.solver_comparison["fine_step_divisor"])
    integration = phase2.integration_settings
    fine_integration = IntegrationSettings(
        control_period_s=integration.control_period_s,
        integrator_step_s=integration.integrator_step_s / divisor,
    )
    cold_fine = phase2.build_plant(integration=fine_integration).rollout(
        voltages,
        initial_state=cold_state,
        load_torques_n_m=load,
    )
    hot_fine = phase2.build_plant(integration=fine_integration).rollout(
        voltages,
        initial_state=hot_state,
        load_torques_n_m=load,
    )

    plant = phase2.build_plant()
    cold_resistance = np.asarray(
        [plant.resistance_ohm(value) for value in cold.states[:, 2]],
        dtype=np.float64,
    )
    hot_resistance = np.asarray(
        [plant.resistance_ohm(value) for value in hot.states[:, 2]],
        dtype=np.float64,
    )
    cold_acceleration = _acceleration_trace(phase2, cold)
    hot_acceleration = _acceleration_trace(phase2, hot)

    output_difference = np.abs(cold.states[:, :2] - hot.states[:, :2])
    scales = np.asarray(spec.solver_comparison["output_scales"], dtype=np.float64)
    thermal_signal = float(np.max(output_difference / scales))
    solver_error = max(
        float(np.max(np.abs(cold.states[:, :2] - cold_fine.states[:, :2]) / scales)),
        float(np.max(np.abs(hot.states[:, :2] - hot_fine.states[:, :2]) / scales)),
    )
    reference = float(pair["tracking_reference_rad_s"])
    dt = integration.control_period_s
    cold_open_iae = float(np.sum(np.abs(reference - cold.states[:-1, 1])) * dt)
    hot_open_iae = float(np.sum(np.abs(reference - hot.states[:-1, 1])) * dt)

    metrics: dict[str, float | bool] = {
        "cold_terminated": cold.terminated,
        "hot_terminated": hot.terminated,
        "cold_fine_terminated": cold_fine.terminated,
        "hot_fine_terminated": hot_fine.terminated,
        "maximum_current_difference_a": float(np.max(output_difference[:, 0])),
        "maximum_speed_difference_rad_s": float(np.max(output_difference[:, 1])),
        "maximum_acceleration_difference_rad_s2": float(
            np.max(np.abs(cold_acceleration - hot_acceleration))
        ),
        "maximum_resistance_difference_ohm": float(
            np.max(np.abs(cold_resistance - hot_resistance))
        ),
        "initial_relative_resistance_difference": float(
            (hot_resistance[0] - cold_resistance[0]) / cold_resistance[0]
        ),
        "maximum_scaled_thermal_output_signal": thermal_signal,
        "maximum_scaled_solver_error": solver_error,
        "signal_to_solver_ratio": thermal_signal / max(solver_error, 1e-16),
        "cold_open_loop_tracking_iae": cold_open_iae,
        "hot_open_loop_tracking_iae": hot_open_iae,
        "open_loop_tracking_iae_relative_difference": _relative_difference(
            cold_open_iae,
            hot_open_iae,
        ),
    }
    return (
        cold,
        hot,
        cold_resistance,
        hot_resistance,
        cold_acceleration,
        hot_acceleration,
        metrics,
    )


def _acceleration_trace(phase2: Phase2Spec, rollout: Rollout) -> FloatArray:
    plant = phase2.build_plant()
    values = [
        plant.derivative(state, voltage, load)[1]
        for state, voltage, load in zip(
            rollout.states[:-1],
            rollout.applied_voltages_v,
            rollout.load_torques_n_m,
            strict=True,
        )
    ]
    return np.asarray(values, dtype=np.float64)


def _diagnostic_pi_rollout(
    spec: Phase3Spec,
    phase2: Phase2Spec,
    temperature_c: float,
) -> _ControlTrace:
    settings = spec.closed_loop
    steps = spec.steps(float(settings["duration_s"]), phase2)
    dt = phase2.integration_settings.control_period_s
    reference = float(settings["reference_rad_s"])
    proportional = float(settings["proportional_gain_v_per_rad_s"])
    integral_gain = float(settings["integral_gain_v_per_rad"])
    limits = phase2.motor_limits
    plant = phase2.build_plant()
    plant.reset(MotorState(0.0, 0.0, temperature_c))

    integral_error = 0.0
    times = [0.0]
    states = [plant.state.as_array()]
    voltages: list[float] = []
    errors: list[float] = []
    for _ in range(steps):
        error = reference - plant.state.speed_rad_s
        unconstrained = proportional * error + integral_gain * integral_error
        constrained = float(
            np.clip(
                unconstrained,
                limits.minimum_voltage_v,
                limits.maximum_voltage_v,
            )
        )
        saturation_residual = unconstrained - constrained
        if math.isclose(saturation_residual, 0.0, abs_tol=1e-15) or (
            math.copysign(1.0, error) != math.copysign(1.0, saturation_residual)
        ):
            integral_error += error * dt

        result = plant.step(constrained)
        times.append(plant.time_s)
        states.append(result.state.as_array())
        voltages.append(result.applied_voltage_v)
        errors.append(error)
        if result.terminated:
            break

    voltage_array = np.asarray(voltages, dtype=np.float64)
    error_array = np.asarray(errors, dtype=np.float64)
    return _ControlTrace(
        times_s=np.asarray(times, dtype=np.float64),
        states=np.asarray(states, dtype=np.float64),
        voltages_v=voltage_array,
        errors_rad_s=error_array,
        tracking_iae=float(np.sum(np.abs(error_array)) * dt),
        control_effort_v_s=float(np.sum(np.abs(voltage_array)) * dt),
        terminated=plant.terminated,
    )


def _control_metrics(
    cold: _ControlTrace,
    hot: _ControlTrace,
) -> dict[str, float | bool]:
    return {
        "cold_terminated": cold.terminated,
        "hot_terminated": hot.terminated,
        "cold_tracking_iae": cold.tracking_iae,
        "hot_tracking_iae": hot.tracking_iae,
        "tracking_iae_relative_difference": _relative_difference(
            cold.tracking_iae,
            hot.tracking_iae,
        ),
        "cold_control_effort_v_s": cold.control_effort_v_s,
        "hot_control_effort_v_s": hot.control_effort_v_s,
        "control_effort_relative_difference": _relative_difference(
            cold.control_effort_v_s,
            hot.control_effort_v_s,
        ),
        "cold_maximum_voltage_v": float(np.max(np.abs(cold.voltages_v))),
        "hot_maximum_voltage_v": float(np.max(np.abs(hot.voltages_v))),
    }


def _run_temperature_probe(
    spec: Phase3Spec,
    phase2: Phase2Spec,
) -> tuple[
    tuple[ProbeWindowResult, ...],
    FloatArray,
    FloatArray,
    bool,
    bool,
]:
    trajectories, pilot_terminated = _generate_pilot_trajectories(spec, phase2)
    train_ids = set(spec.train_trajectory_ids)
    test_ids = set(spec.test_trajectory_ids)
    train = tuple(item for item in trajectories if item.trajectory_id in train_ids)
    test = tuple(item for item in trajectories if item.trajectory_id in test_ids)
    split_is_valid = (
        train_ids.isdisjoint(test_ids)
        and {item.trajectory_id for item in train} == train_ids
        and {item.trajectory_id for item in test} == test_ids
        and len(train) + len(test) == len(trajectories)
    )

    dt = phase2.integration_settings.control_period_s
    stride_steps = spec.steps(float(spec.probe["sample_stride_s"]), phase2)
    minimum_current = float(spec.probe["minimum_informative_current_a"])
    ridge_l2 = float(spec.probe["ridge_l2"])
    results: list[ProbeWindowResult] = []
    selected_true = np.asarray([], dtype=np.float64)
    selected_predicted = np.asarray([], dtype=np.float64)

    for history_s in spec.history_windows_s:
        history_steps = spec.steps(history_s, phase2)
        train_samples = build_probe_samples(
            train,
            parameters=phase2.motor_parameters,
            control_period_s=dt,
            history_steps=history_steps,
            sample_stride_steps=stride_steps,
            minimum_informative_current_a=minimum_current,
        )
        test_samples = build_probe_samples(
            test,
            parameters=phase2.motor_parameters,
            control_period_s=dt,
            history_steps=history_steps,
            sample_stride_steps=stride_steps,
            minimum_informative_current_a=minimum_current,
        )
        history_model = StandardizedRidge.fit(
            train_samples.history_features,
            train_samples.target_temperature_c,
            l2=ridge_l2,
        )
        instant_model = StandardizedRidge.fit(
            train_samples.instantaneous_features,
            train_samples.target_temperature_c,
            l2=ridge_l2,
        )
        history_prediction = history_model.predict(test_samples.history_features)
        instantaneous_prediction = instant_model.predict(
            test_samples.instantaneous_features
        )
        target = test_samples.target_temperature_c
        mean_prediction = np.full_like(target, np.mean(train_samples.target_temperature_c))
        history_error = history_prediction - target
        denominator = float(np.sum((target - np.mean(target)) ** 2))
        r2 = 1.0 - float(np.sum(history_error**2)) / max(denominator, 1e-16)
        result = ProbeWindowResult(
            history_s=history_s,
            train_samples=target_count(train_samples.target_temperature_c),
            test_samples=target_count(target),
            history_mae_c=float(np.mean(np.abs(history_error))),
            history_rmse_c=float(np.sqrt(np.mean(history_error**2))),
            history_r2=r2,
            instantaneous_mae_c=float(
                np.mean(np.abs(instantaneous_prediction - target))
            ),
            mean_baseline_mae_c=float(np.mean(np.abs(mean_prediction - target))),
        )
        results.append(result)
        if math.isclose(history_s, float(spec.probe["selected_history_s"])):
            selected_true = target.copy()
            selected_predicted = history_prediction.copy()

    if selected_true.size == 0:
        raise ValueError("selected history result was not generated")
    return (
        tuple(results),
        selected_true,
        selected_predicted,
        split_is_valid,
        pilot_terminated,
    )


def _generate_pilot_trajectories(
    spec: Phase3Spec,
    phase2: Phase2Spec,
) -> tuple[tuple[ProbeTrajectory, ...], bool]:
    pilot = spec.pilot
    count = int(pilot["trajectory_count"])
    steps = spec.steps(float(pilot["duration_s"]), phase2)
    hold_steps = spec.steps(float(pilot["voltage_hold_s"]), phase2)
    chunk_count = math.ceil(steps / hold_steps)
    levels = np.asarray(pilot["voltage_levels_v"], dtype=np.float64)
    low_temperature, high_temperature = (
        float(value) for value in pilot["initial_temperature_c"]
    )
    temperatures = np.linspace(low_temperature, high_temperature, count)
    trajectories: list[ProbeTrajectory] = []
    any_terminated = False

    for trajectory_id, temperature in enumerate(temperatures):
        rng = np.random.default_rng(int(pilot["seed_base"]) + trajectory_id)
        chunks = rng.choice(levels, size=chunk_count, replace=True)
        controls = np.repeat(chunks, hold_steps)[:steps]
        rollout = phase2.build_plant().rollout(
            controls,
            initial_state=MotorState(
                float(pilot["initial_current_a"]),
                float(pilot["initial_speed_rad_s"]),
                float(temperature),
            ),
            load_torques_n_m=float(pilot["load_torque_n_m"]),
        )
        any_terminated = any_terminated or rollout.terminated
        trajectories.append(
            ProbeTrajectory(
                trajectory_id=trajectory_id,
                states=rollout.states,
                controls_v=rollout.applied_voltages_v,
            )
        )
    return tuple(trajectories), any_terminated


def _gate_checks(
    spec: Phase3Spec,
    physical: dict[str, float | bool],
    control: dict[str, float | bool],
    probe_results: tuple[ProbeWindowResult, ...],
    *,
    split_is_valid: bool,
    pilot_terminated: bool,
) -> tuple[ValidationCheck, ...]:
    gate = spec.gate
    selected = next(
        result
        for result in probe_results
        if math.isclose(result.history_s, float(spec.probe["selected_history_s"]))
    )
    improvement = 1.0 - selected.history_mae_c / max(
        selected.instantaneous_mae_c,
        1e-16,
    )
    no_physical_termination = not any(
        bool(physical[name])
        for name in (
            "cold_terminated",
            "hot_terminated",
            "cold_fine_terminated",
            "hot_fine_terminated",
        )
    )
    no_control_termination = not bool(control["cold_terminated"]) and not bool(
        control["hot_terminated"]
    )
    temperature_free = (
        "winding_temperature_c" not in HISTORY_FEATURE_NAMES
        and "winding_temperature_c" not in INSTANTANEOUS_FEATURE_NAMES
        and all(
            "temperature" not in str(signal).lower()
            for signal in spec.probe["allowed_signals"]
        )
    )

    return (
        ValidationCheck(
            name="thermal_signal_exceeds_solver_error",
            passed=(
                no_physical_termination
                and float(physical["signal_to_solver_ratio"])
                >= float(gate["minimum_signal_to_solver_ratio"])
            ),
            metrics={
                "signal_to_solver_ratio": float(physical["signal_to_solver_ratio"]),
                "minimum_ratio": float(gate["minimum_signal_to_solver_ratio"]),
                "scaled_thermal_signal": float(
                    physical["maximum_scaled_thermal_output_signal"]
                ),
                "scaled_solver_error": float(physical["maximum_scaled_solver_error"]),
            },
            criterion="paired thermal output separation dominates RK4 refinement error",
        ),
        ValidationCheck(
            name="open_loop_outputs_are_separated",
            passed=(
                no_physical_termination
                and float(physical["maximum_current_difference_a"])
                >= float(gate["minimum_max_current_difference_a"])
                and float(physical["maximum_speed_difference_rad_s"])
                >= float(gate["minimum_max_speed_difference_rad_s"])
                and float(physical["maximum_acceleration_difference_rad_s2"])
                >= float(gate["minimum_max_acceleration_difference_rad_s2"])
                and float(physical["initial_relative_resistance_difference"])
                >= float(gate["minimum_relative_resistance_difference"])
            ),
            metrics={
                "maximum_current_difference_a": float(
                    physical["maximum_current_difference_a"]
                ),
                "maximum_speed_difference_rad_s": float(
                    physical["maximum_speed_difference_rad_s"]
                ),
                "maximum_acceleration_difference_rad_s2": float(
                    physical["maximum_acceleration_difference_rad_s2"]
                ),
                "initial_relative_resistance_difference": float(
                    physical["initial_relative_resistance_difference"]
                ),
            },
            criterion="same [i0, omega0, u] yields physically meaningful hot/cold separation",
        ),
        ValidationCheck(
            name="closed_loop_control_is_affected",
            passed=(
                no_control_termination
                and float(control["tracking_iae_relative_difference"])
                >= float(gate["minimum_tracking_iae_relative_difference"])
                and float(control["control_effort_relative_difference"])
                >= float(gate["minimum_control_effort_relative_difference"])
            ),
            metrics={
                "tracking_iae_relative_difference": float(
                    control["tracking_iae_relative_difference"]
                ),
                "control_effort_relative_difference": float(
                    control["control_effort_relative_difference"]
                ),
                "cold_tracking_iae": float(control["cold_tracking_iae"]),
                "hot_tracking_iae": float(control["hot_tracking_iae"]),
            },
            criterion="hidden thermal state changes both tracking IAE and voltage effort",
        ),
        ValidationCheck(
            name="temperature_is_not_in_probe_features",
            passed=temperature_free,
            metrics={
                "history_feature_count": len(HISTORY_FEATURE_NAMES),
                "instantaneous_feature_count": len(INSTANTANEOUS_FEATURE_NAMES),
                "temperature_feature_present": not temperature_free,
            },
            criterion="probe features are derived exclusively from histories of [i, omega, u]",
        ),
        ValidationCheck(
            name="pilot_split_is_by_trajectory",
            passed=split_is_valid and not pilot_terminated,
            metrics={
                "training_trajectories": len(spec.train_trajectory_ids),
                "test_trajectories": len(spec.test_trajectory_ids),
                "pilot_terminated": pilot_terminated,
            },
            criterion="train/test trajectory IDs are disjoint and all pilot rollouts remain safe",
        ),
        ValidationCheck(
            name="instantaneous_probe_is_nontrivial",
            passed=(
                selected.instantaneous_mae_c
                >= float(gate["minimum_instantaneous_probe_mae_c"])
            ),
            metrics={
                "instantaneous_mae_c": selected.instantaneous_mae_c,
                "minimum_mae_c": float(gate["minimum_instantaneous_probe_mae_c"]),
                "mean_baseline_mae_c": selected.mean_baseline_mae_c,
            },
            criterion="one instantaneous [i, omega, u] sample does not reveal temperature",
        ),
        ValidationCheck(
            name="history_probe_recovers_temperature",
            passed=(
                selected.test_samples >= int(gate["minimum_test_samples"])
                and selected.history_mae_c
                <= float(gate["maximum_history_probe_mae_c"])
                and improvement
                >= float(gate["minimum_history_mae_improvement_fraction"])
            ),
            metrics={
                "history_s": selected.history_s,
                "test_samples": selected.test_samples,
                "history_mae_c": selected.history_mae_c,
                "maximum_mae_c": float(gate["maximum_history_probe_mae_c"]),
                "improvement_over_instantaneous": improvement,
            },
            criterion="held-out history probe recovers T with material gain over one sample",
        ),
    )


def _plot_phase3_diagnostics(
    diagnostics: _Phase3Diagnostics,
    output_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "plot generation requires: python -m pip install -e '.[phase3]'"
        ) from error

    cold = diagnostics.cold_open_loop
    hot = diagnostics.hot_open_loop
    cold_control = diagnostics.cold_control
    hot_control = diagnostics.hot_control
    report = diagnostics.report
    figure, axes = plt.subplots(4, 2, figsize=(12, 14), constrained_layout=True)

    axes[0, 0].plot(cold.times_s, cold.states[:, 0], label="cold")
    axes[0, 0].plot(hot.times_s, hot.states[:, 0], label="hot")
    axes[0, 0].set(title="Paired armature current", xlabel="Time [s]", ylabel="Current [A]")

    axes[0, 1].plot(cold.times_s, cold.states[:, 1], label="cold")
    axes[0, 1].plot(hot.times_s, hot.states[:, 1], label="hot")
    axes[0, 1].set(title="Paired angular speed", xlabel="Time [s]", ylabel="Speed [rad/s]")

    axes[1, 0].plot(cold.times_s, diagnostics.cold_resistance_ohm, label="cold")
    axes[1, 0].plot(hot.times_s, diagnostics.hot_resistance_ohm, label="hot")
    axes[1, 0].set(title="Winding resistance", xlabel="Time [s]", ylabel="Resistance [ohm]")

    axes[1, 1].plot(
        cold.times_s[:-1],
        diagnostics.cold_acceleration_rad_s2,
        label="cold",
    )
    axes[1, 1].plot(
        hot.times_s[:-1],
        diagnostics.hot_acceleration_rad_s2,
        label="hot",
    )
    axes[1, 1].set(
        title="Angular acceleration",
        xlabel="Time [s]",
        ylabel="Acceleration [rad/s²]",
    )

    axes[2, 0].plot(
        cold_control.times_s,
        cold_control.states[:, 1],
        label="cold",
    )
    axes[2, 0].plot(
        hot_control.times_s,
        hot_control.states[:, 1],
        label="hot",
    )
    axes[2, 0].axhline(
        diagnostics.control_reference_rad_s,
        color="black",
        linestyle="--",
        linewidth=1,
        label="reference",
    )
    axes[2, 0].set(
        title="Diagnostic PI tracking",
        xlabel="Time [s]",
        ylabel="Speed [rad/s]",
    )

    axes[2, 1].plot(
        cold_control.times_s[:-1],
        cold_control.voltages_v,
        label="cold",
    )
    axes[2, 1].plot(
        hot_control.times_s[:-1],
        hot_control.voltages_v,
        label="hot",
    )
    axes[2, 1].set(
        title="Diagnostic PI voltage",
        xlabel="Time [s]",
        ylabel="Voltage [V]",
    )

    history = np.asarray([result.history_s for result in report.probe_results])
    history_mae = np.asarray(
        [result.history_mae_c for result in report.probe_results]
    )
    instant_mae = np.asarray(
        [result.instantaneous_mae_c for result in report.probe_results]
    )
    mean_mae = np.asarray(
        [result.mean_baseline_mae_c for result in report.probe_results]
    )
    axes[3, 0].plot(history, history_mae, marker="o", label="history probe")
    axes[3, 0].plot(history, instant_mae, linestyle="--", label="instantaneous")
    axes[3, 0].plot(history, mean_mae, linestyle=":", label="train mean")
    axes[3, 0].set_xscale("log")
    axes[3, 0].set_yscale("log")
    axes[3, 0].set(
        title="Held-out temperature MAE",
        xlabel="History [s]",
        ylabel="MAE [C]",
    )

    true = diagnostics.selected_true_temperature_c
    predicted = diagnostics.selected_predicted_temperature_c
    low = float(min(np.min(true), np.min(predicted)))
    high = float(max(np.max(true), np.max(predicted)))
    axes[3, 1].scatter(true, predicted, s=14, alpha=0.65)
    axes[3, 1].plot([low, high], [low, high], color="black", linestyle="--")
    axes[3, 1].set(
        title=f"Temperature probe ({report.selected_history_s:g} s)",
        xlabel="True temperature [C]",
        ylabel="Predicted temperature [C]",
    )

    for axis in axes.flat:
        axis.grid(alpha=0.3)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend()
    figure.suptitle(
        "Phase 3: hidden-temperature observability and diagnostic identifiability"
    )
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _relative_difference(reference: float, comparison: float) -> float:
    return abs(float(comparison) - float(reference)) / max(abs(float(reference)), 1e-16)


def target_count(targets: FloatArray) -> int:
    """Return an ordinary integer for JSON-friendly sample counts."""

    return int(np.asarray(targets).shape[0])
