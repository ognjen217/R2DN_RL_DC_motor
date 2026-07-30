"""Gate-0 physical and numerical validation for the FULL motor."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from r2dn_dc_motor.phase2_spec import Phase2Spec, load_phase2_spec
from r2dn_dc_motor.plants import IntegrationSettings, MotorState


@dataclass(frozen=True)
class ValidationCheck:
    """One named Gate-0 check and its diagnostic metrics."""

    name: str
    passed: bool
    metrics: dict[str, float | str | bool]
    criterion: str


@dataclass(frozen=True)
class Phase2ValidationReport:
    """Complete Gate-0 result."""

    passed: bool
    checks: tuple[ValidationCheck, ...]
    time_constants_s: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "phase": 2,
            "gate": "gate_0_physical_plausibility",
            "passed": self.passed,
            "time_constants_s": self.time_constants_s,
            "checks": [asdict(check) for check in self.checks],
        }

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"PHASE 2 GATE 0: {status}"]
        for check in self.checks:
            marker = "PASS" if check.passed else "FAIL"
            lines.append(f"[{marker}] {check.name}")
        constants = self.time_constants_s
        lines.append(
            "time constants [s]: "
            f"tau_e={constants['electrical']:.6g}, "
            f"tau_m={constants['mechanical']:.6g}, "
            f"tau_th={constants['thermal']:.6g}"
        )
        return "\n".join(lines)


def run_phase2_validation(spec: Phase2Spec | None = None) -> Phase2ValidationReport:
    """Execute all eight mandatory physical/numerical checks."""

    spec = spec or load_phase2_spec()
    settings = spec.validation
    checks = (
        _check_zero_input_decay(spec),
        _check_current_heating(spec),
        _check_zero_current_cooling(spec),
        _check_resistance_monotonicity(spec),
        _check_hot_current(spec),
        _check_rk4_convergence(spec),
        _check_alpha_zero(spec),
        _check_thermal_power_balance(spec),
    )
    required_names = set(spec.gate["required_checks"])
    actual_names = {check.name for check in checks}
    all_passed = (
        actual_names == required_names
        and all(check.passed for check in checks)
        and float(settings["drive_voltage_v"]) > 0.0
    )
    return Phase2ValidationReport(
        passed=all_passed,
        checks=checks,
        time_constants_s=spec.time_constants_s(),
    )


def generate_phase2_artifacts(
    output_directory: Path | str,
    *,
    spec: Phase2Spec | None = None,
) -> tuple[Path, Path]:
    """Write the validation JSON and a basic-response figure."""

    spec = spec or load_phase2_spec()
    report = run_phase2_validation(spec)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    report_path = output / "phase2_validation.json"
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    figure_path = output / "phase2_basic_responses.png"
    _plot_basic_responses(spec, figure_path)
    return report_path, figure_path


def _check_zero_input_decay(spec: Phase2Spec) -> ValidationCheck:
    plant = spec.build_plant()
    duration = float(spec.validation["zero_input_duration_s"])
    steps = _control_steps(spec, duration)
    initial = MotorState(current_a=5.0, speed_rad_s=100.0, temperature_c=40.0)
    rollout = plant.rollout(
        np.zeros(steps),
        initial_state=initial,
        load_torques_n_m=0.0,
    )
    final = MotorState.from_array(rollout.states[-1])
    current_tolerance = float(spec.validation["zero_input_current_tolerance_a"])
    speed_tolerance = float(spec.validation["zero_input_speed_tolerance_rad_s"])
    passed = (
        not rollout.terminated
        and abs(final.current_a) <= current_tolerance
        and abs(final.speed_rad_s) <= speed_tolerance
        and final.temperature_c < initial.temperature_c
    )
    return ValidationCheck(
        name="zero_input_returns_to_rest",
        passed=passed,
        metrics={
            "final_current_a": final.current_a,
            "final_speed_rad_s": final.speed_rad_s,
            "initial_temperature_c": initial.temperature_c,
            "final_temperature_c": final.temperature_c,
        },
        criterion=(
            f"|i| <= {current_tolerance:g} A, "
            f"|omega| <= {speed_tolerance:g} rad/s, and winding cools"
        ),
    )


def _check_current_heating(spec: Phase2Spec) -> ValidationCheck:
    plant = spec.build_plant()
    current = float(spec.validation["validation_current_a"])
    ambient = plant.parameters.ambient_temperature_c
    equilibrium = plant.constant_current_thermal_equilibrium_c(current)
    midpoint = (ambient + equilibrium) / 2.0
    above = equilibrium + 0.1 * (equilibrium - ambient)
    power_at_ambient = plant.thermal_power_balance(current, ambient).net_heating_w
    power_below = plant.thermal_power_balance(current, midpoint).net_heating_w
    power_above = plant.thermal_power_balance(current, above).net_heating_w
    passed = (
        equilibrium > ambient
        and power_at_ambient > 0.0
        and power_below > 0.0
        and power_above < 0.0
    )
    return ValidationCheck(
        name="current_heats_toward_equilibrium",
        passed=passed,
        metrics={
            "current_a": current,
            "ambient_temperature_c": ambient,
            "equilibrium_temperature_c": equilibrium,
            "net_power_at_ambient_w": power_at_ambient,
            "net_power_below_equilibrium_w": power_below,
            "net_power_above_equilibrium_w": power_above,
        },
        criterion="net heat is positive below and negative above the stable equilibrium",
    )


def _check_zero_current_cooling(spec: Phase2Spec) -> ValidationCheck:
    plant = spec.build_plant()
    duration = float(spec.validation["cooling_duration_s"])
    steps = _control_steps(spec, duration)
    initial_temperature = float(spec.validation["hot_temperature_c"])
    initial = MotorState(0.0, 0.0, initial_temperature)
    rollout = plant.rollout(
        np.zeros(steps),
        initial_state=initial,
        load_torques_n_m=0.0,
    )
    actual = float(rollout.states[-1, 2])
    p = plant.parameters
    expected = p.ambient_temperature_c + (
        initial_temperature - p.ambient_temperature_c
    ) * math.exp(
        -duration / (p.thermal_capacitance_j_per_c * p.thermal_resistance_c_per_w)
    )
    error = abs(actual - expected)
    passed = (
        not rollout.terminated
        and p.ambient_temperature_c < actual < initial_temperature
        and error < 1e-9
    )
    return ValidationCheck(
        name="zero_current_cools_to_ambient",
        passed=passed,
        metrics={
            "initial_temperature_c": initial_temperature,
            "actual_temperature_c": actual,
            "analytic_temperature_c": expected,
            "absolute_error_c": error,
        },
        criterion="zero-current RK4 cooling matches the analytic exponential within 1e-9 C",
    )


def _check_resistance_monotonicity(spec: Phase2Spec) -> ValidationCheck:
    plant = spec.build_plant()
    low_temperature = plant.parameters.ambient_temperature_c
    high_temperature = float(spec.validation["hot_temperature_c"])
    low_resistance = plant.resistance_ohm(low_temperature)
    high_resistance = plant.resistance_ohm(high_temperature)
    passed = high_resistance > low_resistance > 0.0
    return ValidationCheck(
        name="resistance_increases_with_temperature",
        passed=passed,
        metrics={
            "low_temperature_c": low_temperature,
            "high_temperature_c": high_temperature,
            "low_resistance_ohm": low_resistance,
            "high_resistance_ohm": high_resistance,
            "relative_increase": high_resistance / low_resistance - 1.0,
        },
        criterion="R(T_hot) > R(T_ambient) > 0",
    )


def _check_hot_current(spec: Phase2Spec) -> ValidationCheck:
    duration = float(spec.validation["hot_cold_duration_s"])
    steps = _control_steps(spec, duration)
    voltage = float(spec.validation["drive_voltage_v"])
    ambient = spec.motor_parameters.ambient_temperature_c
    hot = float(spec.validation["hot_temperature_c"])
    inputs = np.full(steps, voltage)

    cold_plant = spec.build_plant()
    hot_plant = spec.build_plant()
    cold = cold_plant.rollout(
        inputs,
        initial_state=MotorState(0.0, 0.0, ambient),
        load_torques_n_m=0.0,
    )
    heated = hot_plant.rollout(
        inputs,
        initial_state=MotorState(0.0, 0.0, hot),
        load_torques_n_m=0.0,
    )
    cold_current = float(cold.states[-1, 0])
    hot_current = float(heated.states[-1, 0])
    passed = (
        not cold.terminated
        and not heated.terminated
        and hot_current < cold_current
        and cold_current - hot_current > 1e-3
    )
    return ValidationCheck(
        name="hot_motor_draws_less_current",
        passed=passed,
        metrics={
            "comparison_time_s": duration,
            "cold_current_a": cold_current,
            "hot_current_a": hot_current,
            "current_reduction_a": cold_current - hot_current,
        },
        criterion="same voltage and mechanical state produce measurably lower hot current",
    )


def _check_rk4_convergence(spec: Phase2Spec) -> ValidationCheck:
    duration = float(spec.validation["convergence_duration_s"])
    voltage = float(spec.validation["drive_voltage_v"])
    control_period = spec.integration_settings.control_period_s
    coarse_step = spec.integration_settings.integrator_step_s
    inputs = np.full(_control_steps(spec, duration), voltage)
    initial = MotorState(1.0, 10.0, 40.0)

    coarse = spec.build_plant().rollout(inputs, initial_state=initial)
    fine_settings = IntegrationSettings(
        control_period_s=control_period,
        integrator_step_s=coarse_step / 2.0,
    )
    fine = spec.build_plant(integration=fine_settings).rollout(
        inputs,
        initial_state=initial,
    )
    difference = np.abs(coarse.states[-1] - fine.states[-1])
    scales = np.asarray(
        (
            spec.motor_limits.maximum_current_a,
            spec.motor_limits.maximum_speed_rad_s,
            spec.motor_limits.maximum_temperature_c,
        ),
        dtype=np.float64,
    )
    scaled_error = float(np.max(difference / scales))
    tolerance = float(spec.validation["convergence_scaled_tolerance"])
    passed = not coarse.terminated and not fine.terminated and scaled_error <= tolerance
    return ValidationCheck(
        name="rk4_step_converges",
        passed=passed,
        metrics={
            "coarse_step_s": coarse_step,
            "fine_step_s": coarse_step / 2.0,
            "maximum_scaled_final_state_error": scaled_error,
            "current_difference_a": float(difference[0]),
            "speed_difference_rad_s": float(difference[1]),
            "temperature_difference_c": float(difference[2]),
        },
        criterion=f"scaled final-state difference <= {tolerance:g}",
    )


def _check_alpha_zero(spec: Phase2Spec) -> ValidationCheck:
    parameters = spec.motor_parameters.with_temperature_coefficient(0.0)
    duration = float(spec.validation["convergence_duration_s"])
    inputs = np.full(
        _control_steps(spec, duration),
        float(spec.validation["drive_voltage_v"]),
    )
    ambient = parameters.ambient_temperature_c
    hot = float(spec.validation["hot_temperature_c"])
    cold_rollout = spec.build_plant(parameters=parameters).rollout(
        inputs,
        initial_state=MotorState(0.0, 0.0, ambient),
    )
    hot_rollout = spec.build_plant(parameters=parameters).rollout(
        inputs,
        initial_state=MotorState(0.0, 0.0, hot),
    )
    maximum_output_difference = float(
        np.max(np.abs(cold_rollout.states[:, :2] - hot_rollout.states[:, :2]))
    )
    tolerance = float(spec.validation["alpha_zero_output_tolerance"])
    passed = (
        not cold_rollout.terminated
        and not hot_rollout.terminated
        and maximum_output_difference <= tolerance
    )
    return ValidationCheck(
        name="alpha_zero_is_isothermal",
        passed=passed,
        metrics={
            "maximum_current_speed_difference": maximum_output_difference,
            "cold_final_temperature_c": float(cold_rollout.states[-1, 2]),
            "hot_final_temperature_c": float(hot_rollout.states[-1, 2]),
        },
        criterion=f"alpha=0 makes [i, omega] temperature-independent within {tolerance:g}",
    )


def _check_thermal_power_balance(spec: Phase2Spec) -> ValidationCheck:
    plant = spec.build_plant()
    current = float(spec.validation["validation_current_a"])
    equilibrium = plant.constant_current_thermal_equilibrium_c(current)
    balance = plant.thermal_power_balance(current, equilibrium)
    derivative = plant.derivative(
        np.asarray((current, 0.0, equilibrium), dtype=np.float64),
        applied_voltage_v=0.0,
        load_torque_n_m=0.0,
    )
    reconstructed = balance.copper_loss_w - balance.cooling_w
    passed = (
        balance.copper_loss_w > 0.0
        and balance.cooling_w > 0.0
        and abs(balance.net_heating_w) < 1e-12
        and math.isclose(
            balance.net_heating_w,
            reconstructed,
            rel_tol=0.0,
            abs_tol=1e-14,
        )
        and abs(float(derivative[2])) < 1e-13
    )
    return ValidationCheck(
        name="thermal_power_balance_is_consistent",
        passed=passed,
        metrics={
            "equilibrium_temperature_c": equilibrium,
            "copper_loss_w": balance.copper_loss_w,
            "cooling_w": balance.cooling_w,
            "net_heating_w": balance.net_heating_w,
            "temperature_rate_c_per_s": float(derivative[2]),
        },
        criterion="P_copper - P_cooling equals Cth*dT/dt and is zero at equilibrium",
    )


def _plot_basic_responses(spec: Phase2Spec, output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "plot generation requires: python -m pip install -e '.[phase2]'"
        ) from error

    duration = float(spec.validation["plot_response_duration_s"])
    steps = _control_steps(spec, duration)
    voltage = float(spec.validation["drive_voltage_v"])
    ambient = spec.motor_parameters.ambient_temperature_c
    hot_temperature = float(spec.validation["hot_temperature_c"])
    inputs = np.full(steps, voltage)

    cold = spec.build_plant().rollout(
        inputs,
        initial_state=MotorState(0.0, 0.0, ambient),
    )
    hot = spec.build_plant().rollout(
        inputs,
        initial_state=MotorState(0.0, 0.0, hot_temperature),
    )

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes[0, 0].plot(cold.times_s, cold.states[:, 0], label=f"cold ({ambient:g} C)")
    axes[0, 0].plot(
        hot.times_s,
        hot.states[:, 0],
        label=f"hot ({hot_temperature:g} C)",
    )
    axes[0, 0].set(title="Armature current", xlabel="Time [s]", ylabel="Current [A]")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(cold.times_s, cold.states[:, 1], label="cold")
    axes[0, 1].plot(hot.times_s, hot.states[:, 1], label="hot")
    axes[0, 1].set(
        title="Angular speed",
        xlabel="Time [s]",
        ylabel="Speed [rad/s]",
    )
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    temperatures = np.linspace(
        spec.motor_limits.minimum_temperature_c,
        spec.motor_limits.maximum_temperature_c,
        300,
    )
    resistances = np.asarray(
        [spec.build_plant().resistance_ohm(value) for value in temperatures]
    )
    axes[1, 0].plot(temperatures, resistances)
    axes[1, 0].set(
        title="Temperature-dependent resistance",
        xlabel="Temperature [C]",
        ylabel="Resistance [ohm]",
    )
    axes[1, 0].grid(alpha=0.3)

    current = float(spec.validation["validation_current_a"])
    thermal_time = spec.time_constants_s()["thermal"]
    thermal_times = np.linspace(0.0, 5.0 * thermal_time, 500)
    equilibrium = spec.build_plant().constant_current_thermal_equilibrium_c(current)
    effective_rate = (
        1.0 / spec.motor_parameters.thermal_resistance_c_per_w
        - spec.motor_parameters.reference_resistance_ohm
        * spec.motor_parameters.resistance_temperature_coefficient_per_c
        * current**2
    ) / spec.motor_parameters.thermal_capacitance_j_per_c
    heating = equilibrium + (ambient - equilibrium) * np.exp(
        -effective_rate * thermal_times
    )
    cooling = ambient + (hot_temperature - ambient) * np.exp(
        -thermal_times / thermal_time
    )
    axes[1, 1].plot(thermal_times, heating, label=f"heating at {current:g} A")
    axes[1, 1].plot(thermal_times, cooling, label="cooling at 0 A")
    axes[1, 1].axhline(equilibrium, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set(
        title="Single-node thermal response",
        xlabel="Time [s]",
        ylabel="Temperature [C]",
    )
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    figure.suptitle(
        f"FULL motor validation: u={voltage:g} V, "
        f"load={spec.motor_parameters.default_load_torque_n_m:g} N m"
    )
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _control_steps(spec: Phase2Spec, duration_s: float) -> int:
    raw_steps = duration_s / spec.integration_settings.control_period_s
    steps = int(round(raw_steps))
    if not math.isclose(raw_steps, steps, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("validation duration must contain whole control periods")
    return steps
