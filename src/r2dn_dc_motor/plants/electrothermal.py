"""Complete electrothermal armature-controlled DC-motor plant."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from r2dn_dc_motor.numerics import rk4_step

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class MotorState:
    """FULL state in the frozen order ``[current, speed, temperature]``."""

    current_a: float
    speed_rad_s: float
    temperature_c: float

    def as_array(self) -> FloatArray:
        return np.asarray(
            (self.current_a, self.speed_rad_s, self.temperature_c),
            dtype=np.float64,
        )

    @classmethod
    def from_array(cls, value: FloatArray) -> MotorState:
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (3,):
            raise ValueError("motor state must have shape (3,)")
        return cls(
            current_a=float(array[0]),
            speed_rad_s=float(array[1]),
            temperature_c=float(array[2]),
        )


@dataclass(frozen=True)
class MotorParameters:
    """Electrical, mechanical, and single-node thermal parameters."""

    armature_inductance_h: float
    reference_resistance_ohm: float
    reference_temperature_c: float
    resistance_temperature_coefficient_per_c: float
    back_emf_constant_v_s_per_rad: float
    torque_constant_n_m_per_a: float
    inertia_kg_m2: float
    viscous_friction_n_m_s_per_rad: float
    default_load_torque_n_m: float
    thermal_capacitance_j_per_c: float
    thermal_resistance_c_per_w: float
    ambient_temperature_c: float

    def with_temperature_coefficient(self, alpha_per_c: float) -> MotorParameters:
        """Return a copy used by the ``alpha = 0`` ablation."""

        return replace(
            self,
            resistance_temperature_coefficient_per_c=float(alpha_per_c),
        )


@dataclass(frozen=True)
class MotorLimits:
    """Plant domain and actuator limits."""

    minimum_voltage_v: float
    maximum_voltage_v: float
    minimum_current_a: float
    maximum_current_a: float
    minimum_speed_rad_s: float
    maximum_speed_rad_s: float
    minimum_temperature_c: float
    maximum_temperature_c: float


@dataclass(frozen=True)
class IntegrationSettings:
    """Two-rate integration settings."""

    control_period_s: float
    integrator_step_s: float

    @property
    def substeps_per_control(self) -> int:
        ratio = self.control_period_s / self.integrator_step_s
        rounded = int(round(ratio))
        if (
            self.control_period_s <= 0.0
            or self.integrator_step_s <= 0.0
            or not math.isclose(ratio, rounded, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError(
                "control period must contain a positive integer number of RK4 substeps"
            )
        return rounded


@dataclass(frozen=True)
class ResetRanges:
    """Uniform ranges for reproducible randomized plant resets."""

    current_a: tuple[float, float]
    speed_rad_s: tuple[float, float]
    temperature_c: tuple[float, float]


@dataclass(frozen=True)
class ThermalPowerBalance:
    """Instantaneous winding heat-flow terms."""

    copper_loss_w: float
    cooling_w: float
    net_heating_w: float


class TerminationReason(StrEnum):
    """Why a physical rollout left its allowed domain."""

    NONFINITE_STATE = "nonfinite_state"
    CURRENT_BELOW_MINIMUM = "current_below_minimum"
    CURRENT_ABOVE_MAXIMUM = "current_above_maximum"
    SPEED_BELOW_MINIMUM = "speed_below_minimum"
    SPEED_ABOVE_MAXIMUM = "speed_above_maximum"
    TEMPERATURE_BELOW_MINIMUM = "temperature_below_minimum"
    TEMPERATURE_ABOVE_MAXIMUM = "temperature_above_maximum"


@dataclass(frozen=True)
class StepResult:
    """One control-period plant transition."""

    state: MotorState
    commanded_voltage_v: float
    applied_voltage_v: float
    load_torque_n_m: float
    voltage_saturated: bool
    terminated: bool
    termination_reason: TerminationReason | None


@dataclass(frozen=True)
class Rollout:
    """A complete or early-terminated piecewise-constant-input rollout."""

    times_s: FloatArray
    states: FloatArray
    commanded_voltages_v: FloatArray
    applied_voltages_v: FloatArray
    load_torques_n_m: FloatArray
    terminated: bool
    termination_reason: TerminationReason | None


class ElectrothermalDCMotor:
    """FULL DC-motor simulator with temperature-dependent armature resistance.

    Voltage is clipped to the actuator range. Physical states are never clipped:
    crossing the current, speed, or temperature boundary terminates the rollout
    and preserves the violating state for diagnosis.
    """

    def __init__(
        self,
        parameters: MotorParameters,
        limits: MotorLimits,
        integration: IntegrationSettings,
        *,
        default_state: MotorState | None = None,
        reset_ranges: ResetRanges | None = None,
    ) -> None:
        self.parameters = parameters
        self.limits = limits
        self.integration = integration
        self.default_state = default_state or MotorState(
            0.0,
            0.0,
            parameters.ambient_temperature_c,
        )
        self.reset_ranges = reset_ranges
        _ = self.integration.substeps_per_control
        self._validate_static_configuration()
        self._state = self.default_state
        self._time_s = 0.0
        self._termination_reason: TerminationReason | None = None
        self.reset()

    @property
    def state(self) -> MotorState:
        return self._state

    @property
    def time_s(self) -> float:
        return self._time_s

    @property
    def termination_reason(self) -> TerminationReason | None:
        return self._termination_reason

    @property
    def terminated(self) -> bool:
        return self._termination_reason is not None

    def reset(
        self,
        initial_state: MotorState | None = None,
        *,
        seed: int | None = None,
        randomize: bool = False,
    ) -> MotorState:
        """Reset deterministically, optionally sampling configured uniform ranges."""

        if initial_state is not None and randomize:
            raise ValueError("provide an initial state or request randomization, not both")

        if randomize:
            if self.reset_ranges is None:
                raise ValueError("randomized reset requested without reset ranges")
            rng = np.random.default_rng(seed)
            ranges = self.reset_ranges
            initial_state = MotorState(
                current_a=float(rng.uniform(*ranges.current_a)),
                speed_rad_s=float(rng.uniform(*ranges.speed_rad_s)),
                temperature_c=float(rng.uniform(*ranges.temperature_c)),
            )
        else:
            initial_state = initial_state or self.default_state

        reason = self.limit_violation(initial_state)
        if reason is not None:
            raise ValueError(f"initial state violates plant domain: {reason.value}")

        self._state = initial_state
        self._time_s = 0.0
        self._termination_reason = None
        return self._state

    def resistance_ohm(self, temperature_c: float) -> float:
        """Linear winding-resistance law around the declared reference point."""

        parameters = self.parameters
        resistance = parameters.reference_resistance_ohm * (
            1.0
            + parameters.resistance_temperature_coefficient_per_c
            * (float(temperature_c) - parameters.reference_temperature_c)
        )
        if resistance <= 0.0 or not math.isfinite(resistance):
            raise ValueError("temperature produced non-positive or non-finite resistance")
        return resistance

    def derivative(
        self,
        state: FloatArray,
        applied_voltage_v: float,
        load_torque_n_m: float | None = None,
    ) -> FloatArray:
        """Evaluate ``[di/dt, dω/dt, dT/dt]`` without mutating the plant."""

        current_a, speed_rad_s, temperature_c = np.asarray(state, dtype=np.float64)
        p = self.parameters
        load = p.default_load_torque_n_m if load_torque_n_m is None else load_torque_n_m
        resistance = self.resistance_ohm(float(temperature_c))

        current_rate = (
            applied_voltage_v
            - resistance * current_a
            - p.back_emf_constant_v_s_per_rad * speed_rad_s
        ) / p.armature_inductance_h
        speed_rate = (
            p.torque_constant_n_m_per_a * current_a
            - p.viscous_friction_n_m_s_per_rad * speed_rad_s
            - load
        ) / p.inertia_kg_m2
        power = self.thermal_power_balance(float(current_a), float(temperature_c))
        temperature_rate = power.net_heating_w / p.thermal_capacitance_j_per_c
        return np.asarray((current_rate, speed_rate, temperature_rate), dtype=np.float64)

    def thermal_power_balance(
        self,
        current_a: float,
        temperature_c: float,
    ) -> ThermalPowerBalance:
        """Return copper loss, ambient cooling, and net winding heating."""

        p = self.parameters
        copper_loss = self.resistance_ohm(temperature_c) * float(current_a) ** 2
        cooling = (float(temperature_c) - p.ambient_temperature_c) / (
            p.thermal_resistance_c_per_w
        )
        return ThermalPowerBalance(
            copper_loss_w=copper_loss,
            cooling_w=cooling,
            net_heating_w=copper_loss - cooling,
        )

    def constant_current_thermal_equilibrium_c(self, current_a: float) -> float:
        """Analytic one-node equilibrium for an externally held current."""

        p = self.parameters
        current_squared = float(current_a) ** 2
        conductance = 1.0 / p.thermal_resistance_c_per_w
        temperature_gain = (
            p.reference_resistance_ohm
            * p.resistance_temperature_coefficient_per_c
            * current_squared
        )
        denominator = conductance - temperature_gain
        if denominator <= 0.0:
            raise ValueError("constant-current thermal equilibrium is not stable")

        numerator = (
            p.reference_resistance_ohm
            * current_squared
            * (
                1.0
                - p.resistance_temperature_coefficient_per_c
                * p.reference_temperature_c
            )
            + conductance * p.ambient_temperature_c
        )
        return numerator / denominator

    def time_constants_s(self) -> dict[str, float]:
        """Return nominal electrical, reduced mechanical, and thermal constants."""

        p = self.parameters
        resistance = self.resistance_ohm(p.reference_temperature_c)
        electrical = p.armature_inductance_h / resistance
        effective_damping = (
            p.viscous_friction_n_m_s_per_rad
            + p.torque_constant_n_m_per_a
            * p.back_emf_constant_v_s_per_rad
            / resistance
        )
        mechanical = p.inertia_kg_m2 / effective_damping
        thermal = p.thermal_capacitance_j_per_c * p.thermal_resistance_c_per_w
        return {
            "electrical": electrical,
            "mechanical": mechanical,
            "thermal": thermal,
        }

    def step(
        self,
        commanded_voltage_v: float,
        *,
        load_torque_n_m: float | None = None,
    ) -> StepResult:
        """Advance exactly one control period with zero-order-held input."""

        if self.terminated:
            raise RuntimeError(
                f"cannot step a terminated plant: {self._termination_reason.value}"
            )
        if not math.isfinite(commanded_voltage_v):
            raise ValueError("commanded voltage must be finite")

        load = (
            self.parameters.default_load_torque_n_m
            if load_torque_n_m is None
            else float(load_torque_n_m)
        )
        if not math.isfinite(load):
            raise ValueError("load torque must be finite")

        limits = self.limits
        applied_voltage = float(
            np.clip(
                commanded_voltage_v,
                limits.minimum_voltage_v,
                limits.maximum_voltage_v,
            )
        )
        voltage_saturated = not math.isclose(
            commanded_voltage_v,
            applied_voltage,
            rel_tol=0.0,
            abs_tol=0.0,
        )

        state = self._state.as_array()
        step_s = self.integration.integrator_step_s
        for _ in range(self.integration.substeps_per_control):
            rhs = lambda time_s, value: self.derivative(  # noqa: E731
                value,
                applied_voltage,
                load,
            )
            state = rk4_step(rhs, self._time_s, state, step_s)
            self._time_s += step_s
            self._state = MotorState.from_array(state)
            self._termination_reason = self.limit_violation(self._state)
            if self.terminated:
                break

        return StepResult(
            state=self._state,
            commanded_voltage_v=float(commanded_voltage_v),
            applied_voltage_v=applied_voltage,
            load_torque_n_m=load,
            voltage_saturated=voltage_saturated,
            terminated=self.terminated,
            termination_reason=self._termination_reason,
        )

    def rollout(
        self,
        commanded_voltages_v: FloatArray,
        *,
        initial_state: MotorState | None = None,
        load_torques_n_m: FloatArray | float | None = None,
        seed: int | None = None,
        randomize_reset: bool = False,
    ) -> Rollout:
        """Simulate a voltage sequence and stop on the first domain violation."""

        voltages = np.asarray(commanded_voltages_v, dtype=np.float64)
        if voltages.ndim != 1 or not np.isfinite(voltages).all():
            raise ValueError("commanded voltages must be a finite one-dimensional array")

        if load_torques_n_m is None:
            loads = np.full_like(voltages, self.parameters.default_load_torque_n_m)
        elif np.isscalar(load_torques_n_m):
            loads = np.full_like(voltages, float(load_torques_n_m))
        else:
            loads = np.asarray(load_torques_n_m, dtype=np.float64)
            if loads.shape != voltages.shape or not np.isfinite(loads).all():
                raise ValueError("load torque sequence must match the voltage sequence")

        self.reset(initial_state, seed=seed, randomize=randomize_reset)
        times = [0.0]
        states = [self.state.as_array()]
        commanded: list[float] = []
        applied: list[float] = []
        used_loads: list[float] = []

        for voltage, load in zip(voltages, loads, strict=True):
            result = self.step(float(voltage), load_torque_n_m=float(load))
            times.append(self.time_s)
            states.append(result.state.as_array())
            commanded.append(result.commanded_voltage_v)
            applied.append(result.applied_voltage_v)
            used_loads.append(result.load_torque_n_m)
            if result.terminated:
                break

        return Rollout(
            times_s=np.asarray(times, dtype=np.float64),
            states=np.asarray(states, dtype=np.float64),
            commanded_voltages_v=np.asarray(commanded, dtype=np.float64),
            applied_voltages_v=np.asarray(applied, dtype=np.float64),
            load_torques_n_m=np.asarray(used_loads, dtype=np.float64),
            terminated=self.terminated,
            termination_reason=self._termination_reason,
        )

    def limit_violation(self, state: MotorState) -> TerminationReason | None:
        values = state.as_array()
        if not np.isfinite(values).all():
            return TerminationReason.NONFINITE_STATE

        limits = self.limits
        if state.current_a < limits.minimum_current_a:
            return TerminationReason.CURRENT_BELOW_MINIMUM
        if state.current_a > limits.maximum_current_a:
            return TerminationReason.CURRENT_ABOVE_MAXIMUM
        if state.speed_rad_s < limits.minimum_speed_rad_s:
            return TerminationReason.SPEED_BELOW_MINIMUM
        if state.speed_rad_s > limits.maximum_speed_rad_s:
            return TerminationReason.SPEED_ABOVE_MAXIMUM
        if state.temperature_c < limits.minimum_temperature_c:
            return TerminationReason.TEMPERATURE_BELOW_MINIMUM
        if state.temperature_c > limits.maximum_temperature_c:
            return TerminationReason.TEMPERATURE_ABOVE_MAXIMUM
        return None

    def _validate_static_configuration(self) -> None:
        p = self.parameters
        positive = {
            "armature inductance": p.armature_inductance_h,
            "reference resistance": p.reference_resistance_ohm,
            "back-EMF constant": p.back_emf_constant_v_s_per_rad,
            "torque constant": p.torque_constant_n_m_per_a,
            "inertia": p.inertia_kg_m2,
            "thermal capacitance": p.thermal_capacitance_j_per_c,
            "thermal resistance": p.thermal_resistance_c_per_w,
        }
        invalid = [name for name, value in positive.items() if value <= 0.0]
        if invalid:
            raise ValueError(f"motor parameters must be positive: {', '.join(invalid)}")
        if p.viscous_friction_n_m_s_per_rad < 0.0:
            raise ValueError("viscous friction must be non-negative")
        if p.resistance_temperature_coefficient_per_c < 0.0:
            raise ValueError("temperature coefficient must be non-negative")

        limits = self.limits
        ranges = (
            (limits.minimum_voltage_v, limits.maximum_voltage_v, "voltage"),
            (limits.minimum_current_a, limits.maximum_current_a, "current"),
            (limits.minimum_speed_rad_s, limits.maximum_speed_rad_s, "speed"),
            (
                limits.minimum_temperature_c,
                limits.maximum_temperature_c,
                "temperature",
            ),
        )
        invalid_ranges = [name for low, high, name in ranges if low >= high]
        if invalid_ranges:
            raise ValueError(f"invalid plant limits: {', '.join(invalid_ranges)}")
