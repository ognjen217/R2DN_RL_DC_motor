"""Two-state isothermal DC-motor baselines with a world-model interface."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from r2dn_dc_motor.plants.electrothermal import IntegrationSettings, MotorLimits

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class IsothermalParameters:
    """Parameters of the intentionally incomplete isothermal motor."""

    armature_inductance_h: float
    effective_resistance_ohm: float
    back_emf_constant_v_s_per_rad: float
    torque_constant_n_m_per_a: float
    inertia_kg_m2: float
    viscous_friction_n_m_s_per_rad: float
    default_load_torque_n_m: float

    def validate(self) -> None:
        values = asdict(self)
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("isothermal parameters must be finite")
        positive = (
            "armature_inductance_h",
            "effective_resistance_ohm",
            "back_emf_constant_v_s_per_rad",
            "torque_constant_n_m_per_a",
            "inertia_kg_m2",
            "viscous_friction_n_m_s_per_rad",
        )
        for name in positive:
            if values[name] <= 0.0:
                raise ValueError(f"{name} must be positive")


class IsothermalWorldModel:
    """ISO-NOM/ISO-CAL backend over observations ``[i, omega]`` and voltage.

    The applied armature voltage is the only runtime input. Load torque is not
    read from a dataset trajectory; the model always uses its single nominal
    load parameter, preserving the frozen temperature-free world-model
    interface and leaving load shifts as a genuine OOD disturbance.
    """

    observation_names = ("armature_current_a", "angular_speed_rad_s")
    control_names = ("armature_voltage_v",)

    def __init__(
        self,
        parameters: IsothermalParameters,
        limits: MotorLimits,
        integration: IntegrationSettings,
        *,
        name: str,
    ) -> None:
        if name not in {"ISO-NOM", "ISO-CAL"}:
            raise ValueError("isothermal model name must be ISO-NOM or ISO-CAL")
        parameters.validate()
        _ = integration.substeps_per_control
        self.parameters = parameters
        self.limits = limits
        self.integration = integration
        self.name = name
        self._transition_matrix = self._build_transition_matrix()

    def derivative(
        self,
        observation: FloatArray,
        armature_voltage_v: float,
    ) -> FloatArray:
        state = _observation(observation)
        p = self.parameters
        current_rate = (
            float(armature_voltage_v)
            - p.effective_resistance_ohm * state[0]
            - p.back_emf_constant_v_s_per_rad * state[1]
        ) / p.armature_inductance_h
        speed_rate = (
            p.torque_constant_n_m_per_a * state[0]
            - p.viscous_friction_n_m_s_per_rad * state[1]
            - p.default_load_torque_n_m
        ) / p.inertia_kg_m2
        return np.asarray((current_rate, speed_rate), dtype=np.float64)

    def predict_next(
        self,
        observation: FloatArray,
        armature_voltage_v: float | FloatArray,
    ) -> FloatArray:
        """Predict one control-period transition with the Phase-2 RK4 scheme."""

        state = _observation(observation)
        voltage = float(np.asarray(armature_voltage_v, dtype=np.float64).reshape(-1)[0])
        voltage = float(
            np.clip(
                voltage,
                self.limits.minimum_voltage_v,
                self.limits.maximum_voltage_v,
            )
        )
        augmented = np.asarray((state[0], state[1], voltage, 1.0), dtype=np.float64)
        return (self._transition_matrix @ augmented)[:2]

    def predict_next_batch(
        self,
        observations: FloatArray,
        armature_voltages_v: FloatArray,
    ) -> FloatArray:
        """Vectorized teacher-forced one-step predictions."""

        states = np.asarray(observations, dtype=np.float64)
        controls = np.asarray(armature_voltages_v, dtype=np.float64)
        if states.ndim != 2 or states.shape[1] != 2:
            raise ValueError("observations must have shape (N, 2)")
        if controls.ndim == 1:
            controls = controls[:, None]
        if controls.shape != (states.shape[0], 1):
            raise ValueError("controls must have shape (N, 1)")
        if not np.isfinite(states).all() or not np.isfinite(controls).all():
            raise ValueError("batch contains NaN or infinite values")
        clipped = np.clip(
            controls[:, 0],
            self.limits.minimum_voltage_v,
            self.limits.maximum_voltage_v,
        )
        augmented = np.column_stack(
            (states, clipped, np.ones(states.shape[0], dtype=np.float64))
        )
        return augmented @ self._transition_matrix[:2, :].T

    def free_rollout(
        self,
        initial_observation: FloatArray,
        armature_voltages_v: FloatArray,
    ) -> FloatArray:
        """Autoregressive rollout with no teacher forcing after initialization."""

        controls = np.asarray(armature_voltages_v, dtype=np.float64)
        if controls.ndim == 1:
            controls = controls[:, None]
        if controls.ndim != 2 or controls.shape[1] != 1:
            raise ValueError("controls must have shape (T, 1)")
        if not np.isfinite(controls).all():
            raise ValueError("controls contain NaN or infinite values")
        states = np.empty((controls.shape[0] + 1, 2), dtype=np.float64)
        states[0] = _observation(initial_observation)
        for index, voltage in enumerate(controls[:, 0]):
            states[index + 1] = self.predict_next(states[index], voltage)
        return states

    def _build_transition_matrix(self) -> FloatArray:
        """Precompute the repeated RK4 affine map for one control period."""

        p = self.parameters
        generator = np.zeros((4, 4), dtype=np.float64)
        generator[0, 0] = -p.effective_resistance_ohm / p.armature_inductance_h
        generator[0, 1] = (
            -p.back_emf_constant_v_s_per_rad / p.armature_inductance_h
        )
        generator[0, 2] = 1.0 / p.armature_inductance_h
        generator[1, 0] = p.torque_constant_n_m_per_a / p.inertia_kg_m2
        generator[1, 1] = (
            -p.viscous_friction_n_m_s_per_rad / p.inertia_kg_m2
        )
        generator[1, 3] = (
            -p.default_load_torque_n_m / p.inertia_kg_m2
        )

        scaled = self.integration.integrator_step_s * generator
        identity = np.eye(4, dtype=np.float64)
        rk4_substep = (
            identity
            + scaled
            + scaled @ scaled / 2.0
            + scaled @ scaled @ scaled / 6.0
            + scaled @ scaled @ scaled @ scaled / 24.0
        )
        return np.linalg.matrix_power(
            rk4_substep,
            self.integration.substeps_per_control,
        )


def _observation(value: FloatArray) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,):
        raise ValueError("observation must have shape (2,)")
    if not np.isfinite(array).all():
        raise ValueError("observation contains NaN or infinite values")
    return array
