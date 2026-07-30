from dataclasses import replace

import numpy as np
import pytest

from r2dn_dc_motor.phase2_spec import load_phase2_spec
from r2dn_dc_motor.plants import MotorState, TerminationReason


@pytest.fixture
def spec():
    return load_phase2_spec()


@pytest.fixture
def plant(spec):
    return spec.build_plant()


def test_resistance_law_matches_reference_and_increases(plant):
    parameters = plant.parameters
    reference = plant.resistance_ohm(parameters.reference_temperature_c)
    hot = plant.resistance_ohm(parameters.reference_temperature_c + 50.0)

    assert reference == pytest.approx(parameters.reference_resistance_ohm)
    assert hot == pytest.approx(
        reference
        * (
            1.0
            + 50.0 * parameters.resistance_temperature_coefficient_per_c
        )
    )
    assert hot > reference


def test_derivative_matches_electrothermal_equations(plant):
    state = np.asarray([4.0, 120.0, 60.0])
    voltage = 24.0
    load = 0.1
    derivative = plant.derivative(state, voltage, load)
    p = plant.parameters
    resistance = plant.resistance_ohm(60.0)

    expected_current_rate = (
        voltage - resistance * state[0] - p.back_emf_constant_v_s_per_rad * state[1]
    ) / p.armature_inductance_h
    expected_speed_rate = (
        p.torque_constant_n_m_per_a * state[0]
        - p.viscous_friction_n_m_s_per_rad * state[1]
        - load
    ) / p.inertia_kg_m2
    expected_temperature_rate = (
        resistance * state[0] ** 2
        - (state[2] - p.ambient_temperature_c) / p.thermal_resistance_c_per_w
    ) / p.thermal_capacitance_j_per_c

    np.testing.assert_allclose(
        derivative,
        [expected_current_rate, expected_speed_rate, expected_temperature_rate],
    )


def test_step_saturates_voltage_but_does_not_clip_state(plant):
    result = plant.step(100.0, load_torque_n_m=0.0)

    assert result.commanded_voltage_v == 100.0
    assert result.applied_voltage_v == plant.limits.maximum_voltage_v
    assert result.voltage_saturated is True


def test_current_limit_terminates_rollout_without_clipping(spec):
    plant = spec.build_plant()
    initial = MotorState(
        current_a=plant.limits.maximum_current_a - 0.01,
        speed_rad_s=0.0,
        temperature_c=plant.parameters.ambient_temperature_c,
    )
    plant.reset(initial)
    result = plant.step(plant.limits.maximum_voltage_v, load_torque_n_m=0.0)

    assert result.terminated is True
    assert result.termination_reason == TerminationReason.CURRENT_ABOVE_MAXIMUM
    assert result.state.current_a > plant.limits.maximum_current_a
    with pytest.raises(RuntimeError, match="terminated plant"):
        plant.step(0.0)


def test_randomized_reset_is_seed_reproducible(plant):
    first = plant.reset(seed=1234, randomize=True)
    second = plant.reset(seed=1234, randomize=True)
    third = plant.reset(seed=4321, randomize=True)

    assert first == second
    assert first != third
    assert plant.limit_violation(first) is None
    assert plant.limit_violation(third) is None


def test_rollout_shapes_follow_phase1_sequence_contract(plant):
    rollout = plant.rollout(
        np.asarray([0.0, 4.0, 8.0, 0.0]),
        load_torques_n_m=0.0,
    )

    assert rollout.states.shape == (5, 3)
    assert rollout.times_s.shape == (5,)
    assert rollout.commanded_voltages_v.shape == (4,)
    assert rollout.applied_voltages_v.shape == (4,)
    assert rollout.load_torques_n_m.shape == (4,)
    assert rollout.times_s[-1] == pytest.approx(4 * plant.integration.control_period_s)


def test_zero_current_cooling_matches_analytic_solution(plant):
    initial_temperature = 90.0
    duration = 1.0
    steps = round(duration / plant.integration.control_period_s)
    rollout = plant.rollout(
        np.zeros(steps),
        initial_state=MotorState(0.0, 0.0, initial_temperature),
        load_torques_n_m=0.0,
    )
    p = plant.parameters
    expected = p.ambient_temperature_c + (
        initial_temperature - p.ambient_temperature_c
    ) * np.exp(
        -duration / (p.thermal_capacitance_j_per_c * p.thermal_resistance_c_per_w)
    )

    assert rollout.states[-1, 2] == pytest.approx(expected, abs=1e-10)


def test_constant_current_equilibrium_has_zero_thermal_power(plant):
    equilibrium = plant.constant_current_thermal_equilibrium_c(5.0)
    balance = plant.thermal_power_balance(5.0, equilibrium)

    assert equilibrium > plant.parameters.ambient_temperature_c
    assert balance.copper_loss_w > 0.0
    assert balance.cooling_w > 0.0
    assert balance.net_heating_w == pytest.approx(0.0, abs=1e-12)


def test_alpha_zero_removes_temperature_from_electromechanical_dynamics(spec):
    parameters = spec.motor_parameters.with_temperature_coefficient(0.0)
    cold = spec.build_plant(parameters=parameters)
    hot = spec.build_plant(parameters=parameters)
    inputs = np.full(100, 18.0)

    cold_rollout = cold.rollout(
        inputs,
        initial_state=MotorState(0.0, 0.0, 25.0),
    )
    hot_rollout = hot.rollout(
        inputs,
        initial_state=MotorState(0.0, 0.0, 90.0),
    )

    np.testing.assert_array_equal(
        cold_rollout.states[:, :2],
        hot_rollout.states[:, :2],
    )
    assert cold_rollout.states[-1, 2] != hot_rollout.states[-1, 2]


def test_unstable_constant_current_thermal_equilibrium_is_rejected(spec):
    parameters = replace(
        spec.motor_parameters,
        thermal_resistance_c_per_w=10.0,
    )
    plant = spec.build_plant(parameters=parameters)

    with pytest.raises(ValueError, match="not stable"):
        plant.constant_current_thermal_equilibrium_c(10.0)


@pytest.mark.parametrize("voltage", [np.nan, np.inf, -np.inf])
def test_nonfinite_voltage_is_rejected(plant, voltage):
    with pytest.raises(ValueError, match="finite"):
        plant.step(voltage)
