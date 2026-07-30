from dataclasses import replace

import numpy as np
import pytest

from r2dn_dc_motor.models import nominal_isothermal_parameters
from r2dn_dc_motor.phase2_spec import load_phase2_spec
from r2dn_dc_motor.plants import (
    IsothermalParameters,
    IsothermalWorldModel,
    MotorState,
)


@pytest.fixture
def phase2():
    return load_phase2_spec()


def test_iso_nom_parameters_are_exact_phase2_projection(phase2):
    parameters = nominal_isothermal_parameters(phase2)
    full = phase2.motor_parameters

    assert parameters == IsothermalParameters(
        armature_inductance_h=full.armature_inductance_h,
        effective_resistance_ohm=full.reference_resistance_ohm,
        back_emf_constant_v_s_per_rad=full.back_emf_constant_v_s_per_rad,
        torque_constant_n_m_per_a=full.torque_constant_n_m_per_a,
        inertia_kg_m2=full.inertia_kg_m2,
        viscous_friction_n_m_s_per_rad=full.viscous_friction_n_m_s_per_rad,
        default_load_torque_n_m=full.default_load_torque_n_m,
    )


def test_iso_nom_rk4_map_matches_alpha_zero_full_plant(phase2):
    model = IsothermalWorldModel(
        nominal_isothermal_parameters(phase2),
        phase2.motor_limits,
        phase2.integration_settings,
        name="ISO-NOM",
    )
    full_parameters = phase2.motor_parameters.with_temperature_coefficient(0.0)
    full = phase2.build_plant(parameters=full_parameters)
    initial = MotorState(1.7, -23.0, 78.0)
    full.reset(initial)

    full_result = full.step(13.0)
    predicted = model.predict_next(
        np.asarray((initial.current_a, initial.speed_rad_s)),
        13.0,
    )

    np.testing.assert_allclose(
        predicted,
        full_result.state.as_array()[:2],
        rtol=0.0,
        atol=5e-14,
    )


def test_batch_and_scalar_one_step_are_identical(phase2):
    model = IsothermalWorldModel(
        nominal_isothermal_parameters(phase2),
        phase2.motor_limits,
        phase2.integration_settings,
        name="ISO-NOM",
    )
    states = np.asarray([[0.0, 0.0], [1.0, 20.0], [-2.0, -50.0]])
    controls = np.asarray([[0.0], [12.0], [-8.0]])

    batch = model.predict_next_batch(states, controls)
    scalar = np.vstack(
        [
            model.predict_next(state, voltage)
            for state, voltage in zip(states, controls, strict=True)
        ]
    )

    np.testing.assert_allclose(batch, scalar, rtol=0.0, atol=1e-14)


def test_free_rollout_is_autoregressive(phase2):
    model = IsothermalWorldModel(
        nominal_isothermal_parameters(phase2),
        phase2.motor_limits,
        phase2.integration_settings,
        name="ISO-NOM",
    )
    controls = np.full((25, 1), 8.0)
    rollout = model.free_rollout(np.asarray([0.0, 0.0]), controls)

    assert rollout.shape == (26, 2)
    for index in range(25):
        np.testing.assert_allclose(
            rollout[index + 1],
            model.predict_next(rollout[index], controls[index]),
        )


def test_nonpositive_physical_parameter_is_rejected(phase2):
    parameters = replace(
        nominal_isothermal_parameters(phase2),
        effective_resistance_ohm=0.0,
    )

    with pytest.raises(ValueError, match="positive"):
        IsothermalWorldModel(
            parameters,
            phase2.motor_limits,
            phase2.integration_settings,
            name="ISO-NOM",
        )
