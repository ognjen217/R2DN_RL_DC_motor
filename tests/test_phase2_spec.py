from dataclasses import replace

import pytest

from r2dn_dc_motor.phase2_spec import REQUIRED_GATE_CHECKS, load_phase2_spec
from r2dn_dc_motor.spec import Range, SpecValidationError, load_phase0_spec


@pytest.fixture
def phase0():
    return load_phase0_spec()


@pytest.fixture
def phase2():
    return load_phase2_spec()


def test_canonical_phase2_spec_is_valid(phase0, phase2):
    phase2.validate(phase0)


def test_phase2_state_and_limits_match_phase0(phase0, phase2):
    assert tuple(phase2.state["order"]) == phase0.signals.state
    assert tuple(phase2.state["measured"]) == phase0.signals.plant_output
    assert tuple(phase2.state["hidden"]) == phase0.signals.hidden_state

    for name, bounds in phase2.limits.items():
        assert bounds == phase0.limits[name]


def test_phase2_resolves_all_three_time_scales(phase2):
    constants = phase2.time_constants_s()
    integration = phase2.integration_settings

    assert constants["electrical"] == pytest.approx(0.0025)
    assert constants["mechanical"] == pytest.approx(0.2727272727)
    assert constants["thermal"] == pytest.approx(24.0)
    assert integration.integrator_step_s <= constants["electrical"] / 20.0
    assert phase2.time_horizons["pilot_burn_in_s"] >= 5.0 * constants["mechanical"]
    assert phase2.time_horizons["core_episode_s"] >= 4.0 * constants["thermal"]


def test_phase2_implements_all_gate0_checks(phase2):
    assert set(phase2.gate["required_checks"]) == REQUIRED_GATE_CHECKS


def test_limit_drift_from_phase0_is_rejected(phase0, phase2):
    changed_limits = {
        **phase2.limits,
        "armature_current_a": Range(-30.0, 30.0),
    }
    changed_spec = replace(phase2, limits=changed_limits)

    with pytest.raises(SpecValidationError, match="limits drifted"):
        changed_spec.validate(phase0)


def test_nonpositive_thermal_coefficient_is_rejected(phase0, phase2):
    changed_parameters = {
        **phase2.parameters,
        "electrical": {
            **phase2.parameters["electrical"],
            "resistance_temperature_coefficient_per_c": 0.0,
        },
    }
    changed_spec = replace(phase2, parameters=changed_parameters)

    with pytest.raises(SpecValidationError, match="temperature coefficient"):
        changed_spec.validate(phase0)


def test_integrator_step_that_underresolves_electrical_dynamics_is_rejected(
    phase0,
    phase2,
):
    changed_spec = replace(
        phase2,
        numerics={
            **phase2.numerics,
            "integrator_step_s": 0.001,
        },
    )

    with pytest.raises(SpecValidationError, match="integrator step drifted"):
        changed_spec.validate(phase0)


def test_short_burn_in_is_rejected(phase0, phase2):
    changed_spec = replace(
        phase2,
        time_horizons={
            **phase2.time_horizons,
            "pilot_burn_in_s": 0.1,
        },
    )

    with pytest.raises(SpecValidationError, match="five mechanical"):
        changed_spec.validate(phase0)


def test_random_reset_ranges_stay_in_domain(phase2):
    ranges = phase2.reset_ranges
    limits = phase2.motor_limits

    assert limits.minimum_current_a <= ranges.current_a[0] < ranges.current_a[1]
    assert ranges.current_a[1] <= limits.maximum_current_a
    assert limits.minimum_speed_rad_s <= ranges.speed_rad_s[0] < ranges.speed_rad_s[1]
    assert ranges.speed_rad_s[1] <= limits.maximum_speed_rad_s
    assert (
        limits.minimum_temperature_c
        <= ranges.temperature_c[0]
        < ranges.temperature_c[1]
        <= limits.maximum_temperature_c
    )
