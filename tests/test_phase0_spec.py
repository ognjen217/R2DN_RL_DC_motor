from dataclasses import replace

import pytest

from r2dn_dc_motor.spec import SpecValidationError, load_phase0_spec


@pytest.fixture
def spec():
    return load_phase0_spec()


def test_canonical_phase0_spec_is_valid(spec):
    spec.validate()


def test_temperature_is_hidden_from_world_model_and_policy(spec):
    hidden = set(spec.signals.hidden_state)

    assert hidden.isdisjoint(spec.signals.world_model_input)
    assert hidden.isdisjoint(spec.signals.world_model_output)
    assert hidden.isdisjoint(spec.signals.policy_observation)


def test_temperature_is_available_only_to_full_plant_and_evaluation(spec):
    assumptions = spec.assumptions

    assert assumptions["temperature_available_to_full_plant"] is True
    assert assumptions["temperature_available_to_world_models"] is False
    assert assumptions["temperature_available_to_policy"] is False
    assert assumptions["temperature_used_for_evaluation"] is True


def test_reference_speed_stays_inside_plant_domain(spec):
    assert spec.limits["angular_speed_rad_s"].contains(
        spec.limits["angular_speed_reference_rad_s"]
    )


def test_r2dn_reference_is_pinned(spec):
    commit = spec.r2dn_reference["commit"]

    assert len(commit) == 40
    assert spec.r2dn_reference["class_name"] == "ContractingR2DN"
    assert spec.temporal_representation["r2dn"] == "discrete_recurrent_state_space"


def test_temperature_leak_is_rejected(spec):
    leaked_signals = replace(
        spec.signals,
        policy_observation=spec.signals.policy_observation + ("winding_temperature_c",),
    )
    leaked_spec = replace(spec, signals=leaked_signals)

    with pytest.raises(SpecValidationError, match="temperature leaked"):
        leaked_spec.validate()


def test_primary_metric_drift_is_rejected(spec):
    changed_metrics = {**spec.metrics, "primary": ("one_step_rmse",)}
    changed_spec = replace(spec, metrics=changed_metrics)

    with pytest.raises(SpecValidationError, match="primary metrics changed"):
        changed_spec.validate()

