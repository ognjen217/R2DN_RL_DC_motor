from dataclasses import replace

import pytest

from r2dn_dc_motor.phase1_spec import load_phase1_spec
from r2dn_dc_motor.spec import SpecValidationError, load_phase0_spec


@pytest.fixture
def phase0():
    return load_phase0_spec()


@pytest.fixture
def phase1():
    return load_phase1_spec()


def test_canonical_phase1_spec_is_valid(phase0, phase1):
    phase1.validate(phase0)


def test_phase1_interface_matches_frozen_phase0(phase0, phase1):
    assert tuple(phase1.interface["regressor_features"]) == phase0.signals.world_model_input
    assert tuple(phase1.interface["target_features"]) == phase0.signals.world_model_output
    assert phase1.interface["temporal_representation"] == (
        phase0.temporal_representation["r2dn"]
    )
    assert phase1.upstream["commit"] == phase0.r2dn_reference["commit"]


def test_latent_state_is_larger_than_measured_output(phase1):
    assert phase1.interface["minimum_latent_size"] > len(
        phase1.interface["observation_features"]
    )
    assert phase1.interface["pilot_latent_size"] >= phase1.interface["minimum_latent_size"]


def test_temperature_leak_into_regressor_is_rejected(phase0, phase1):
    changed_interface = {
        **phase1.interface,
        "regressor_features": (
            *phase1.interface["regressor_features"],
            "winding_temperature_c",
        ),
        "input_size": 4,
    }
    changed_spec = replace(phase1, interface=changed_interface)

    with pytest.raises(SpecValidationError, match="temperature leaked"):
        changed_spec.validate(phase0)


def test_teacher_forcing_in_free_rollout_is_rejected(phase0, phase1):
    changed_spec = replace(
        phase1,
        rollout={**phase1.rollout, "teacher_forcing_after_burn_in": True},
    )

    with pytest.raises(SpecValidationError, match="must not use teacher forcing"):
        changed_spec.validate(phase0)


def test_contractivity_scope_cannot_be_overclaimed(phase0, phase1):
    changed_spec = replace(
        phase1,
        contractivity={
            **phase1.contractivity,
            "autoregressive_scope": "formally_certified",
        },
    )

    with pytest.raises(SpecValidationError, match="must not be presented"):
        changed_spec.validate(phase0)


def test_burn_in_length_remains_deferred_until_motor_time_constants_are_known(phase1):
    assert phase1.initialization["burn_in_steps_policy"] == (
        "derive_after_phase2_time_constant_analysis"
    )
