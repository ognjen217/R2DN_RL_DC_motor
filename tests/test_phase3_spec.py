from dataclasses import replace

import pytest

from r2dn_dc_motor.phase2_spec import load_phase2_spec
from r2dn_dc_motor.phase3_spec import (
    ALLOWED_PROBE_SIGNALS,
    REQUIRED_GATE_CHECKS,
    load_phase3_spec,
)
from r2dn_dc_motor.spec import SpecValidationError


@pytest.fixture
def phase2():
    return load_phase2_spec()


@pytest.fixture
def phase3(phase2):
    return load_phase3_spec(phase2=phase2)


def test_canonical_phase3_spec_is_valid(phase2, phase3):
    phase3.validate(phase2)


def test_probe_view_is_temperature_free(phase3):
    assert tuple(phase3.probe["allowed_signals"]) == ALLOWED_PROBE_SIGNALS
    assert phase3.probe["forbidden_signals"] == ["winding_temperature_c"]
    assert all(
        "temperature" not in signal for signal in phase3.probe["allowed_signals"]
    )


def test_pilot_split_uses_disjoint_whole_trajectories(phase3):
    train = set(phase3.train_trajectory_ids)
    test = set(phase3.test_trajectory_ids)

    assert train.isdisjoint(test)
    assert train | test == set(range(phase3.pilot["trajectory_count"]))
    assert phase3.pilot["split_policy"] == "whole_trajectory_stratified_temperature"


def test_phase3_implements_all_gate1_checks(phase3):
    assert set(phase3.gate["required_checks"]) == REQUIRED_GATE_CHECKS


def test_temperature_leakage_is_rejected(phase2, phase3):
    changed = replace(
        phase3,
        probe={
            **phase3.probe,
            "allowed_signals": [
                *phase3.probe["allowed_signals"],
                "winding_temperature_c",
            ],
        },
    )

    with pytest.raises(SpecValidationError, match="temperature"):
        changed.validate(phase2)


def test_sample_level_split_policy_is_rejected(phase2, phase3):
    changed = replace(
        phase3,
        pilot={
            **phase3.pilot,
            "split_policy": "random_sample_split",
        },
    )

    with pytest.raises(SpecValidationError, match="whole trajectory"):
        changed.validate(phase2)


def test_selected_history_must_be_evaluated(phase2, phase3):
    changed = replace(
        phase3,
        probe={
            **phase3.probe,
            "selected_history_s": 0.3,
        },
    )

    with pytest.raises(SpecValidationError, match="selected probe history"):
        changed.validate(phase2)


def test_unphysical_voltage_level_is_rejected(phase2, phase3):
    changed = replace(
        phase3,
        pilot={
            **phase3.pilot,
            "voltage_levels_v": [-18.0, 0.0, 60.0],
        },
    )

    with pytest.raises(SpecValidationError, match="actuator limits"):
        changed.validate(phase2)
