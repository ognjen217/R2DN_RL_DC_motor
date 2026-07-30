from dataclasses import replace

import pytest

from r2dn_dc_motor.phase2_spec import load_phase2_spec
from r2dn_dc_motor.phase4_spec import load_phase4_spec
from r2dn_dc_motor.phase5_spec import (
    REQUIRED_PHASE5_CHECKS,
    load_phase5_spec,
)
from r2dn_dc_motor.spec import SpecValidationError, load_phase0_spec


@pytest.fixture
def specifications():
    phase0 = load_phase0_spec()
    phase2 = load_phase2_spec(phase0=phase0)
    phase4 = load_phase4_spec(phase0=phase0, phase2=phase2)
    phase5 = load_phase5_spec(
        phase0=phase0,
        phase2=phase2,
        phase4=phase4,
    )
    return phase0, phase2, phase4, phase5


def test_canonical_phase5_spec_is_valid(specifications):
    phase0, phase2, phase4, phase5 = specifications

    phase5.validate(phase0=phase0, phase2=phase2, phase4=phase4)

    assert set(phase5.validation["required_checks"]) == REQUIRED_PHASE5_CHECKS
    assert phase5.rollout_horizons_s == {
        "short": 0.1,
        "medium": 1.0,
        "long": 10.0,
    }


def test_temperature_or_load_calibration_is_rejected(specifications):
    phase0, phase2, phase4, phase5 = specifications
    changed = replace(
        phase5,
        calibration={
            **phase5.calibration,
            "temperature_used": True,
            "load_torque_used": True,
        },
    )

    with pytest.raises(SpecValidationError, match="temperature"):
        changed.validate(phase0=phase0, phase2=phase2, phase4=phase4)


def test_validation_tuning_is_rejected(specifications):
    phase0, phase2, phase4, phase5 = specifications
    changed = replace(
        phase5,
        calibration={
            **phase5.calibration,
            "selection_policy": "choose_best_on_id_test",
        },
    )

    with pytest.raises(SpecValidationError, match="validation/test"):
        changed.validate(phase0=phase0, phase2=phase2, phase4=phase4)


def test_per_episode_parameter_source_is_rejected(specifications):
    phase0, phase2, phase4, phase5 = specifications
    changed = replace(
        phase5,
        models={
            **phase5.models,
            "iso_cal": {
                **phase5.models["iso_cal"],
                "parameter_source": "fit_each_test_episode",
            },
        },
    )

    with pytest.raises(SpecValidationError, match="global"):
        changed.validate(phase0=phase0, phase2=phase2, phase4=phase4)


def test_extra_calibrated_parameter_is_rejected(specifications):
    phase0, phase2, phase4, phase5 = specifications
    changed = replace(
        phase5,
        models={
            **phase5.models,
            "iso_cal": {
                **phase5.models["iso_cal"],
                "calibrated_parameters": [
                    *phase5.models["iso_cal"]["calibrated_parameters"],
                    "inertia_kg_m2",
                ],
            },
        },
    )

    with pytest.raises(SpecValidationError, match="only effective R and b"):
        changed.validate(phase0=phase0, phase2=phase2, phase4=phase4)
