from dataclasses import replace

import pytest

from r2dn_dc_motor.phase2_spec import load_phase2_spec
from r2dn_dc_motor.phase4_spec import (
    EXCITATION_FAMILIES,
    REQUIRED_INTEGRITY_CHECKS,
    SPLIT_NAMES,
    load_phase4_spec,
)
from r2dn_dc_motor.spec import SpecValidationError, load_phase0_spec


@pytest.fixture
def phase0():
    return load_phase0_spec()


@pytest.fixture
def phase2():
    return load_phase2_spec()


@pytest.fixture
def phase4(phase2):
    return load_phase4_spec(phase2=phase2)


def test_canonical_phase4_spec_is_valid(phase0, phase2, phase4):
    phase4.validate(phase0=phase0, phase2=phase2)
    assert set(phase4.integrity["required_checks"]) == REQUIRED_INTEGRITY_CHECKS


def test_final_profile_is_locked_to_320_trajectories_and_4_8m_transitions(phase4):
    final = phase4.profile("final")

    assert final.trajectory_count == 320
    assert final.minimum_total_transitions == 4_800_000
    assert final.split_counts == {
        "train": 224,
        "validation": 32,
        "id_test": 32,
        "ood_test": 32,
    }


def test_every_profile_covers_each_family_in_each_split(phase4):
    for profile in phase4.profiles.values():
        for split in SPLIT_NAMES:
            assert profile.split_counts[split] >= len(EXCITATION_FAMILIES)
            assert profile.split_counts[split] % len(EXCITATION_FAMILIES) == 0


def test_temperature_leakage_into_model_view_is_rejected(phase0, phase2, phase4):
    changed = replace(
        phase4,
        features={
            **phase4.features,
            "model_observation": [
                *phase4.features["model_observation"],
                "winding_temperature_c",
            ],
        },
    )

    with pytest.raises(SpecValidationError, match="temperature"):
        changed.validate(phase0=phase0, phase2=phase2)


def test_validation_normalization_is_rejected(phase0, phase2, phase4):
    changed = replace(
        phase4,
        normalization={
            **phase4.normalization,
            "fit_split": "validation",
        },
    )

    with pytest.raises(SpecValidationError, match="train"):
        changed.validate(phase0=phase0, phase2=phase2)


def test_non_full_source_is_rejected(phase0, phase2, phase4):
    changed = replace(
        phase4,
        dataset={
            **phase4.dataset,
            "simulator": "ISO-NOM",
        },
    )

    with pytest.raises(SpecValidationError, match="FULL"):
        changed.validate(phase0=phase0, phase2=phase2)


def test_sample_level_split_is_rejected(phase0, phase2, phase4):
    changed = replace(
        phase4,
        splits={
            **phase4.splits,
            "policy": "random_sample_split",
        },
    )

    with pytest.raises(SpecValidationError, match="whole trajectories"):
        changed.validate(phase0=phase0, phase2=phase2)
