from dataclasses import replace

import pytest

from r2dn_dc_motor.phase1_spec import load_phase1_spec
from r2dn_dc_motor.phase4_spec import load_phase4_spec
from r2dn_dc_motor.phase6_spec import REQUIRED_PHASE6_CHECKS, load_phase6_spec
from r2dn_dc_motor.spec import SpecValidationError, load_phase0_spec


@pytest.fixture
def specifications():
    phase0 = load_phase0_spec()
    phase1 = load_phase1_spec(phase0=phase0)
    phase4 = load_phase4_spec(phase0=phase0)
    phase6 = load_phase6_spec(
        phase0=phase0,
        phase1=phase1,
        phase4=phase4,
    )
    return phase0, phase1, phase4, phase6


def test_canonical_phase6_spec_is_valid(specifications):
    phase0, phase1, phase4, phase6 = specifications

    phase6.validate(phase0=phase0, phase1=phase1, phase4=phase4)

    assert set(phase6.validation["required_checks"]) == REQUIRED_PHASE6_CHECKS
    assert phase6.profile("final").candidate_latent_sizes == (4, 6, 8)
    assert phase6.profile("final").training_seeds == (17, 29, 43)
    assert phase6.profile("final").burn_in_steps == 250
    assert tuple(
        stage.rollout_steps for stage in phase6.profile("final").pilot_stages
    ) == (1, 10, 100)
    assert sum(
        stage.updates for stage in phase6.profile("final").pilot_stages
    ) == 400


def test_hidden_temperature_in_regressor_is_rejected(specifications):
    phase0, phase1, phase4, phase6 = specifications
    changed = replace(
        phase6,
        interface={
            **phase6.interface,
            "regressor": [
                *phase6.interface["regressor"],
                "winding_temperature_c",
            ],
        },
    )

    with pytest.raises(SpecValidationError, match="regressor"):
        changed.validate(phase0=phase0, phase1=phase1, phase4=phase4)


def test_id_or_ood_checkpoint_selection_is_rejected(specifications):
    phase0, phase1, phase4, phase6 = specifications
    changed = replace(
        phase6,
        selection={
            **phase6.selection,
            "selection_split": "id_test",
            "id_test_used_for_selection": True,
            "ood_test_used_for_selection": True,
        },
    )

    with pytest.raises(SpecValidationError, match="validation"):
        changed.validate(phase0=phase0, phase1=phase1, phase4=phase4)


def test_final_profile_requires_multiple_seeds(specifications):
    phase0, phase1, phase4, phase6 = specifications
    final = replace(phase6.profile("final"), training_seeds=(17,))
    changed = replace(
        phase6,
        profiles={
            **phase6.profiles,
            "final": final,
        },
    )

    with pytest.raises(SpecValidationError, match="at least three seeds"):
        changed.validate(phase0=phase0, phase1=phase1, phase4=phase4)


def test_teacher_forcing_in_rollout_loss_is_rejected(specifications):
    phase0, phase1, phase4, phase6 = specifications
    changed = replace(
        phase6,
        loss={
            **phase6.loss,
            "teacher_forcing_after_burn_in": True,
        },
    )

    with pytest.raises(SpecValidationError, match="teacher-force"):
        changed.validate(phase0=phase0, phase1=phase1, phase4=phase4)


def test_final_pilot_without_short_rollout_bridge_is_rejected(specifications):
    phase0, phase1, phase4, phase6 = specifications
    final = phase6.profile("final")
    changed_final = replace(
        final,
        pilot_stages=(final.pilot_stages[0], final.pilot_stages[-1]),
    )
    changed = replace(
        phase6,
        profiles={
            **phase6.profiles,
            "final": changed_final,
        },
    )

    with pytest.raises(SpecValidationError, match="must bridge"):
        changed.validate(phase0=phase0, phase1=phase1, phase4=phase4)
