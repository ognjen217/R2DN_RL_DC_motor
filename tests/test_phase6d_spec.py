from __future__ import annotations

from dataclasses import replace

import pytest

from r2dn_dc_motor.phase6_spec import load_phase6_spec
from r2dn_dc_motor.phase6d_spec import VARIANT_NAMES, Phase6DSpec, load_phase6d_spec
from r2dn_dc_motor.spec import SpecValidationError


def test_final_accuracy_ablation_is_locked_and_controlled() -> None:
    phase6 = load_phase6_spec()
    spec = load_phase6d_spec(phase6=phase6)
    profile = spec.profile("final")

    assert tuple(value.name for value in profile.variants) == VARIANT_NAMES
    assert profile.seeds == (17, 29, 43)
    assert profile.selection_horizon_steps == 10_000
    assert tuple(
        (value.latent_size, value.burn_in_steps)
        for value in profile.variants
    ) == ((4, 250), (8, 250), (8, 1000), (8, 1000))
    assert spec.stages(profile.variants[0]) == phase6.profile("final").final_stages
    assert spec.stages(profile.variants[-1])[-1].rollout_steps == 5000


def test_phase6d_rejects_test_split_selection() -> None:
    phase6 = load_phase6_spec()
    spec = load_phase6d_spec(phase6=phase6)
    raw = {
        "schema_version": spec.schema_version,
        "phase": spec.phase,
        "selection": {**spec.selection, "id_test_used_for_selection": True},
        "screen": spec.screen,
        "profiles": {
            name: {
                **profile.__dict__,
                "variants": [value.__dict__ for value in profile.variants],
            }
            for name, profile in spec.profiles.items()
        },
        "curricula": {
            name: [stage.__dict__ for stage in stages]
            for name, stages in spec.curricula.items()
        },
        "validation": spec.validation,
    }
    with pytest.raises(SpecValidationError, match="id_test_used_for_selection"):
        Phase6DSpec.from_dict(raw, phase6=phase6)


def test_profile_dataclasses_remain_immutable() -> None:
    spec = load_phase6d_spec()
    profile = spec.profile("ci")
    changed = replace(profile, validation_windows=profile.validation_windows + 1)
    assert changed.validation_windows == profile.validation_windows + 1
    assert spec.profile("ci").validation_windows == profile.validation_windows
