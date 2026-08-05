from __future__ import annotations

from dataclasses import asdict

import pytest

from r2dn_dc_motor.phase2_spec import load_phase2_spec
from r2dn_dc_motor.phase6_spec import load_phase6_spec
from r2dn_dc_motor.phase6b_spec import load_phase6b_spec
from r2dn_dc_motor.phase6e_spec import load_phase6e_spec
from r2dn_dc_motor.phase6f_spec import Phase6FSpec, load_phase6f_spec
from r2dn_dc_motor.spec import SpecValidationError


def _raw_spec(spec: Phase6FSpec) -> dict:
    return {
        "schema_version": spec.schema_version,
        "phase": spec.phase,
        "selection": spec.selection,
        "profiles": {name: asdict(value) for name, value in spec.profiles.items()},
        "validation": spec.validation,
    }


def test_final_optimizer_floor_protocol_is_locked() -> None:
    spec = load_phase6f_spec()
    profile = spec.profile("final")

    assert profile.latent_size == 16
    assert profile.seed == 43
    assert profile.burn_in_steps == 250
    assert profile.selection_duration_s == 100.0
    assert profile.selection_anchor_indices == (3, 0, 2)
    assert tuple(value.name for value in profile.variants) == (
        "baseline_phase6e",
        "cosine_1x",
        "cosine_2x",
    )
    assert tuple(value.stage_update_multiplier for value in profile.variants) == (1, 1, 2)
    assert profile.variants[1].initial_learning_rate == 1e-3
    assert profile.variants[1].final_learning_rate == 1e-5


def test_phase6f_rejects_phase6e_selection_scenario_reuse() -> None:
    phase2 = load_phase2_spec()
    phase6 = load_phase6_spec()
    phase6b = load_phase6b_spec(phase2=phase2, phase6=phase6)
    phase6e = load_phase6e_spec(phase2=phase2, phase6=phase6, phase6b=phase6b)
    spec = load_phase6f_spec(
        phase2=phase2,
        phase6=phase6,
        phase6b=phase6b,
        phase6e=phase6e,
    )
    raw = _raw_spec(spec)
    raw["profiles"]["final"]["multisine_scenarios"][0] = asdict(
        phase6e.profile("final").multisine_scenarios[0]
    )

    with pytest.raises(SpecValidationError, match="selection scenario was reused"):
        Phase6FSpec.from_dict(
            raw,
            phase2=phase2,
            phase6=phase6,
            phase6b=phase6b,
            phase6e=phase6e,
        )


def test_phase6f_rejects_network_or_seed_drift() -> None:
    phase2 = load_phase2_spec()
    phase6 = load_phase6_spec()
    phase6b = load_phase6b_spec(phase2=phase2, phase6=phase6)
    phase6e = load_phase6e_spec(phase2=phase2, phase6=phase6, phase6b=phase6b)
    spec = load_phase6f_spec(
        phase2=phase2,
        phase6=phase6,
        phase6b=phase6b,
        phase6e=phase6e,
    )
    raw = _raw_spec(spec)
    raw["profiles"]["final"]["seed"] = 29

    with pytest.raises(SpecValidationError, match="latent-16/seed-43"):
        Phase6FSpec.from_dict(
            raw,
            phase2=phase2,
            phase6=phase6,
            phase6b=phase6b,
            phase6e=phase6e,
        )
