from __future__ import annotations

from dataclasses import asdict

import pytest

from r2dn_dc_motor.phase2_spec import load_phase2_spec
from r2dn_dc_motor.phase6_spec import load_phase6_spec
from r2dn_dc_motor.phase6b_spec import load_phase6b_spec
from r2dn_dc_motor.phase6e_spec import Phase6ESpec, load_phase6e_spec
from r2dn_dc_motor.spec import SpecValidationError


def _raw_spec(spec: Phase6ESpec) -> dict:
    profiles = {name: asdict(profile) for name, profile in spec.profiles.items()}
    for profile in profiles.values():
        profile["multisine_scenarios"] = list(profile["multisine_scenarios"])
    return {
        "schema_version": spec.schema_version,
        "phase": spec.phase,
        "selection": spec.selection,
        "profiles": profiles,
        "validation": spec.validation,
    }


def test_final_larger_latent_protocol_is_locked() -> None:
    phase6 = load_phase6_spec()
    spec = load_phase6e_spec(phase6=phase6)
    profile = spec.profile("final")

    assert profile.latent_sizes == (8, 12, 16)
    assert profile.seeds == (17, 29, 43)
    assert profile.burn_in_steps == 250
    assert profile.selection_duration_s == 100.0
    assert profile.selection_anchor_indices == (0, 1, 2)
    assert len(profile.multisine_scenarios) == 3
    assert phase6.profile("final").final_stages[-1].rollout_steps == 1000
    assert spec.selection["canonical_phase6c_scenario_used_for_selection"] is False


def test_phase6e_rejects_canonical_phase6c_selection_leak() -> None:
    phase2 = load_phase2_spec()
    phase6 = load_phase6_spec()
    phase6b = load_phase6b_spec(phase2=phase2, phase6=phase6)
    spec = load_phase6e_spec(phase2=phase2, phase6=phase6, phase6b=phase6b)
    raw = _raw_spec(spec)
    canonical = next(value for value in phase6b.scenarios if value["name"] == "multisine")
    raw["profiles"]["final"]["multisine_scenarios"][0] = {
        **canonical,
        "name": "leaked_canonical",
    }

    with pytest.raises(SpecValidationError, match="canonical Phase-6C multisine leaked"):
        Phase6ESpec.from_dict(
            raw,
            phase2=phase2,
            phase6=phase6,
            phase6b=phase6b,
        )


def test_phase6e_rejects_nonidentical_seed_catalog() -> None:
    phase2 = load_phase2_spec()
    phase6 = load_phase6_spec()
    phase6b = load_phase6b_spec(phase2=phase2, phase6=phase6)
    spec = load_phase6e_spec(phase2=phase2, phase6=phase6, phase6b=phase6b)
    raw = _raw_spec(spec)
    raw["profiles"]["final"]["seeds"] = [17, 29]

    with pytest.raises(SpecValidationError, match="seeds must remain"):
        Phase6ESpec.from_dict(
            raw,
            phase2=phase2,
            phase6=phase6,
            phase6b=phase6b,
        )
