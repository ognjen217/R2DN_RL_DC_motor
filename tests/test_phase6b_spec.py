from dataclasses import replace

import pytest

from r2dn_dc_motor.models.r2dn_phase6b import (
    adaptive_candidate_required,
    select_latent_from_medians,
)
from r2dn_dc_motor.phase2_spec import load_phase2_spec
from r2dn_dc_motor.phase4_spec import load_phase4_spec
from r2dn_dc_motor.phase6_spec import load_phase6_spec
from r2dn_dc_motor.phase6b_spec import (
    REQUIRED_PHASE6B_CHECKS,
    STRESS_SCENARIOS,
    load_phase6b_spec,
)
from r2dn_dc_motor.spec import SpecValidationError
from r2dn_dc_motor.validation.phase6b import scenario_voltage


@pytest.fixture
def specifications():
    phase2 = load_phase2_spec()
    phase4 = load_phase4_spec(phase2=phase2)
    phase6 = load_phase6_spec(phase4=phase4)
    phase6b = load_phase6b_spec(
        phase2=phase2,
        phase4=phase4,
        phase6=phase6,
    )
    return phase2, phase4, phase6, phase6b


def test_canonical_phase6b_spec_is_valid(specifications):
    phase2, phase4, phase6, phase6b = specifications

    phase6b.validate(phase2=phase2, phase4=phase4, phase6=phase6)

    final = phase6b.profile("final")
    assert final.candidate_latent_sizes == (4, 6, 8, 10, 12, 16)
    assert final.pilot_seeds == (1701, 2701, 3701)
    assert final.adaptive_candidate_latent == 24
    assert final.stress_horizon_steps == (10_000, 100_000, 1_000_000)
    assert tuple(value["name"] for value in phase6b.scenarios) == STRESS_SCENARIOS
    assert set(phase6b.validation["required_checks"]) == REQUIRED_PHASE6B_CHECKS


def test_near_tie_rule_prefers_smaller_latent():
    selected = select_latent_from_medians(
        {4: 0.40, 8: 0.103, 12: 0.100, 16: 0.20},
        relative_tolerance=0.03,
    )

    assert selected == 8


def test_adaptive_latent_requires_strictly_more_than_five_percent():
    assert (
        adaptive_candidate_required(
            {12: 0.20, 16: 0.19},
            reference_latent=12,
            boundary_latent=16,
            improvement_threshold=0.05,
        )
        is False
    )
    assert (
        adaptive_candidate_required(
            {12: 0.20, 16: 0.189},
            reference_latent=12,
            boundary_latent=16,
            improvement_threshold=0.05,
        )
        is True
    )


def test_id_or_ood_search_selection_is_rejected(specifications):
    phase2, phase4, phase6, phase6b = specifications
    changed = replace(
        phase6b,
        search={
            **phase6b.search,
            "selection_split": "ood_test",
            "ood_test_used_for_selection": True,
        },
    )

    with pytest.raises(SpecValidationError, match="search rule"):
        changed.validate(phase2=phase2, phase4=phase4, phase6=phase6)


def test_all_synthetic_scenarios_stay_inside_safe_voltage(specifications):
    _, phase4, _, phase6b = specifications
    safe = float(phase4.domain["safe_voltage_limit_v"])

    for scenario in phase6b.scenarios:
        first = scenario_voltage(
            scenario,
            start_step=0,
            steps=20_000,
            control_period_s=0.001,
        )
        second = scenario_voltage(
            scenario,
            start_step=10_000,
            steps=10_000,
            control_period_s=0.001,
        )
        assert abs(first).max() <= safe + 1e-12
        assert second == pytest.approx(first[10_000:])
