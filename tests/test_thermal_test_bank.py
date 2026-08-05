import math
from types import SimpleNamespace

import numpy as np
import pytest

from r2dn_dc_motor.phase2_spec import load_phase2_spec
from r2dn_dc_motor.phase7_spec import load_phase7_spec
from r2dn_dc_motor.validation.thermal_test_bank import (
    aggregate_thermal_test_cases,
    resolve_test_bank_scenario,
    run_thermal_test_bank_preflight,
    select_thermal_test_cases,
    validate_thermal_test_bank_preflight,
)


def _accuracy(value):
    return {
        "horizons": [
            {"duration_s": 1.0, "combined_nrmse": value},
            {"duration_s": 10.0, "combined_nrmse": 2.0 * value},
        ]
    }


def _case(category, first, second):
    return {
        "case": {"category": category},
        "models": {
            "R2DN": {"divergent": False, "accuracy": _accuracy(first)},
            "ISO-CAL": {"divergent": False, "accuracy": _accuracy(second)},
        },
    }


def test_test_bank_reports_mean_median_worst_wins_and_regimes():
    cases = (
        _case("id", 0.1, 0.2),
        _case("excitation_ood", 0.3, 0.2),
        _case("thermal_ood", 0.5, 0.4),
    )

    result = aggregate_thermal_test_cases(
        cases,
        model_names=("R2DN", "ISO-CAL"),
        horizons_s=(1.0, 10.0),
    )

    r2dn = result["overall"]["1s"]["R2DN"]
    assert r2dn["mean_combined_nrmse"] == pytest.approx(0.3)
    assert r2dn["median_combined_nrmse"] == pytest.approx(0.3)
    assert r2dn["best_combined_nrmse"] == pytest.approx(0.1)
    assert r2dn["worst_combined_nrmse"] == pytest.approx(0.5)
    assert result["model_totals"]["R2DN"]["horizon_wins"]["1s"] == 1
    assert result["model_totals"]["ISO-CAL"]["horizon_wins"]["1s"] == 2
    assert result["by_category"]["thermal_ood"]["10s"]["R2DN"][
        "median_combined_nrmse"
    ] == pytest.approx(1.0)


def test_test_bank_counts_divergence_without_dropping_other_models():
    case = _case("id", 0.1, 0.2)
    case["models"]["R2DN"] = {
        "divergent": True,
        "accuracy": None,
    }

    result = aggregate_thermal_test_cases(
        (case,),
        model_names=("R2DN", "ISO-CAL"),
        horizons_s=(1.0,),
    )

    assert result["model_totals"]["R2DN"]["divergent_cases"] == 1
    assert math.isinf(result["overall"]["1s"]["R2DN"]["median_combined_nrmse"])
    assert result["overall"]["1s"]["ISO-CAL"]["finite_cases"] == 1


class _TinyEvaluationDataset:
    fingerprint = "tiny-phase7-test-bank"

    def __init__(self):
        self._records = {
            "id": {
                "split": "id_test",
                "ood_axis": None,
                "excitation_family": "multisine_voltage",
            },
            "excitation": {
                "split": "ood_test",
                "ood_axis": "novel_profile",
                "excitation_family": "chirp_voltage",
            },
            "thermal": {
                "split": "ood_test",
                "ood_axis": "higher_initial_temperature",
                "excitation_family": "pi_closed_loop",
            },
        }
        self._temperatures = {"id": 25.0, "excitation": 55.0, "thermal": 85.0}

    def trajectory_ids(self, split):
        return tuple(
            name for name, record in self._records.items() if record["split"] == split
        )

    def record(self, trajectory_id):
        return dict(self._records[trajectory_id])

    def load_trajectory(self, trajectory_id):
        states = np.zeros((301, 3), dtype=np.float32)
        states[:, 2] = self._temperatures[trajectory_id]
        return SimpleNamespace(states=states, transitions=300)


def test_phase7_scenarios_are_owned_by_the_test_bank_not_phase6b():
    phase7 = load_phase7_spec()

    prbs = resolve_test_bank_scenario(phase7, "prbs")

    assert prbs["amplitude_v"] == 5.0
    assert prbs["hold_steps"] == 250


def test_full_rk4_preflight_freezes_and_validates_the_bank():
    dataset = _TinyEvaluationDataset()
    phase2 = load_phase2_spec()
    phase7 = load_phase7_spec()

    report = run_thermal_test_bank_preflight(
        evaluation_dataset=dataset,
        phase2=phase2,
        phase7=phase7,
        profile_name="ci",
        duration_s=1.0,
    )

    assert report.passed is True
    assert len(report.cases) == 3
    assert max(
        value["reference"]["maximum_temperature_c"] for value in report.cases
    ) <= 110.0
    cases = select_thermal_test_cases(
        dataset,
        phase7=phase7,
        profile_name="ci",
        minimum_burn_in_steps=250,
    )
    validate_thermal_test_bank_preflight(
        report.to_dict(),
        evaluation_dataset=dataset,
        phase7=phase7,
        profile_name="ci",
        duration_s=1.0,
        cases=cases,
    )

    stale = report.to_dict()
    stale["test_bank_fingerprint"] = "stale"
    with pytest.raises(ValueError, match="fingerprint"):
        validate_thermal_test_bank_preflight(
            stale,
            evaluation_dataset=dataset,
            phase7=phase7,
            profile_name="ci",
            duration_s=1.0,
            cases=cases,
        )
