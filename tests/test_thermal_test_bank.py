import math

import pytest

from r2dn_dc_motor.validation.thermal_test_bank import aggregate_thermal_test_cases


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
