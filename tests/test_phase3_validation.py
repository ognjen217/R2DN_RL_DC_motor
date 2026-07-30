import pytest

from r2dn_dc_motor.phase3_spec import REQUIRED_GATE_CHECKS
from r2dn_dc_motor.validation import run_phase3_validation


@pytest.fixture(scope="module")
def report():
    return run_phase3_validation()


@pytest.mark.phase3_gate
def test_gate1_validation_passes_all_required_checks(report):
    assert report.passed is True
    assert {check.name for check in report.checks} == REQUIRED_GATE_CHECKS
    assert all(check.passed for check in report.checks)


@pytest.mark.phase3_gate
def test_thermal_signal_dominates_solver_error(report):
    metrics = report.physical_metrics

    assert metrics["signal_to_solver_ratio"] >= 10_000.0
    assert metrics["maximum_current_difference_a"] >= 0.5
    assert metrics["maximum_speed_difference_rad_s"] >= 5.0


@pytest.mark.phase3_gate
def test_selected_history_probe_beats_instantaneous_probe(report):
    selected = next(
        item
        for item in report.probe_results
        if item.history_s == report.selected_history_s
    )

    assert selected.test_samples >= 100
    assert selected.history_mae_c <= 2.0
    assert selected.instantaneous_mae_c >= 5.0
    assert selected.history_mae_c < 0.5 * selected.instantaneous_mae_c
