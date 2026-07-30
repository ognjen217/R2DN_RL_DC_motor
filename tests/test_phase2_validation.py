from r2dn_dc_motor.phase2_spec import REQUIRED_GATE_CHECKS
from r2dn_dc_motor.validation import run_phase2_validation


def test_gate0_validation_passes_all_required_checks():
    report = run_phase2_validation()

    assert report.passed is True
    assert {check.name for check in report.checks} == REQUIRED_GATE_CHECKS
    assert all(check.passed for check in report.checks)


def test_validation_report_contains_ordered_time_constants():
    report = run_phase2_validation()
    constants = report.time_constants_s

    assert constants["electrical"] < constants["mechanical"] < constants["thermal"]
