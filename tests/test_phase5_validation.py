import json

import pytest

from r2dn_dc_motor.data import Phase4Dataset, generate_phase4_dataset
from r2dn_dc_motor.models import fit_global_isothermal_parameters
from r2dn_dc_motor.phase2_spec import load_phase2_spec
from r2dn_dc_motor.phase4_spec import load_phase4_spec
from r2dn_dc_motor.phase5_spec import REQUIRED_PHASE5_CHECKS, load_phase5_spec
from r2dn_dc_motor.validation.phase5 import (
    generate_phase5_artifacts,
    run_phase5_validation,
)


@pytest.mark.phase5_gate
def test_ci_baselines_fit_and_evaluate_without_leakage(tmp_path):
    phase2 = load_phase2_spec()
    phase4 = load_phase4_spec(phase2=phase2)
    phase5 = load_phase5_spec(phase2=phase2, phase4=phase4)
    dataset_root = tmp_path / "dataset"
    generate_phase4_dataset(
        dataset_root,
        profile_name="ci",
        spec=phase4,
        phase2=phase2,
    )
    dataset = Phase4Dataset(dataset_root)
    checkpoint = fit_global_isothermal_parameters(
        dataset,
        phase2,
        resistance_bounds=phase5.resistance_bounds,
        friction_bounds=phase5.friction_bounds,
        minimum_regressor_energy=float(
            phase5.calibration["minimum_regressor_energy"]
        ),
        forbidden_fit_features=tuple(phase5.interface["forbidden_fit_features"]),
        method=str(phase5.calibration["method"]),
        selection_policy=str(phase5.calibration["selection_policy"]),
    )

    report = run_phase5_validation(
        dataset,
        checkpoint,
        spec=phase5,
        phase2=phase2,
    )
    report_path, figure_path = generate_phase5_artifacts(
        report,
        tmp_path / "results",
    )

    assert report.passed is True
    assert {check.name for check in report.checks} == REQUIRED_PHASE5_CHECKS
    assert all(check.passed for check in report.checks)
    assert set(report.metrics) == {"ISO-NOM", "ISO-CAL"}
    assert checkpoint.sufficient_statistics.trajectory_count == 8
    assert json.loads(report_path.read_text())["passed"] is True
    assert figure_path.stat().st_size > 1_000
