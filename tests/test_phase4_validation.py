import json

import pytest

from r2dn_dc_motor.data import Phase4Dataset, generate_phase4_dataset
from r2dn_dc_motor.phase2_spec import load_phase2_spec
from r2dn_dc_motor.phase4_spec import REQUIRED_INTEGRITY_CHECKS, load_phase4_spec
from r2dn_dc_motor.validation.phase4 import run_phase4_validation


@pytest.fixture(scope="module")
def phase4_ci_dataset(tmp_path_factory):
    phase2 = load_phase2_spec()
    phase4 = load_phase4_spec(phase2=phase2)
    root = tmp_path_factory.mktemp("phase4") / "dataset"
    generate_phase4_dataset(
        root,
        profile_name="ci",
        spec=phase4,
        phase2=phase2,
    )
    return root, phase2, phase4


@pytest.mark.phase4_gate
def test_ci_dataset_passes_all_integrity_checks(phase4_ci_dataset):
    root, phase2, phase4 = phase4_ci_dataset

    report = run_phase4_validation(root, spec=phase4, phase2=phase2)

    assert report.passed is True
    assert report.trajectory_count == 32
    assert report.transition_count == 7_600
    assert {check.name for check in report.checks} == REQUIRED_INTEGRITY_CHECKS
    assert all(check.passed for check in report.checks)


@pytest.mark.phase4_gate
def test_loader_detects_manifest_tampering(phase4_ci_dataset):
    root, _, _ = phase4_ci_dataset
    manifest_path = root / "manifest.json"
    original = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(original)
    manifest["transition_count"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="fingerprint"):
            Phase4Dataset(root)
    finally:
        manifest_path.write_text(original, encoding="utf-8")
