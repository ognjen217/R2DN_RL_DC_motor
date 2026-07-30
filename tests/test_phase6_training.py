import json

import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("optax")
pytest.importorskip("robustnn")

from r2dn_dc_motor.data import Phase4Dataset, generate_phase4_dataset
from r2dn_dc_motor.models import (
    load_phase6_checkpoint,
    save_phase6_checkpoint,
    train_phase6_study,
)
from r2dn_dc_motor.phase6_spec import REQUIRED_PHASE6_CHECKS, load_phase6_spec
from r2dn_dc_motor.validation.phase6 import (
    generate_phase6_artifacts,
    run_phase6_validation,
)

pytestmark = pytest.mark.phase6_gate


@pytest.fixture(scope="module")
def trained_checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("phase6")
    dataset_root = root / "dataset"
    generate_phase4_dataset(dataset_root, profile_name="ci")
    dataset = Phase4Dataset(dataset_root)
    spec = load_phase6_spec()
    study = train_phase6_study(dataset, spec=spec, profile_name="ci")
    checkpoint_root = root / "checkpoint"
    save_phase6_checkpoint(
        checkpoint_root,
        dataset=dataset,
        study=study,
        spec=spec,
    )
    loaded = load_phase6_checkpoint(
        checkpoint_root,
        dataset=dataset,
        spec=spec,
    )
    return root, dataset, spec, loaded


def test_ci_training_checkpoint_passes_all_phase6_guards(trained_checkpoint):
    root, dataset, spec, checkpoint = trained_checkpoint

    report = run_phase6_validation(dataset, checkpoint, spec=spec)
    report_path, figure_path = generate_phase6_artifacts(
        report,
        checkpoint.training_history,
        root / "results",
    )

    assert report.passed is True
    assert report.phase7_gate_claimed is False
    assert {check.name for check in report.checks} == REQUIRED_PHASE6_CHECKS
    assert all(check.passed for check in report.checks)
    assert report.contractivity_margin > 0.0
    assert report.validation_free_rollout_nrmse >= 0.0
    assert json.loads(report_path.read_text())["passed"] is True
    assert figure_path.stat().st_size > 1_000


def test_checkpoint_content_tampering_is_rejected(trained_checkpoint):
    root, dataset, spec, _ = trained_checkpoint
    checkpoint_root = root / "checkpoint"
    history_path = checkpoint_root / "training_history.json"
    original = history_path.read_text(encoding="utf-8")
    history_path.write_text(original + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="content hash mismatch"):
        load_phase6_checkpoint(
            checkpoint_root,
            dataset=dataset,
            spec=spec,
        )

    history_path.write_text(original, encoding="utf-8")
