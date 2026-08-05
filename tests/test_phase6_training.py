import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("optax")
pytest.importorskip("robustnn")

import r2dn_dc_motor.models.r2dn_training as r2dn_training
from r2dn_dc_motor.data import Phase4Dataset, generate_phase4_dataset
from r2dn_dc_motor.models import (
    load_phase6_checkpoint,
    save_phase6_checkpoint,
    train_phase6_study,
)
from r2dn_dc_motor.models.r2dn_adapter import OfficialR2DNAdapter, R2DNArchitecture
from r2dn_dc_motor.models.r2dn_training import _make_train_step, train_r2dn_run
from r2dn_dc_motor.phase6_spec import (
    REQUIRED_PHASE6_CHECKS,
    CurriculumStage,
    load_phase6_spec,
)
from r2dn_dc_motor.validation.phase6 import (
    generate_phase6_artifacts,
    run_phase6_validation,
)

pytestmark = pytest.mark.phase6_gate


def test_nonfinite_minibatch_is_rejected_without_poisoning_optimizer_state():
    import jax
    import jax.numpy as jnp
    import optax

    spec = load_phase6_spec()
    architecture = R2DNArchitecture(
        input_size=int(spec.architecture["input_size"]),
        state_size=4,
        features=int(spec.architecture["feature_size"]),
        output_size=int(spec.architecture["output_size"]),
        hidden=tuple(int(value) for value in spec.architecture["hidden_sizes"]),
        init_method=str(spec.architecture["initialization"]),
        do_polar_param=True,
    )
    adapter = OfficialR2DNAdapter(architecture)
    parameters, _ = adapter.initialize(seed=123, batch_size=2)
    optimizer = optax.chain(
        optax.clip_by_global_norm(float(spec.optimizer["gradient_clip_norm"])),
        optax.adamw(
            learning_rate=float(spec.optimizer["learning_rate"]),
            weight_decay=float(spec.optimizer["weight_decay"]),
        ),
    )
    optimizer_state = optimizer.init(parameters)
    stage = CurriculumStage(
        name="nonfinite_guard_probe",
        updates=1,
        rollout_steps=2,
        batch_size=2,
        one_step_weight=0.25,
        rollout_weight=0.75,
        reconstruction_weight=0.02,
    )
    train_step = _make_train_step(
        adapter,
        optimizer,
        burn_in_steps=2,
        stage=stage,
        jax=jax,
        jnp=jnp,
        optax=optax,
    )
    observations = jnp.full((5, 2, 2), jnp.nan, dtype=jnp.float32)
    controls = jnp.zeros((4, 2, 1), dtype=jnp.float32)
    parameter_leaves_before = [
        np.asarray(value).copy() for value in jax.tree_util.tree_leaves(parameters)
    ]
    optimizer_leaves_before = [
        np.asarray(value).copy()
        for value in jax.tree_util.tree_leaves(optimizer_state)
    ]

    next_parameters, next_optimizer_state, metrics, applied = train_step(
        parameters,
        optimizer_state,
        observations,
        controls,
    )

    assert bool(np.asarray(applied)) is False
    assert not np.isfinite(np.asarray(metrics, dtype=np.float64)).all()
    for before, after in zip(
        parameter_leaves_before,
        jax.tree_util.tree_leaves(next_parameters),
        strict=True,
    ):
        np.testing.assert_array_equal(before, np.asarray(after))
    for before, after in zip(
        optimizer_leaves_before,
        jax.tree_util.tree_leaves(next_optimizer_state),
        strict=True,
    ):
        np.testing.assert_array_equal(before, np.asarray(after))


def test_run_retries_rejected_batch_and_counts_only_finite_updates(
    tmp_path,
    monkeypatch,
):
    dataset_root = tmp_path / "dataset"
    generate_phase4_dataset(dataset_root, profile_name="ci")
    dataset = Phase4Dataset(dataset_root)
    spec = load_phase6_spec()
    profile = spec.profile("ci")
    original_sample = r2dn_training.R2DNWindowSampler.sample
    injected = False

    def sample_with_one_nonfinite_window(sampler, **kwargs):
        nonlocal injected
        window = original_sample(sampler, **kwargs)
        if sampler.split != "train" or injected:
            return window
        injected = True
        observations = np.full_like(window.observations, np.nan)
        return SimpleNamespace(
            observations=observations,
            controls=window.controls,
        )

    monkeypatch.setattr(
        r2dn_training.R2DNWindowSampler,
        "sample",
        sample_with_one_nonfinite_window,
    )
    run = train_r2dn_run(
        dataset,
        spec=spec,
        profile=profile,
        latent_size=4,
        seed=7,
        stages=profile.final_stages,
        validation_horizon_steps=profile.selection_horizon_steps,
        validation_window_seed=profile.selection_validation_seed,
        role="finite_retry_probe",
    )

    assert injected is True
    assert run.update_count == sum(stage.updates for stage in profile.final_stages)
    assert run.nonfinite_batch_retries == 1
    assert run.summary()["nonfinite_batch_retries"] == 1
    assert run.validation.finite is True


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
