import json

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("optax")
pytest.importorskip("robustnn")

from r2dn_dc_motor.data import Phase4Dataset, generate_phase4_dataset
from r2dn_dc_motor.models.jax_runtime import inspect_jax_runtime
from r2dn_dc_motor.models.r2dn_adapter import OfficialR2DNAdapter, R2DNArchitecture
from r2dn_dc_motor.models.r2dn_phase6b import (
    load_phase6b_checkpoint,
    save_phase6b_checkpoint,
    train_phase6b_study,
)
from r2dn_dc_motor.models.r2dn_training import (
    save_phase6_checkpoint,
    train_phase6_study,
)
from r2dn_dc_motor.phase2_spec import load_phase2_spec
from r2dn_dc_motor.phase6_spec import load_phase6_spec
from r2dn_dc_motor.phase6b_spec import REQUIRED_PHASE6B_CHECKS, load_phase6b_spec
from r2dn_dc_motor.validation.phase6b import (
    generate_phase6b_artifacts,
    run_phase6b_validation,
)
from r2dn_dc_motor.validation.r2dn_rk4_benchmark import (
    build_benchmark_report,
    build_voltage_trace,
    generate_benchmark_artifacts,
    load_benchmark_anchor,
    resolve_scenario,
    run_r2dn_trace,
    run_rk4_trace,
)

pytestmark = pytest.mark.phase6b_gate


@pytest.mark.parametrize("latent_size", (10, 12, 16, 24))
def test_official_backend_executes_every_extended_latent_size(latent_size):
    phase6 = load_phase6_spec()
    adapter = OfficialR2DNAdapter(
        R2DNArchitecture(
            input_size=int(phase6.architecture["input_size"]),
            state_size=latent_size,
            features=int(phase6.architecture["feature_size"]),
            output_size=int(phase6.architecture["output_size"]),
            hidden=tuple(phase6.architecture["hidden_sizes"]),
            init_method=str(phase6.architecture["initialization"]),
            do_polar_param=True,
        )
    )
    parameters, state = adapter.initialize(seed=123, batch_size=2)
    observations = np.zeros((5, 2, 2), dtype=np.float32)
    controls = np.zeros((5, 2, 1), dtype=np.float32)

    burned_state, _ = adapter.burn_in(
        parameters,
        state,
        observations,
        controls,
    )
    final_state, predictions = adapter.free_rollout(
        parameters,
        burned_state,
        observations[-1],
        controls,
    )

    assert final_state.shape == (2, latent_size)
    assert predictions.shape == (5, 2, 2)
    assert np.isfinite(np.asarray(predictions)).all()
    assert adapter.contractivity_certificate_margin(parameters) > 0.0


def test_ci_search_checkpoint_and_stress_pipeline(tmp_path):
    dataset_root = tmp_path / "dataset"
    generate_phase4_dataset(dataset_root, profile_name="ci")
    dataset = Phase4Dataset(dataset_root)
    phase6 = load_phase6_spec()
    phase6b = load_phase6b_spec(phase6=phase6)
    reusable_phase6_root = tmp_path / "phase6-checkpoint"
    phase6_study = train_phase6_study(
        dataset,
        spec=phase6,
        profile_name="ci",
    )
    save_phase6_checkpoint(
        reusable_phase6_root,
        dataset=dataset,
        study=phase6_study,
        spec=phase6,
    )
    study = train_phase6b_study(
        dataset,
        phase6=phase6,
        phase6b=phase6b,
        profile_name="ci",
        cache_directory=tmp_path / "cache",
        reusable_phase6_checkpoint=reusable_phase6_root,
    )
    assert study.selected_run_source == "phase6_checkpoint_reuse"
    resumed = train_phase6b_study(
        dataset,
        phase6=phase6,
        phase6b=phase6b,
        profile_name="ci",
        cache_directory=tmp_path / "cache",
        reusable_phase6_checkpoint=reusable_phase6_root,
    )
    assert resumed.selected_latent_size == study.selected_latent_size
    assert [
        value.median_validation_free_rollout_nrmse
        for value in resumed.pilot_aggregates
    ] == [
        value.median_validation_free_rollout_nrmse
        for value in study.pilot_aggregates
    ]
    checkpoint_root = tmp_path / "checkpoint"
    save_phase6b_checkpoint(
        checkpoint_root,
        dataset=dataset,
        study=study,
        phase6=phase6,
        phase6b=phase6b,
    )
    checkpoint = load_phase6b_checkpoint(
        checkpoint_root,
        dataset=dataset,
        phase6=phase6,
        phase6b=phase6b,
    )
    report = run_phase6b_validation(
        dataset,
        checkpoint,
        phase6=phase6,
        phase6b=phase6b,
    )
    report_path, figure_path = generate_phase6b_artifacts(
        report,
        tmp_path / "results",
    )

    assert {value.name for value in report.checks} == REQUIRED_PHASE6B_CHECKS
    assert report.phase7_gate_claimed is False
    assert len(report.pilot_aggregates) == 2
    assert all(len(value["scores"]) == 3 for value in report.pilot_aggregates)
    assert len(report.replay_metrics) == 6
    assert len(report.stress_metrics) == 48
    assert json.loads(report_path.read_text())["selected_seed"] == 7
    assert figure_path.stat().st_size > 1_000

    phase2 = load_phase2_spec()
    scenario = resolve_scenario(phase6b, "multisine")
    duration_s = 0.02
    voltage = build_voltage_trace(
        scenario,
        duration_s=duration_s,
        control_period_s=phase2.integration_settings.control_period_s,
    )
    anchor, phase6b_payload = load_benchmark_anchor(
        dataset,
        report_path,
        split="validation",
        anchor_index=0,
        checkpoint=checkpoint,
    )
    r2dn_trace = run_r2dn_trace(
        checkpoint,
        anchor,
        voltage,
        duration_s=duration_s,
        chunk_steps=10,
    )
    rk4_trace = run_rk4_trace(
        phase2,
        anchor.initial_full_state,
        voltage,
        duration_s=duration_s,
    )
    benchmark = build_benchmark_report(
        dataset=dataset,
        checkpoint=checkpoint,
        phase6b_report=phase6b_payload,
        phase2=phase2,
        scenario=scenario,
        duration_s=duration_s,
        chunk_steps=10,
        anchor=anchor,
        r2dn=r2dn_trace,
        rk4=rk4_trace,
        runtime=inspect_jax_runtime(),
    )
    benchmark_json, benchmark_png = generate_benchmark_artifacts(
        benchmark,
        r2dn_observations=r2dn_trace.observations,
        rk4_observations=rk4_trace.observations,
        physical_voltages_v=voltage,
        output_directory=tmp_path / "phase6c-results",
        maximum_plot_points=100,
    )
    assert benchmark.passed
    assert benchmark.accuracy.compared_steps == 20
    assert json.loads(benchmark_json.read_text())["benchmark"] == (
        "phase6c_r2dn_vs_full_rk4"
    )
    assert benchmark_png.stat().st_size > 1_000

    history_path = checkpoint_root / "study_history.json"
    original = history_path.read_text(encoding="utf-8")
    history_path.write_text(original + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="content hash mismatch"):
        load_phase6b_checkpoint(
            checkpoint_root,
            dataset=dataset,
            phase6=phase6,
            phase6b=phase6b,
        )
    history_path.write_text(original, encoding="utf-8")
