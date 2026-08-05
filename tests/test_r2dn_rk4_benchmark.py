import json
from types import SimpleNamespace

import numpy as np
import pytest

import r2dn_dc_motor.compare_r2dn_rk4 as benchmark_cli
from r2dn_dc_motor.compare_r2dn_rk4 import (
    build_parser,
    load_benchmark_checkpoint,
)
from r2dn_dc_motor.data import NormalizationStatistics
from r2dn_dc_motor.phase2_spec import load_phase2_spec
from r2dn_dc_motor.phase6b_spec import load_phase6b_spec
from r2dn_dc_motor.validation.r2dn_rk4_benchmark import (
    build_voltage_trace,
    calculate_accuracy,
    load_benchmark_anchor,
    resolve_scenario,
    run_rk4_trace,
)


def test_cli_defaults_to_one_thousand_second_multisine():
    args = build_parser().parse_args([])

    assert args.duration_s == 1000.0
    assert args.scenario == "multisine"
    assert args.split == "validation"
    assert args.anchor_index == 0
    assert args.chunk_steps == 10_000


def test_benchmark_loader_accepts_screen_selected_phase6_checkpoint(
    tmp_path,
    monkeypatch,
):
    checkpoint_directory = tmp_path / "phase6"
    checkpoint_directory.mkdir()
    (checkpoint_directory / "manifest.json").write_text(
        json.dumps({"phase": 6}),
        encoding="utf-8",
    )
    expected = object()

    def fake_load(path, *, dataset, spec):
        assert path == checkpoint_directory
        assert dataset == "dataset"
        assert spec == "phase6"
        return expected

    monkeypatch.setattr(benchmark_cli, "load_phase6_checkpoint", fake_load)

    checkpoint, require_match = load_benchmark_checkpoint(
        checkpoint_directory,
        dataset="dataset",
        phase6="phase6",
        phase6b="phase6b",
        phase6d="phase6d",
    )

    assert checkpoint is expected
    assert require_match is False


def test_benchmark_loader_accepts_phase6e_checkpoint(tmp_path, monkeypatch):
    checkpoint_directory = tmp_path / "phase6e"
    checkpoint_directory.mkdir()
    (checkpoint_directory / "manifest.json").write_text(
        json.dumps({"phase": "6E"}),
        encoding="utf-8",
    )
    expected = object()

    def fake_spec(*, phase6, phase6b):
        assert phase6 == "phase6"
        assert phase6b == "phase6b"
        return "phase6e"

    def fake_load(path, *, dataset, phase6e, phase6):
        assert path == checkpoint_directory
        assert dataset == "dataset"
        assert phase6e == "phase6e"
        assert phase6 == "phase6"
        return expected

    monkeypatch.setattr(benchmark_cli, "load_phase6e_spec", fake_spec)
    monkeypatch.setattr(benchmark_cli, "load_phase6e_checkpoint", fake_load)

    checkpoint, require_match = load_benchmark_checkpoint(
        checkpoint_directory,
        dataset="dataset",
        phase6="phase6",
        phase6b="phase6b",
        phase6d="phase6d",
    )

    assert checkpoint is expected
    assert require_match is False


def test_benchmark_loader_accepts_phase6f_checkpoint(tmp_path, monkeypatch):
    checkpoint_directory = tmp_path / "phase6f"
    checkpoint_directory.mkdir()
    (checkpoint_directory / "manifest.json").write_text(
        json.dumps({"phase": "6F"}),
        encoding="utf-8",
    )
    expected = object()

    monkeypatch.setattr(
        benchmark_cli,
        "load_phase6e_spec",
        lambda *, phase6, phase6b: "phase6e",
    )
    monkeypatch.setattr(
        benchmark_cli,
        "load_phase6f_spec",
        lambda *, phase6, phase6b, phase6e: "phase6f",
    )

    def fake_load(path, *, dataset, phase6f, phase6e, phase6):
        assert path == checkpoint_directory
        assert dataset == "dataset"
        assert phase6f == "phase6f"
        assert phase6e == "phase6e"
        assert phase6 == "phase6"
        return expected

    monkeypatch.setattr(benchmark_cli, "load_phase6f_checkpoint", fake_load)

    checkpoint, require_match = load_benchmark_checkpoint(
        checkpoint_directory,
        dataset="dataset",
        phase6="phase6",
        phase6b="phase6b",
        phase6d="phase6d",
    )

    assert checkpoint is expected
    assert require_match is False


def test_benchmark_loader_rejects_unknown_checkpoint_phase(tmp_path):
    checkpoint_directory = tmp_path / "unknown"
    checkpoint_directory.mkdir()
    (checkpoint_directory / "manifest.json").write_text(
        json.dumps({"phase": "other"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected 6, 6B, 6D, 6E, or 6F"):
        load_benchmark_checkpoint(
            checkpoint_directory,
            dataset="dataset",
            phase6="phase6",
            phase6b="phase6b",
            phase6d="phase6d",
        )


def test_voltage_trace_has_exact_million_step_horizon():
    phase6b = load_phase6b_spec()
    scenario = resolve_scenario(phase6b, "multisine")

    voltage = build_voltage_trace(
        scenario,
        duration_s=1000.0,
        control_period_s=0.001,
    )

    assert voltage.shape == (1_000_000,)
    assert np.isfinite(voltage).all()
    assert np.max(np.abs(voltage)) <= 15.0 + 1e-12


def test_accuracy_uses_train_normalized_feature_scales():
    reference = np.zeros((1000, 2), dtype=np.float64)
    prediction = np.column_stack(
        (
            np.full(1000, 2.0),
            np.full(1000, -10.0),
        )
    )

    metrics = calculate_accuracy(
        prediction,
        reference,
        np.asarray((4.0, 20.0)),
        control_period_s=0.001,
        requested_horizons_s=(0.1, 1.0),
    )

    assert metrics.current_rmse_a == pytest.approx(2.0)
    assert metrics.speed_rmse_rad_s == pytest.approx(10.0)
    assert metrics.current_nrmse == pytest.approx(0.5)
    assert metrics.speed_nrmse == pytest.approx(0.5)
    assert metrics.combined_nrmse == pytest.approx(0.5)
    assert [value.horizon_steps for value in metrics.horizons] == [100, 1000]


def test_phase6b_report_reuses_exact_stress_anchor(tmp_path):
    normalization = NormalizationStatistics(
        observation_mean=np.asarray((1.0, 2.0)),
        observation_std=np.asarray((2.0, 4.0)),
        control_mean=np.asarray((0.5,)),
        control_std=np.asarray((2.0,)),
        observation_count=100,
        control_count=99,
    )
    states = np.column_stack(
        (
            np.arange(11, dtype=np.float64),
            2.0 * np.arange(11, dtype=np.float64),
            25.0 + np.arange(11, dtype=np.float64),
        )
    )
    trajectory = SimpleNamespace(
        states=states,
        applied_voltages=np.arange(10, dtype=np.float64)[:, None],
        transitions=10,
    )

    class FakeDataset:
        fingerprint = "dataset-hash"

        @staticmethod
        def load_trajectory(trajectory_id):
            assert trajectory_id == "validation-test-0001"
            return trajectory

    checkpoint = SimpleNamespace(
        normalization=normalization,
        manifest=SimpleNamespace(latent_size=4, seed=17),
    )
    report_path = tmp_path / "phase6b.json"
    report_path.write_text(
        json.dumps(
            {
                "passed": True,
                "dataset_fingerprint": "dataset-hash",
                "selected_latent_size": 4,
                "selected_seed": 17,
                "stress_anchors": [
                    {
                        "split": "validation",
                        "trajectory_id": "validation-test-0001",
                        "start_step": 3,
                        "burn_in_steps": 4,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    anchor, report = load_benchmark_anchor(
        FakeDataset(),
        report_path,
        split="validation",
        anchor_index=0,
        checkpoint=checkpoint,
    )

    assert report["passed"] is True
    assert anchor.provenance.start_step == 3
    assert anchor.initial_full_state.current_a == pytest.approx(7.0)
    assert anchor.initial_full_state.temperature_c == pytest.approx(32.0)
    assert anchor.burn_in_observations_normalized.shape == (4, 1, 2)
    assert anchor.burn_in_controls_normalized.shape == (4, 1, 1)
    assert anchor.initial_observation_normalized[0] == pytest.approx((3.0, 3.0))


def test_phase6d_longer_burn_in_preserves_same_initial_physical_state(tmp_path):
    normalization = NormalizationStatistics(
        observation_mean=np.zeros(2),
        observation_std=np.ones(2),
        control_mean=np.zeros(1),
        control_std=np.ones(1),
        observation_count=100,
        control_count=99,
    )
    states = np.column_stack(
        (
            np.arange(21, dtype=np.float64),
            2.0 * np.arange(21, dtype=np.float64),
            25.0 + np.arange(21, dtype=np.float64),
        )
    )
    trajectory = SimpleNamespace(
        states=states,
        applied_voltages=np.arange(20, dtype=np.float64)[:, None],
        transitions=20,
    )

    class FakeDataset:
        fingerprint = "dataset-hash"

        @staticmethod
        def load_trajectory(_trajectory_id):
            return trajectory

    checkpoint = SimpleNamespace(
        normalization=normalization,
        manifest=SimpleNamespace(latent_size=8, seed=29, burn_in_steps=10),
    )
    report_path = tmp_path / "phase6b.json"
    report_path.write_text(
        json.dumps(
            {
                "passed": True,
                "dataset_fingerprint": "dataset-hash",
                "selected_latent_size": 4,
                "selected_seed": 17,
                "stress_anchors": [
                    {
                        "split": "validation",
                        "trajectory_id": "validation-test-0001",
                        "start_step": 8,
                        "burn_in_steps": 4,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    anchor, _ = load_benchmark_anchor(
        FakeDataset(),
        report_path,
        split="validation",
        anchor_index=0,
        checkpoint=checkpoint,
        require_checkpoint_match=False,
    )

    assert anchor.provenance.start_step == 2
    assert anchor.provenance.burn_in_steps == 10
    assert anchor.initial_full_state.current_a == pytest.approx(12.0)
    assert anchor.burn_in_observations_normalized.shape == (10, 1, 2)


def test_canonical_rk4_trace_counts_locked_substeps():
    phase2 = load_phase2_spec()
    steps = 20
    duration = steps * phase2.integration_settings.control_period_s

    trace = run_rk4_trace(
        phase2,
        phase2.default_state,
        np.zeros(steps, dtype=np.float64),
        duration_s=duration,
    )

    assert trace.terminated is False
    assert trace.observations.shape == (steps, 2)
    assert trace.temperatures_c.shape == (steps,)
    assert trace.timing.control_steps_completed == steps
    assert trace.timing.rk4_substeps_per_control_step == 10
    assert trace.timing.rk4_substeps_completed == 200
