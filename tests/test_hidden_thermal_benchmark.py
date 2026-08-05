from types import SimpleNamespace

import numpy as np
import pytest

from r2dn_dc_motor.compare_hidden_thermal_models import (
    LOCKED_HORIZONS_S,
    build_parser,
)
from r2dn_dc_motor.models import nominal_isothermal_parameters
from r2dn_dc_motor.phase2_spec import load_phase2_spec
from r2dn_dc_motor.plants import IsothermalWorldModel
from r2dn_dc_motor.validation.hidden_thermal_benchmark import (
    build_horizon_ranking,
    run_isothermal_trace,
)
from r2dn_dc_motor.validation.r2dn_rk4_benchmark import calculate_accuracy


def test_cli_locks_best_checkpoint_and_requested_horizons():
    args = build_parser().parse_args([])

    assert args.checkpoint_dir.as_posix() == "checkpoints/phase6e/r2dn-v1"
    assert args.iso_cal_checkpoint.as_posix() == "checkpoints/phase5/iso_cal.json"
    assert args.duration_s == 1000.0
    assert LOCKED_HORIZONS_S == (1.0, 10.0, 100.0, 1000.0)


def test_isothermal_trace_is_temperature_free_and_fully_autoregressive():
    phase2 = load_phase2_spec()
    model = IsothermalWorldModel(
        nominal_isothermal_parameters(phase2),
        phase2.motor_limits,
        phase2.integration_settings,
        name="ISO-NOM",
    )
    initial = np.asarray((1.2, -4.0), dtype=np.float64)
    voltages = np.linspace(-3.0, 5.0, 25)

    trace = run_isothermal_trace(
        model,
        initial,
        voltages,
        duration_s=0.025,
    )

    expected = model.free_rollout(initial, voltages[:, None])[1:]
    assert model.observation_names == ("armature_current_a", "angular_speed_rad_s")
    assert model.control_names == ("armature_voltage_v",)
    assert trace.observations.shape == (25, 2)
    np.testing.assert_allclose(trace.observations, expected, rtol=0.0, atol=0.0)


def test_horizon_ranking_uses_combined_nrmse_and_preserves_all_channels():
    reference = np.zeros((1000, 2), dtype=np.float64)
    scale = np.ones(2, dtype=np.float64)
    accuracy = {
        "R2DN": calculate_accuracy(
            np.full((1000, 2), 0.1),
            reference,
            scale,
            control_period_s=0.001,
            requested_horizons_s=(0.5, 1.0),
        ),
        "ISO-NOM": calculate_accuracy(
            np.full((1000, 2), 0.3),
            reference,
            scale,
            control_period_s=0.001,
            requested_horizons_s=(0.5, 1.0),
        ),
        "ISO-CAL": calculate_accuracy(
            np.full((1000, 2), 0.2),
            reference,
            scale,
            control_period_s=0.001,
            requested_horizons_s=(0.5, 1.0),
        ),
    }

    ranking = build_horizon_ranking(accuracy, (0.5, 1.0))

    assert [row["model"] for row in ranking["0.5s"]] == [
        "R2DN",
        "ISO-CAL",
        "ISO-NOM",
    ]
    assert ranking["1s"][0] == {
        "model": "R2DN",
        "combined_nrmse": pytest.approx(0.1),
        "current_nrmse": pytest.approx(0.1),
        "speed_nrmse": pytest.approx(0.1),
        "rank": 1,
    }


def test_isothermal_trace_rejects_temperature_in_initial_observation():
    fake_model = SimpleNamespace(name="ISO-NOM")

    with pytest.raises(ValueError, match="current and speed"):
        run_isothermal_trace(
            fake_model,
            np.asarray((1.0, 2.0, 80.0)),
            np.zeros(2),
            duration_s=0.002,
        )
