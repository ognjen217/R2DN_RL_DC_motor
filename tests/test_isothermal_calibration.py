from dataclasses import replace

import numpy as np
import pytest

from r2dn_dc_motor.data import ModelSequenceBatch
from r2dn_dc_motor.models import (
    IsothermalCalibrationCheckpoint,
    fit_global_isothermal_parameters,
    nominal_isothermal_parameters,
)
from r2dn_dc_motor.phase2_spec import load_phase2_spec
from r2dn_dc_motor.phase5_spec import load_phase5_spec
from r2dn_dc_motor.plants import IsothermalWorldModel


class TemperatureGuardDataset:
    """Minimal dataset double exposing only the frozen model-view method."""

    fingerprint = "synthetic-temperature-free-dataset"

    def __init__(self, view):
        self._view = view

    def trajectory_ids(self, split=None):
        if split in (None, "train"):
            return ("train-synthetic-0000",)
        return ()

    def model_view(self, trajectory_id):
        assert trajectory_id == "train-synthetic-0000"
        return self._view

    def load_trajectory(self, _trajectory_id):
        raise AssertionError("calibration attempted to access raw temperature/load")

    def record(self, _trajectory_id):
        raise AssertionError("calibration attempted to access evaluation metadata")


def _synthetic_dataset(phase2):
    true_parameters = replace(
        nominal_isothermal_parameters(phase2),
        effective_resistance_ohm=1.65,
        viscous_friction_n_m_s_per_rad=0.0035,
    )
    model = IsothermalWorldModel(
        true_parameters,
        phase2.motor_limits,
        phase2.integration_settings,
        name="ISO-CAL",
    )
    rng = np.random.default_rng(4105)
    controls = rng.uniform(-16.0, 16.0, size=(20_000, 1))
    observations = model.free_rollout(np.asarray([0.2, -2.0]), controls)
    return (
        TemperatureGuardDataset(
            ModelSequenceBatch(
                observations=observations[:, None, :],
                controls=controls[:, None, :],
            )
        ),
        true_parameters,
    )


def test_global_fit_recovers_temperature_free_synthetic_parameters():
    phase2 = load_phase2_spec()
    phase5 = load_phase5_spec(phase2=phase2)
    dataset, true_parameters = _synthetic_dataset(phase2)

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

    assert checkpoint.parameters.effective_resistance_ohm == pytest.approx(
        true_parameters.effective_resistance_ohm,
        rel=2e-4,
    )
    assert (
        checkpoint.parameters.viscous_friction_n_m_s_per_rad
        == pytest.approx(
            true_parameters.viscous_friction_n_m_s_per_rad,
            rel=1e-3,
        )
    )
    assert checkpoint.temperature_used is False
    assert checkpoint.load_torque_used is False
    assert checkpoint.sufficient_statistics.transition_count == 20_000


def test_checkpoint_round_trip_is_dataset_bound(tmp_path):
    phase2 = load_phase2_spec()
    phase5 = load_phase5_spec(phase2=phase2)
    dataset, _ = _synthetic_dataset(phase2)
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
    path = tmp_path / "iso_cal.json"
    checkpoint.save(path)

    loaded = IsothermalCalibrationCheckpoint.load(path, dataset=dataset)

    assert loaded == checkpoint
    wrong_dataset = TemperatureGuardDataset(dataset._view)
    wrong_dataset.fingerprint = "different"
    with pytest.raises(ValueError, match="fingerprint"):
        IsothermalCalibrationCheckpoint.load(path, dataset=wrong_dataset)
