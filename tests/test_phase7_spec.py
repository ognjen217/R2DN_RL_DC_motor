import tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from r2dn_dc_motor.models.r2dn_phase7 import (
    Phase7VariantAggregate,
    _select_variant,
)
from r2dn_dc_motor.models.r2dn_training import (
    R2DNTrainingObjective,
    compute_train_increment_std,
)
from r2dn_dc_motor.phase2_spec import load_phase2_spec
from r2dn_dc_motor.phase4_spec import load_phase4_spec
from r2dn_dc_motor.phase7_spec import Phase7Spec, load_phase7_spec
from r2dn_dc_motor.spec import SpecValidationError


def test_broadband_dataset_and_phase7_specs_lock_the_hidden_thermal_protocol():
    phase2 = load_phase2_spec()
    broadband = load_phase4_spec(
        Path("configs/phase4_broadband.toml"),
        phase2=phase2,
    )
    phase7 = load_phase7_spec()

    assert broadband.dataset["version"] == "2.0.0"
    assert broadband.excitation["multisine_frequency_hz"] == [0.05, 4.0]
    assert broadband.excitation["heating_cooling_reheat"] is True
    assert broadband.profile("final").default_duration_s == 60.0
    assert broadband.profile("final").heating_cooling_duration_s == 120.0
    assert phase7.training["model_type"] == "ContractingR2DN"
    assert phase7.training["absolute_state_output_retained"] is True
    assert phase7.interface["hidden_evaluation_only"] == ["winding_temperature_c"]
    assert tuple(value.name for value in phase7.variants) == (
        "broadband_standard",
        "broadband_delta_multiscale",
        "broadband_delta_multiscale_wide",
    )


def test_phase7_rejects_temperature_as_a_training_feature():
    raw = tomllib.loads(Path("configs/phase7.toml").read_text(encoding="utf-8"))
    raw["interface"]["observation"].append("winding_temperature_c")

    with pytest.raises(SpecValidationError, match="observations"):
        Phase7Spec.from_dict(raw)


def test_increment_scale_uses_only_train_model_views():
    train_observations = np.asarray(
        [[0.0, 0.0], [1.0, 2.0], [3.0, 6.0]],
        dtype=np.float64,
    )
    validation_observations = np.asarray(
        [[0.0, 0.0], [1000.0, 1000.0]],
        dtype=np.float64,
    )
    views = {
        "train-a": SimpleNamespace(observations=train_observations[:, None, :]),
        "validation-a": SimpleNamespace(
            observations=validation_observations[:, None, :]
        ),
    }
    dataset = SimpleNamespace(
        normalization=SimpleNamespace(observation_std=np.asarray([2.0, 4.0])),
        trajectory_ids=lambda split: ("train-a",) if split == "train" else ("validation-a",),
        model_view=lambda trajectory_id: views[trajectory_id],
    )

    scale = compute_train_increment_std(dataset)

    expected = np.std(np.asarray([[0.5, 0.5], [1.0, 1.0]]), axis=0)
    np.testing.assert_allclose(scale, expected)


def test_phase7_objective_builds_cumulative_prefixes_without_changing_output_mode():
    objective = R2DNTrainingObjective(
        delta_weight=0.01,
        delta_std_normalized=(0.2, 0.1),
        rollout_horizon_weights=((1, 0.1), (10, 0.2), (100, 0.3), (1000, 0.4)),
    )

    objective.validate()

    assert objective.active_rollout_horizons(100) == (
        (1, 0.1),
        (10, 0.2),
        (100, 0.3),
    )
    assert objective.active_rollout_horizons(500) == (
        (1, 0.1),
        (10, 0.2),
        (100, 0.3),
        (500, 0.3),
    )


def test_phase7_profile_remains_immutable_when_inspected():
    phase7 = load_phase7_spec()
    profile = phase7.profile("final")

    modified = replace(profile, burn_in_steps=1000)

    assert modified.burn_in_steps == 1000
    assert phase7.profile("final").burn_in_steps == 250


def test_phase7_selection_keeps_simpler_ablation_inside_three_percent():
    phase7 = load_phase7_spec()
    aggregates = [
        Phase7VariantAggregate(name, (17, 29, 43), (score,) * 3, score)
        for name, score in (
            ("broadband_standard", 0.101),
            ("broadband_delta_multiscale", 0.100),
            ("broadband_delta_multiscale_wide", 0.099),
        )
    ]

    selected = _select_variant(phase7, aggregates, relative_tolerance=0.03)

    assert selected == "broadband_standard"
