import numpy as np
import pytest

from r2dn_dc_motor.data.sequences import (
    FullTrajectoryBatch,
    ModelSequenceBatch,
    SequenceValidationError,
)


def test_training_pairs_have_frozen_timing_and_shape():
    observations = np.array(
        [
            [[0.0, 10.0]],
            [[1.0, 11.0]],
            [[2.0, 12.0]],
        ]
    )
    controls = np.array([[[20.0]], [[30.0]]])
    batch = ModelSequenceBatch(observations, controls)

    regressors, targets = batch.training_pairs()

    np.testing.assert_array_equal(
        regressors,
        np.array([[[0.0, 10.0, 20.0]], [[1.0, 11.0, 30.0]]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        targets,
        np.array([[[1.0, 11.0]], [[2.0, 12.0]]], dtype=np.float32),
    )
    assert regressors.shape == (2, 1, 3)
    assert targets.shape == (2, 1, 2)


def test_full_trajectory_model_view_drops_temperature():
    full_states = np.array(
        [
            [[0.0, 10.0, 25.0]],
            [[1.0, 11.0, 30.0]],
        ]
    )
    controls = np.array([[[20.0]]])
    full = FullTrajectoryBatch(full_states, controls)

    model_view = full.model_view()

    assert full.temperature.shape == (2, 1, 1)
    assert not hasattr(model_view, "temperature")
    np.testing.assert_array_equal(model_view.observations, full_states[..., :2])


def test_burn_in_returns_measured_history_and_last_measured_rollout_seed():
    observations = np.arange(16, dtype=np.float32).reshape(4, 2, 2)
    controls = np.arange(6, dtype=np.float32).reshape(3, 2, 1)
    batch = ModelSequenceBatch(observations, controls)

    burn_observations, burn_controls, rollout_seed = batch.burn_in_inputs(2)

    np.testing.assert_array_equal(burn_observations, observations[:2])
    np.testing.assert_array_equal(burn_controls, controls[:2])
    np.testing.assert_array_equal(rollout_seed, observations[2])


@pytest.mark.parametrize(
    ("observations", "controls", "message"),
    [
        (
            np.zeros((2, 1, 2)),
            np.zeros((2, 1, 1)),
            "exactly one more",
        ),
        (
            np.zeros((2, 1, 3)),
            np.zeros((1, 1, 1)),
            "dimension must be 2",
        ),
        (
            np.zeros((2, 1, 2)),
            np.zeros((1, 2, 1)),
            "batch dimensions must match",
        ),
    ],
)
def test_invalid_model_sequence_shapes_are_rejected(observations, controls, message):
    with pytest.raises(SequenceValidationError, match=message):
        ModelSequenceBatch(observations, controls)


def test_non_finite_model_data_is_rejected():
    observations = np.zeros((2, 1, 2))
    observations[1, 0, 0] = np.nan

    with pytest.raises(SequenceValidationError, match="NaN"):
        ModelSequenceBatch(observations, np.zeros((1, 1, 1)))


def test_invalid_burn_in_length_is_rejected():
    batch = ModelSequenceBatch(
        observations=np.zeros((3, 1, 2)),
        controls=np.zeros((2, 1, 1)),
    )

    with pytest.raises(SequenceValidationError, match="at least one"):
        batch.burn_in_inputs(0)
    with pytest.raises(SequenceValidationError, match="longer"):
        batch.burn_in_inputs(3)
