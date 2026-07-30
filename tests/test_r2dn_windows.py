import numpy as np
import pytest

from r2dn_dc_motor.data import (
    Phase4Dataset,
    R2DNWindowSampler,
    generate_phase4_dataset,
)


@pytest.mark.phase6_gate
def test_r2dn_windows_are_reproducible_normalized_and_temperature_free(tmp_path):
    dataset_root = tmp_path / "dataset"
    generate_phase4_dataset(dataset_root, profile_name="ci")
    dataset = Phase4Dataset(dataset_root)
    left = R2DNWindowSampler(dataset, split="train", seed=101)
    right = R2DNWindowSampler(dataset, split="train", seed=101)

    left_batch = left.sample(batch_size=4, burn_in_steps=20, rollout_steps=40)
    right_batch = right.sample(batch_size=4, burn_in_steps=20, rollout_steps=40)

    assert left_batch.split == "train"
    assert left_batch.observations.shape == (61, 4, 2)
    assert left_batch.controls.shape == (60, 4, 1)
    assert left_batch.trajectory_ids == right_batch.trajectory_ids
    assert left_batch.start_steps == right_batch.start_steps
    np.testing.assert_array_equal(left_batch.observations, right_batch.observations)
    np.testing.assert_array_equal(left_batch.controls, right_batch.controls)
    assert not hasattr(left_batch, "temperature")
    assert not hasattr(left_batch, "load_torque")


def test_r2dn_sampler_rejects_test_splits():
    with pytest.raises(ValueError, match="train or validation"):
        R2DNWindowSampler.__new__(R2DNWindowSampler).__init__(
            None,
            split="id_test",
            seed=0,
        )
