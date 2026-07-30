import numpy as np
import pytest

from r2dn_dc_motor.models import (
    HISTORY_FEATURE_NAMES,
    INSTANTANEOUS_FEATURE_NAMES,
    ProbeTrajectory,
    StandardizedRidge,
    build_probe_samples,
)
from r2dn_dc_motor.phase2_spec import load_phase2_spec
from r2dn_dc_motor.plants import MotorState


def test_standardized_ridge_recovers_linear_mapping():
    rng = np.random.default_rng(42)
    features = rng.normal(size=(200, 3))
    targets = 30.0 + features @ np.asarray([2.0, -3.0, 0.5])
    model = StandardizedRidge.fit(features, targets, l2=1e-8)

    prediction = model.predict(features)

    np.testing.assert_allclose(prediction, targets, atol=1e-7)


def test_history_feature_builder_uses_only_measured_dynamics():
    phase2 = load_phase2_spec()
    steps = 300
    controls = np.full(steps, 12.0)
    rollout = phase2.build_plant().rollout(
        controls,
        initial_state=MotorState(0.0, 0.0, 60.0),
        load_torques_n_m=0.0,
    )
    samples = build_probe_samples(
        (
            ProbeTrajectory(
                trajectory_id=7,
                states=rollout.states,
                controls_v=rollout.applied_voltages_v,
            ),
        ),
        parameters=phase2.motor_parameters,
        control_period_s=phase2.integration_settings.control_period_s,
        history_steps=50,
        sample_stride_steps=25,
        minimum_informative_current_a=0.5,
    )

    assert samples.history_features.shape[1] == len(HISTORY_FEATURE_NAMES)
    assert samples.instantaneous_features.shape[1] == len(
        INSTANTANEOUS_FEATURE_NAMES
    )
    assert set(samples.trajectory_ids) == {7}
    assert all("temperature" not in name for name in HISTORY_FEATURE_NAMES)
    assert all("temperature" not in name for name in INSTANTANEOUS_FEATURE_NAMES)


def test_probe_rejects_too_short_history():
    phase2 = load_phase2_spec()
    trajectory = ProbeTrajectory(
        trajectory_id=0,
        states=np.zeros((3, 3)),
        controls_v=np.zeros(2),
    )

    with pytest.raises(ValueError, match="at least two"):
        build_probe_samples(
            (trajectory,),
            parameters=phase2.motor_parameters,
            control_period_s=phase2.integration_settings.control_period_s,
            history_steps=1,
            sample_stride_steps=1,
            minimum_informative_current_a=0.5,
        )
