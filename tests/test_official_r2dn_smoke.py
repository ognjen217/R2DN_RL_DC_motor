import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("robustnn")

from r2dn_dc_motor.models.r2dn_adapter import OfficialR2DNAdapter

pytestmark = pytest.mark.r2dn_integration


def test_official_backend_carries_state_through_burn_in_and_free_rollout():
    adapter = OfficialR2DNAdapter()
    parameters, initial_state = adapter.initialize(seed=7, batch_size=2)

    measured_observations = np.zeros((5, 2, 2), dtype=np.float32)
    measured_observations[..., 0] = np.linspace(0.0, 0.4, 5)[:, None]
    applied_controls = np.full((5, 2, 1), 0.1, dtype=np.float32)

    burned_state, burn_predictions = adapter.burn_in(
        parameters,
        initial_state,
        measured_observations,
        applied_controls,
    )
    final_state, predictions = adapter.free_rollout(
        parameters,
        burned_state,
        measured_observations[-1],
        np.full((32, 2, 1), 0.1, dtype=np.float32),
    )

    assert initial_state.shape == burned_state.shape == final_state.shape == (2, 4)
    assert burn_predictions.shape == (5, 2, 2)
    assert predictions.shape == (32, 2, 2)
    assert np.isfinite(np.asarray(burned_state)).all()
    assert np.isfinite(np.asarray(predictions)).all()
    assert not np.allclose(np.asarray(initial_state), np.asarray(burned_state))


def test_official_backend_satisfies_equation_20_parameterization_test():
    adapter = OfficialR2DNAdapter()
    parameters, _ = adapter.initialize(seed=11, batch_size=1)

    margin = adapter.contractivity_certificate_margin(parameters)

    assert np.isfinite(margin)
    assert margin > 0.0
