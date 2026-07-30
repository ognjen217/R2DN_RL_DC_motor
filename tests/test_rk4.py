import math

import numpy as np
import pytest

from r2dn_dc_motor.numerics import integrate_interval, rk4_step


def test_rk4_matches_scalar_exponential():
    state = integrate_interval(
        lambda time_s, value: -2.0 * value,
        np.asarray([1.0]),
        start_time_s=0.0,
        duration_s=1.0,
        step_s=0.01,
    )

    assert state[0] == pytest.approx(math.exp(-2.0), abs=4e-10)


def test_halving_rk4_step_reduces_error_by_about_fourth_order_factor():
    exact = math.exp(-2.0)
    coarse = integrate_interval(
        lambda time_s, value: -2.0 * value,
        np.asarray([1.0]),
        start_time_s=0.0,
        duration_s=1.0,
        step_s=0.1,
    )
    fine = integrate_interval(
        lambda time_s, value: -2.0 * value,
        np.asarray([1.0]),
        start_time_s=0.0,
        duration_s=1.0,
        step_s=0.05,
    )

    assert abs(coarse[0] - exact) / abs(fine[0] - exact) > 10.0


def test_rk4_rejects_nonpositive_step():
    with pytest.raises(ValueError, match="positive"):
        rk4_step(
            lambda time_s, value: value,
            0.0,
            np.asarray([1.0]),
            0.0,
        )


def test_integrator_rejects_fractional_step_count():
    with pytest.raises(ValueError, match="integer number"):
        integrate_interval(
            lambda time_s, value: value,
            np.asarray([1.0]),
            start_time_s=0.0,
            duration_s=1.0,
            step_s=0.3,
        )
