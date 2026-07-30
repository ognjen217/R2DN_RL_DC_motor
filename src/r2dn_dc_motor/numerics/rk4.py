"""Fixed-step classical fourth-order Runge--Kutta integration."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]
RightHandSide = Callable[[float, FloatArray], FloatArray]


def rk4_step(
    rhs: RightHandSide,
    time_s: float,
    state: FloatArray,
    step_s: float,
) -> FloatArray:
    """Advance one classical RK4 step.

    ``rhs`` must be a pure function of time and state. Constant inputs are bound
    by the caller, which keeps this integrator reusable for FULL and isothermal
    plant backends.
    """

    if not np.isfinite(step_s) or step_s <= 0.0:
        raise ValueError("RK4 step must be finite and positive")

    x = np.asarray(state, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("RK4 state must be a finite one-dimensional array")

    k1 = np.asarray(rhs(time_s, x), dtype=np.float64)
    k2 = np.asarray(rhs(time_s + step_s / 2.0, x + step_s * k1 / 2.0))
    k3 = np.asarray(rhs(time_s + step_s / 2.0, x + step_s * k2 / 2.0))
    k4 = np.asarray(rhs(time_s + step_s, x + step_s * k3))

    for slope in (k1, k2, k3, k4):
        if slope.shape != x.shape:
            raise ValueError("RK4 right-hand side changed the state shape")

    return x + step_s * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def integrate_interval(
    rhs: RightHandSide,
    initial_state: FloatArray,
    *,
    start_time_s: float,
    duration_s: float,
    step_s: float,
) -> FloatArray:
    """Integrate a fixed interval containing an integer number of RK4 steps."""

    if not np.isfinite(duration_s) or duration_s < 0.0:
        raise ValueError("integration duration must be finite and non-negative")
    if not np.isfinite(start_time_s):
        raise ValueError("integration start time must be finite")

    raw_steps = duration_s / step_s
    steps = int(round(raw_steps))
    if not np.isclose(raw_steps, steps, rtol=0.0, atol=1e-12):
        raise ValueError("integration duration must contain an integer number of steps")

    state = np.asarray(initial_state, dtype=np.float64).copy()
    time_s = float(start_time_s)
    for _ in range(steps):
        state = rk4_step(rhs, time_s, state, step_s)
        time_s += step_s
    return state
