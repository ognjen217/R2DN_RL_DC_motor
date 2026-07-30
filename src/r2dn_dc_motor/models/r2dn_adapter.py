"""Adapter from the official ``ContractingR2DN`` API to the project contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class R2DNBackendUnavailable(ImportError):
    """Raised when the optional pinned JAX/R2DN backend is not installed."""


@dataclass(frozen=True)
class R2DNArchitecture:
    """Small Phase-1 architecture used only for integration smoke tests."""

    input_size: int = 3
    state_size: int = 4
    features: int = 8
    output_size: int = 2
    hidden: tuple[int, ...] = (8, 8)
    init_method: str = "long_memory"
    do_polar_param: bool = True

    def validate(self) -> None:
        errors: list[str] = []
        if self.input_size != 3:
            errors.append("R2DN regressor must contain [current, speed, voltage]")
        if self.output_size != 2:
            errors.append("R2DN output must contain [next current, next speed]")
        if self.state_size <= self.output_size:
            errors.append("latent state must be larger than the measured output")
        if self.features < 1 or not self.hidden or min(self.hidden) < 1:
            errors.append("all R2DN layer widths must be positive")
        if self.init_method not in {"random", "long_memory"}:
            errors.append("unsupported upstream initialization method")
        if not self.do_polar_param:
            errors.append("Phase-1 pilot must retain the upstream polar parameterization")
        if errors:
            raise ValueError("\n".join(errors))


class OfficialR2DNAdapter:
    """Observation-burn-in and autoregressive rollout around upstream R2DN.

    The official model receives regressors ``[y_k, u_k]`` and returns a latent
    state update plus the prediction ``y_{k+1}``. During burn-in, ``y_k`` is
    measured. During free rollout, ``y_k`` is the preceding model prediction.
    """

    def __init__(self, architecture: R2DNArchitecture | None = None) -> None:
        self.architecture = architecture or R2DNArchitecture()
        self.architecture.validate()

        try:
            import jax
            import jax.numpy as jnp
            from robustnn import r2dn
        except ImportError as error:
            raise R2DNBackendUnavailable(
                'Install the pinned backend with: pip install -e ".[r2dn]"'
            ) from error

        self._jax = jax
        self._jnp = jnp
        self.model = r2dn.ContractingR2DN(
            self.architecture.input_size,
            self.architecture.state_size,
            self.architecture.features,
            self.architecture.output_size,
            self.architecture.hidden,
            init_method=self.architecture.init_method,
            do_polar_param=self.architecture.do_polar_param,
        )
        self.audit_upstream_api()

    def audit_upstream_api(self) -> None:
        """Fail loudly if the pinned upstream class no longer matches the adapter."""

        if type(self.model).__name__ != "ContractingR2DN":
            raise TypeError("unexpected upstream R2DN class")
        required = {
            "initialize_carry",
            "simulate_sequence",
            "direct_to_explicit",
            "explicit_call",
        }
        missing = sorted(name for name in required if not callable(getattr(self.model, name, None)))
        if missing:
            raise TypeError(f"upstream R2DN API is missing: {', '.join(missing)}")

    def initialize(self, *, seed: int, batch_size: int) -> tuple[Any, Any]:
        """Create upstream parameters and a zero latent carry."""

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        key = self._jax.random.key(seed)
        carry_key, parameter_key = self._jax.random.split(key)
        state = self.model.initialize_carry(
            carry_key,
            (batch_size, self.architecture.input_size),
        )
        dummy_regressor = self._jnp.zeros(
            (batch_size, self.architecture.input_size),
            dtype=self._jnp.float32,
        )
        parameters = self.model.init(parameter_key, state, dummy_regressor)
        return parameters, state

    def step(
        self,
        parameters: Any,
        state: Any,
        observation: Any,
        control: Any,
    ) -> tuple[Any, Any]:
        """Predict one transition from ``[y_k, u_k]``."""

        self._validate_state_and_inputs(state, observation, control)
        regressor = self._jnp.concatenate(
            (
                self._jnp.asarray(observation, dtype=self._jnp.float32),
                self._jnp.asarray(control, dtype=self._jnp.float32),
            ),
            axis=-1,
        )
        return self.model.apply(parameters, state, regressor)

    def burn_in(
        self,
        parameters: Any,
        state: Any,
        measured_observations: Any,
        applied_controls: Any,
    ) -> tuple[Any, Any]:
        """Advance the latent state using measured ``[y_k, u_k]`` history."""

        observations = self._jnp.asarray(measured_observations, dtype=self._jnp.float32)
        controls = self._jnp.asarray(applied_controls, dtype=self._jnp.float32)
        self._validate_sequence_inputs(state, observations, controls)
        regressors = self._jnp.concatenate((observations, controls), axis=-1)
        return self.model.simulate_sequence(parameters, state, regressors)

    def free_rollout(
        self,
        parameters: Any,
        state: Any,
        initial_observation: Any,
        future_controls: Any,
    ) -> tuple[Any, Any]:
        """Run a free rollout without access to future measured observations."""

        observation = self._jnp.asarray(initial_observation, dtype=self._jnp.float32)
        controls = self._jnp.asarray(future_controls, dtype=self._jnp.float32)
        if controls.ndim != 3 or controls.shape[-1] != 1:
            raise ValueError("future_controls must have shape (H, B, 1)")
        if observation.ndim != 2 or observation.shape[-1] != 2:
            raise ValueError("initial_observation must have shape (B, 2)")
        if state.shape != (observation.shape[0], self.architecture.state_size):
            raise ValueError("state must have shape (B, latent_size)")
        if controls.shape[1] != observation.shape[0]:
            raise ValueError("rollout control and observation batch dimensions must match")
        if controls.shape[0] < 1:
            raise ValueError("free rollout horizon must be positive")

        explicit = self.model.direct_to_explicit(parameters)

        def rollout_step(carry: tuple[Any, Any], control: Any) -> tuple[tuple[Any, Any], Any]:
            latent_state, predicted_observation = carry
            regressor = self._jnp.concatenate((predicted_observation, control), axis=-1)
            next_state, next_observation = self.model.explicit_call(
                parameters,
                latent_state,
                regressor,
                explicit,
            )
            return (next_state, next_observation), next_observation

        (final_state, _), predictions = self._jax.lax.scan(
            rollout_step,
            (state, observation),
            controls,
        )
        return final_state, predictions

    def contractivity_certificate_margin(self, parameters: Any) -> float:
        """Return the smallest eigenvalue of the Equation-20 residual.

        The pinned implementation constructs

        ``H - diag(C1.T @ C1, B1 @ B1.T) = scaled(X.T @ X) + eps I``.

        A positive minimum eigenvalue verifies the strict matrix inequality used
        by the direct contracting parameterization in Equations 20--22 of the
        R2DN paper. It does not certify the outer autoregressive feedback loop.
        """

        direct = parameters["params"]
        x_matrix = direct["X"]
        polar_scale = direct["p"]
        epsilon = self._jnp.finfo(self._jnp.float32).eps
        gram = x_matrix.T @ x_matrix
        if self.architecture.do_polar_param:
            denominator = self._jnp.sum(x_matrix**2) + epsilon
            gram = polar_scale**2 * gram / denominator
        residual = gram + epsilon * self._jnp.identity(
            x_matrix.shape[0],
            dtype=x_matrix.dtype,
        )
        residual = (residual + residual.T) / 2
        return float(self._jnp.linalg.eigvalsh(residual).min())

    def _validate_state_and_inputs(
        self,
        state: Any,
        observation: Any,
        control: Any,
    ) -> None:
        state_shape = np.shape(state)
        observation_shape = np.shape(observation)
        control_shape = np.shape(control)
        if len(observation_shape) != 2 or observation_shape[-1] != 2:
            raise ValueError("observation must have shape (B, 2)")
        if len(control_shape) != 2 or control_shape[-1] != 1:
            raise ValueError("control must have shape (B, 1)")
        if state_shape != (observation_shape[0], self.architecture.state_size):
            raise ValueError("state must have shape (B, latent_size)")
        if control_shape[0] != observation_shape[0]:
            raise ValueError("control and observation batch dimensions must match")

    def _validate_sequence_inputs(
        self,
        state: Any,
        observations: Any,
        controls: Any,
    ) -> None:
        if observations.ndim != 3 or observations.shape[-1] != 2:
            raise ValueError("measured_observations must have shape (T, B, 2)")
        if controls.ndim != 3 or controls.shape[-1] != 1:
            raise ValueError("applied_controls must have shape (T, B, 1)")
        if observations.shape[:2] != controls.shape[:2]:
            raise ValueError("burn-in observation and control dimensions must match")
        if state.shape != (observations.shape[1], self.architecture.state_size):
            raise ValueError("state must have shape (B, latent_size)")
        if observations.shape[0] < 1:
            raise ValueError("burn-in must contain at least one transition")
