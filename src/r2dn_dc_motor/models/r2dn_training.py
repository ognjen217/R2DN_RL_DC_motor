"""Official-R2DN curriculum training and versioned Phase-6 checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from r2dn_dc_motor.data import (
    NormalizationStatistics,
    Phase4Dataset,
    R2DNWindowBatch,
    R2DNWindowSampler,
)
from r2dn_dc_motor.models.jax_runtime import inspect_jax_runtime
from r2dn_dc_motor.models.r2dn_adapter import OfficialR2DNAdapter, R2DNArchitecture
from r2dn_dc_motor.phase1_spec import load_phase1_spec
from r2dn_dc_motor.phase6_spec import (
    FORBIDDEN_TRAINING_FEATURES,
    CurriculumStage,
    Phase6Spec,
    TrainingProfile,
    load_phase6_spec,
)

ProgressCallback = Callable[[str], None]
MINIMUM_NONFINITE_RETRY_BUDGET = 10
NONFINITE_RETRY_BUDGET_FRACTION = 0.05
ALLOWED_TRAINING_FEATURES = (
    "armature_current_a",
    "angular_speed_rad_s",
    "armature_voltage_v",
)


@dataclass(frozen=True)
class ValidationRolloutMetrics:
    """Selection metrics measured on fixed held-out validation windows."""

    free_rollout_nrmse: float
    one_step_nrmse: float
    current_free_rollout_nrmse: float
    speed_free_rollout_nrmse: float
    maximum_absolute_normalized_prediction: float
    finite: bool
    horizon_steps: int
    windows: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class R2DNRunResult:
    """One trained parameter tree plus compact, serializable evidence."""

    parameters: Any
    architecture: R2DNArchitecture
    seed: int
    role: str
    update_count: int
    contractivity_margin: float
    validation: ValidationRolloutMetrics
    history: tuple[dict[str, Any], ...]

    @property
    def selection_score(self) -> float:
        return self.validation.free_rollout_nrmse

    @property
    def nonfinite_batch_retries(self) -> int:
        """Number of rejected minibatches before finite optimizer updates."""

        return max(
            (
                int(record.get("nonfinite_batch_retries", 0))
                for record in self.history
            ),
            default=0,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "seed": self.seed,
            "latent_size": self.architecture.state_size,
            "feature_size": self.architecture.features,
            "hidden_sizes": list(self.architecture.hidden),
            "update_count": self.update_count,
            "nonfinite_batch_retries": self.nonfinite_batch_retries,
            "contractivity_margin": self.contractivity_margin,
            "validation": self.validation.to_dict(),
        }


@dataclass(frozen=True)
class Phase6TrainingStudy:
    """Pilot architecture search followed by repeated final training."""

    profile_name: str
    dataset_fingerprint: str
    selected_latent_size: int
    selected_run: R2DNRunResult
    pilot_runs: tuple[R2DNRunResult, ...]
    final_runs: tuple[R2DNRunResult, ...]

    def history_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "phase": 6,
            "profile": self.profile_name,
            "dataset_fingerprint": self.dataset_fingerprint,
            "selection_metric": "validation_free_rollout_nrmse",
            "selection_mode": "min",
            "selected_latent_size": self.selected_latent_size,
            "selected_seed": self.selected_run.seed,
            "pilot_runs": [run.summary() for run in self.pilot_runs],
            "final_runs": [run.summary() for run in self.final_runs],
            "selected_run_history": list(self.selected_run.history),
        }


@dataclass(frozen=True)
class Phase6CheckpointManifest:
    """Metadata that binds trained parameters to code, data, and protocol."""

    schema_version: int
    phase: int
    model_type: str
    training_profile: str
    dataset_fingerprint: str
    train_trajectory_ids: tuple[str, ...]
    validation_trajectory_ids: tuple[str, ...]
    upstream_commit: str
    observation_features: tuple[str, ...]
    control_features: tuple[str, ...]
    forbidden_training_features: tuple[str, ...]
    latent_size: int
    feature_size: int
    hidden_sizes: tuple[int, ...]
    initialization: str
    polar_parameterization: bool
    seed: int
    burn_in_steps: int
    selection_horizon_steps: int
    validation_window_seed: int
    selection_metric: str
    update_count: int
    validation_free_rollout_nrmse: float
    contractivity_margin: float
    parameter_sha256: str
    normalization_sha256: str
    training_history_sha256: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Phase6CheckpointManifest:
        return cls(
            schema_version=int(raw["schema_version"]),
            phase=int(raw["phase"]),
            model_type=str(raw["model_type"]),
            training_profile=str(raw["training_profile"]),
            dataset_fingerprint=str(raw["dataset_fingerprint"]),
            train_trajectory_ids=tuple(raw["train_trajectory_ids"]),
            validation_trajectory_ids=tuple(raw["validation_trajectory_ids"]),
            upstream_commit=str(raw["upstream_commit"]),
            observation_features=tuple(raw["observation_features"]),
            control_features=tuple(raw["control_features"]),
            forbidden_training_features=tuple(raw["forbidden_training_features"]),
            latent_size=int(raw["latent_size"]),
            feature_size=int(raw["feature_size"]),
            hidden_sizes=tuple(int(value) for value in raw["hidden_sizes"]),
            initialization=str(raw["initialization"]),
            polar_parameterization=bool(raw["polar_parameterization"]),
            seed=int(raw["seed"]),
            burn_in_steps=int(raw["burn_in_steps"]),
            selection_horizon_steps=int(raw["selection_horizon_steps"]),
            validation_window_seed=int(raw["validation_window_seed"]),
            selection_metric=str(raw["selection_metric"]),
            update_count=int(raw["update_count"]),
            validation_free_rollout_nrmse=float(
                raw["validation_free_rollout_nrmse"]
            ),
            contractivity_margin=float(raw["contractivity_margin"]),
            parameter_sha256=str(raw["parameter_sha256"]),
            normalization_sha256=str(raw["normalization_sha256"]),
            training_history_sha256=str(raw["training_history_sha256"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(
        self,
        *,
        dataset: Phase4Dataset,
        spec: Phase6Spec,
    ) -> None:
        phase1 = load_phase1_spec()
        profile = spec.profile(self.training_profile)
        errors: list[str] = []
        if self.schema_version != 1 or self.phase != 6:
            errors.append("checkpoint schema or phase mismatch")
        if self.model_type != "ContractingR2DN":
            errors.append("checkpoint model is not ContractingR2DN")
        if self.dataset_fingerprint != dataset.fingerprint:
            errors.append("checkpoint belongs to a different Phase-4 dataset")
        if self.train_trajectory_ids != dataset.trajectory_ids("train"):
            errors.append("checkpoint did not use exactly the dataset train trajectories")
        if self.validation_trajectory_ids != dataset.trajectory_ids("validation"):
            errors.append("checkpoint validation trajectory catalog mismatch")
        if self.upstream_commit != phase1.upstream["commit"]:
            errors.append("checkpoint was not produced with the pinned upstream commit")
        if self.observation_features != tuple(spec.interface["observation"]):
            errors.append("checkpoint observation order mismatch")
        if self.control_features != tuple(spec.interface["control"]):
            errors.append("checkpoint control order mismatch")
        if self.forbidden_training_features != FORBIDDEN_TRAINING_FEATURES:
            errors.append("checkpoint forbidden-feature catalog mismatch")
        if self.latent_size not in profile.candidate_latent_sizes:
            errors.append("checkpoint latent size was not a profile candidate")
        if self.feature_size != int(spec.architecture["feature_size"]):
            errors.append("checkpoint feature size mismatch")
        if self.hidden_sizes != tuple(spec.architecture["hidden_sizes"]):
            errors.append("checkpoint hidden-size mismatch")
        if self.initialization != spec.architecture["initialization"]:
            errors.append("checkpoint initialization mismatch")
        if self.polar_parameterization is not True:
            errors.append("checkpoint disabled the contracting polar parameterization")
        if self.seed not in profile.training_seeds:
            errors.append("checkpoint seed is not a locked final-training seed")
        if self.burn_in_steps != profile.burn_in_steps:
            errors.append("checkpoint burn-in length mismatch")
        if self.selection_horizon_steps != profile.selection_horizon_steps:
            errors.append("checkpoint selection horizon mismatch")
        if self.validation_window_seed != profile.selection_validation_seed:
            errors.append("checkpoint validation-window seed mismatch")
        if self.selection_metric != spec.selection["metric"]:
            errors.append("checkpoint selection metric mismatch")
        if self.update_count != sum(stage.updates for stage in profile.final_stages):
            errors.append("checkpoint final-training update count mismatch")
        numeric = (
            self.validation_free_rollout_nrmse,
            self.contractivity_margin,
        )
        if not all(math.isfinite(value) for value in numeric):
            errors.append("checkpoint validation or certificate metric is non-finite")
        if self.validation_free_rollout_nrmse < 0.0:
            errors.append("checkpoint validation NRMSE must be non-negative")
        if self.contractivity_margin <= 0.0:
            errors.append("checkpoint contractivity certificate is not positive")
        for name, digest in (
            ("parameter", self.parameter_sha256),
            ("normalization", self.normalization_sha256),
            ("training history", self.training_history_sha256),
        ):
            if len(digest) != 64:
                errors.append(f"{name} SHA-256 is malformed")
        if errors:
            raise ValueError("\n".join(errors))


@dataclass(frozen=True)
class LoadedPhase6Checkpoint:
    """Validated parameters and their bound metadata."""

    adapter: OfficialR2DNAdapter
    parameters: Any
    manifest: Phase6CheckpointManifest
    normalization: NormalizationStatistics
    training_history: dict[str, Any]


def train_phase6_study(
    dataset: Phase4Dataset,
    *,
    spec: Phase6Spec | None = None,
    profile_name: str = "ci",
    progress: ProgressCallback | None = None,
) -> Phase6TrainingStudy:
    """Run pilot latent-size selection, then repeated final curriculum training."""

    spec = spec or load_phase6_spec()
    profile = spec.profile(profile_name)
    progress = progress or (lambda _: None)
    pilot_runs: list[R2DNRunResult] = []
    for latent_size in profile.candidate_latent_sizes:
        progress(
            f"pilot latent={latent_size}, seed={profile.pilot_seed}, "
            f"updates={sum(stage.updates for stage in profile.pilot_stages)}"
        )
        pilot_runs.append(
            _train_single_run(
                dataset,
                spec=spec,
                profile=profile,
                latent_size=latent_size,
                seed=profile.pilot_seed,
                stages=profile.pilot_stages,
                validation_horizon_steps=profile.pilot_validation_horizon_steps,
                validation_window_seed=profile.pilot_validation_seed,
                role="pilot",
                progress=progress,
            )
        )
    selected_pilot = min(pilot_runs, key=lambda run: run.selection_score)
    selected_latent_size = selected_pilot.architecture.state_size
    progress(
        f"pilot selected latent={selected_latent_size} "
        f"(validation NRMSE={selected_pilot.selection_score:.6g})"
    )

    final_runs: list[R2DNRunResult] = []
    for seed in profile.training_seeds:
        progress(
            f"final latent={selected_latent_size}, seed={seed}, "
            f"updates={sum(stage.updates for stage in profile.final_stages)}"
        )
        final_runs.append(
            _train_single_run(
                dataset,
                spec=spec,
                profile=profile,
                latent_size=selected_latent_size,
                seed=seed,
                stages=profile.final_stages,
                validation_horizon_steps=profile.selection_horizon_steps,
                validation_window_seed=profile.selection_validation_seed,
                role="final",
                progress=progress,
            )
        )
    selected_run = min(final_runs, key=lambda run: run.selection_score)
    progress(
        f"selected final seed={selected_run.seed}, latent={selected_latent_size}, "
        f"validation NRMSE={selected_run.selection_score:.6g}"
    )
    return Phase6TrainingStudy(
        profile_name=profile_name,
        dataset_fingerprint=dataset.fingerprint,
        selected_latent_size=selected_latent_size,
        selected_run=selected_run,
        pilot_runs=tuple(pilot_runs),
        final_runs=tuple(final_runs),
    )


def _train_single_run(
    dataset: Phase4Dataset,
    *,
    spec: Phase6Spec,
    profile: TrainingProfile,
    latent_size: int,
    seed: int,
    stages: Sequence[CurriculumStage],
    validation_horizon_steps: int,
    validation_window_seed: int,
    role: str,
    progress: ProgressCallback,
    optimizer_override: Mapping[str, Any] | None = None,
) -> R2DNRunResult:
    try:
        import jax
        import jax.numpy as jnp
        import optax
    except ImportError as error:
        raise ImportError(
            'Phase-6 training requires: python -m pip install -e ".[phase6]"'
        ) from error

    architecture = R2DNArchitecture(
        input_size=int(spec.architecture["input_size"]),
        state_size=latent_size,
        features=int(spec.architecture["feature_size"]),
        output_size=int(spec.architecture["output_size"]),
        hidden=tuple(int(value) for value in spec.architecture["hidden_sizes"]),
        init_method=str(spec.architecture["initialization"]),
        do_polar_param=bool(spec.architecture["polar_parameterization"]),
    )
    adapter = OfficialR2DNAdapter(architecture)
    parameters, _ = adapter.initialize(seed=seed, batch_size=1)
    optimizer_settings = dict(spec.optimizer)
    if optimizer_override is not None:
        optimizer_settings.update(optimizer_override)
    schedule_name = str(optimizer_settings.get("schedule", "constant"))
    initial_learning_rate = float(
        optimizer_settings.get(
            "initial_learning_rate",
            optimizer_settings["learning_rate"],
        )
    )
    final_learning_rate = float(
        optimizer_settings.get("final_learning_rate", initial_learning_rate)
    )
    if schedule_name == "constant":
        learning_rate: Any = initial_learning_rate
    elif schedule_name == "cosine_decay":
        total_scheduled_updates = sum(stage.updates for stage in stages)
        learning_rate = optax.cosine_decay_schedule(
            init_value=initial_learning_rate,
            decay_steps=max(total_scheduled_updates - 1, 1),
            alpha=final_learning_rate / initial_learning_rate,
        )
    else:
        raise ValueError(f"unsupported learning-rate schedule: {schedule_name}")
    optimizer = optax.chain(
        optax.clip_by_global_norm(float(optimizer_settings["gradient_clip_norm"])),
        optax.adamw(
            learning_rate=learning_rate,
            weight_decay=float(optimizer_settings["weight_decay"]),
        ),
    )
    optimizer_state = optimizer.init(parameters)
    train_sampler = R2DNWindowSampler(
        dataset,
        split="train",
        seed=seed + 100_000,
    )
    history: list[dict[str, Any]] = []
    total_updates = 0
    total_nonfinite_retries = 0

    for stage in stages:
        train_step = _make_train_step(
            adapter,
            optimizer,
            burn_in_steps=profile.burn_in_steps,
            stage=stage,
            jax=jax,
            jnp=jnp,
            optax=optax,
        )
        stage_updates = 0
        stage_nonfinite_retries = 0
        retry_budget = max(
            MINIMUM_NONFINITE_RETRY_BUDGET,
            math.ceil(stage.updates * NONFINITE_RETRY_BUDGET_FRACTION),
        )
        while stage_updates < stage.updates:
            # Execute at most one logging interval asynchronously. The device-side
            # finite guard prevents a bad long-rollout minibatch from modifying
            # either parameters or Adam state. The host synchronizes once per
            # interval, counts only applied updates, and deterministically samples
            # replacement windows for rejected batches.
            attempts = min(
                profile.history_log_interval,
                stage.updates - stage_updates,
            )
            interval_metrics: list[Any] = []
            interval_applied: list[Any] = []
            for _ in range(attempts):
                window = train_sampler.sample(
                    batch_size=stage.batch_size,
                    burn_in_steps=profile.burn_in_steps,
                    rollout_steps=stage.rollout_steps,
                )
                parameters, optimizer_state, metrics, update_applied = train_step(
                    parameters,
                    optimizer_state,
                    window.observations,
                    window.controls,
                )
                interval_metrics.append(metrics)
                interval_applied.append(update_applied)

            applied_mask = np.asarray(
                jax.device_get(jnp.stack(interval_applied)),
                dtype=bool,
            )
            metric_matrix = np.asarray(
                jax.device_get(
                    jnp.stack(
                        [jnp.stack(values) for values in interval_metrics],
                    )
                ),
                dtype=np.float64,
            )
            applied_count = int(np.count_nonzero(applied_mask))
            rejected_count = attempts - applied_count
            stage_updates += applied_count
            total_updates += applied_count
            stage_nonfinite_retries += rejected_count
            total_nonfinite_retries += rejected_count

            if stage_nonfinite_retries > retry_budget:
                raise FloatingPointError(
                    f"Phase-6 non-finite retry budget exceeded at {stage.name}: "
                    f"rejected={stage_nonfinite_retries}, budget={retry_budget}, "
                    f"applied={stage_updates}/{stage.updates}"
                )
            if applied_count == 0:
                progress(
                    f"{role} latent={latent_size} seed={seed} {stage.name}: "
                    f"rejected {rejected_count} non-finite minibatch(es), "
                    f"retrying ({stage_nonfinite_retries}/{retry_budget})"
                )
                continue

            last_finite = int(np.flatnonzero(applied_mask)[-1])
            metric_values = {
                name: float(value)
                for name, value in zip(
                    (
                        "total_loss",
                        "one_step_loss",
                        "rollout_loss",
                        "reconstruction_loss",
                        "gradient_norm",
                    ),
                    metric_matrix[last_finite],
                    strict=True,
                )
            }
            if not all(math.isfinite(value) for value in metric_values.values()):
                raise AssertionError("finite-guard accepted a non-finite update")
            record = {
                "stage": stage.name,
                "stage_update": stage_updates,
                "global_update": total_updates,
                "nonfinite_batch_retries": total_nonfinite_retries,
                **metric_values,
            }
            history.append(record)
            retry_suffix = (
                f", rejected_nonfinite={stage_nonfinite_retries}"
                if stage_nonfinite_retries
                else ""
            )
            progress(
                f"{role} latent={latent_size} seed={seed} "
                f"{stage.name} {stage_updates}/{stage.updates}: "
                f"loss={metric_values['total_loss']:.6g}{retry_suffix}"
            )

    validation_sampler = R2DNWindowSampler(
        dataset,
        split="validation",
        seed=seed + 200_000,
    )
    validation_window = validation_sampler.fixed_validation_windows(
        count=profile.validation_windows,
        burn_in_steps=profile.burn_in_steps,
        rollout_steps=validation_horizon_steps,
        seed=validation_window_seed,
    )
    validation = evaluate_validation_rollout(
        adapter,
        parameters,
        validation_window,
        burn_in_steps=profile.burn_in_steps,
    )
    margin = adapter.contractivity_certificate_margin(parameters)
    if not math.isfinite(margin) or margin <= 0.0:
        raise FloatingPointError("trained R2DN lost its contractivity certificate")
    if not validation.finite:
        raise FloatingPointError("trained R2DN produced a non-finite validation rollout")
    return R2DNRunResult(
        parameters=parameters,
        architecture=architecture,
        seed=seed,
        role=role,
        update_count=total_updates,
        contractivity_margin=margin,
        validation=validation,
        history=tuple(history),
    )


def train_r2dn_run(
    dataset: Phase4Dataset,
    *,
    spec: Phase6Spec,
    profile: TrainingProfile,
    latent_size: int,
    seed: int,
    stages: Sequence[CurriculumStage],
    validation_horizon_steps: int,
    validation_window_seed: int,
    role: str,
    progress: ProgressCallback | None = None,
) -> R2DNRunResult:
    """Train one auditable run using the frozen Phase-6 optimizer and loss.

    Phase 6B uses this narrow public entry point to repeat the exact Phase-6
    curriculum over a wider latent-size and seed catalog without changing the
    historical Phase-6 study or checkpoint contract.
    """

    return _train_single_run(
        dataset,
        spec=spec,
        profile=profile,
        latent_size=latent_size,
        seed=seed,
        stages=stages,
        validation_horizon_steps=validation_horizon_steps,
        validation_window_seed=validation_window_seed,
        role=role,
        progress=progress or (lambda _: None),
    )


def train_r2dn_run_with_optimizer(
    dataset: Phase4Dataset,
    *,
    spec: Phase6Spec,
    profile: TrainingProfile,
    latent_size: int,
    seed: int,
    stages: Sequence[CurriculumStage],
    validation_horizon_steps: int,
    validation_window_seed: int,
    role: str,
    optimizer: Mapping[str, Any],
    progress: ProgressCallback | None = None,
) -> R2DNRunResult:
    """Train one run with an explicitly audited optimizer ablation.

    Historical Phase-6/6B/6D/6E calls continue through :func:`train_r2dn_run`
    and therefore retain the frozen constant-learning-rate optimizer.
    """

    return _train_single_run(
        dataset,
        spec=spec,
        profile=profile,
        latent_size=latent_size,
        seed=seed,
        stages=stages,
        validation_horizon_steps=validation_horizon_steps,
        validation_window_seed=validation_window_seed,
        role=role,
        progress=progress or (lambda _: None),
        optimizer_override=optimizer,
    )


def _make_train_step(
    adapter: OfficialR2DNAdapter,
    optimizer: Any,
    *,
    burn_in_steps: int,
    stage: CurriculumStage,
    jax: Any,
    jnp: Any,
    optax: Any,
) -> Callable[..., Any]:
    one_weight = stage.one_step_weight
    rollout_weight = stage.rollout_weight
    reconstruction_weight = stage.reconstruction_weight
    batch_size = stage.batch_size
    latent_size = adapter.architecture.state_size

    def loss_function(
        parameters: Any,
        observations: Any,
        controls: Any,
    ) -> tuple[Any, tuple[Any, Any, Any]]:
        initial_state = jnp.zeros((batch_size, latent_size), dtype=jnp.float32)
        burned_state, burn_predictions = adapter.burn_in(
            parameters,
            initial_state,
            observations[:burn_in_steps],
            controls[:burn_in_steps],
        )
        targets = observations[
            burn_in_steps + 1 : burn_in_steps + stage.rollout_steps + 1
        ]

        teacher_regressors = jnp.concatenate(
            (
                observations[
                    burn_in_steps : burn_in_steps + stage.rollout_steps
                ],
                controls[burn_in_steps:],
            ),
            axis=-1,
        )
        _, one_step_predictions = adapter.model.simulate_sequence(
            parameters,
            burned_state,
            teacher_regressors,
        )
        one_step_loss = jnp.mean((one_step_predictions - targets) ** 2)

        if rollout_weight > 0.0:
            _, rollout_predictions = adapter.free_rollout(
                parameters,
                burned_state,
                observations[burn_in_steps],
                controls[burn_in_steps:],
            )
            rollout_loss = jnp.mean((rollout_predictions - targets) ** 2)
        else:
            rollout_loss = jnp.asarray(0.0, dtype=jnp.float32)

        if reconstruction_weight > 0.0:
            reconstruction_targets = observations[1 : burn_in_steps + 1]
            reconstruction_loss = jnp.mean(
                (burn_predictions - reconstruction_targets) ** 2
            )
        else:
            reconstruction_loss = jnp.asarray(0.0, dtype=jnp.float32)
        total = (
            one_weight * one_step_loss
            + rollout_weight * rollout_loss
            + reconstruction_weight * reconstruction_loss
        )
        return total, (one_step_loss, rollout_loss, reconstruction_loss)

    value_and_gradient = jax.value_and_grad(loss_function, has_aux=True)

    @jax.jit
    def train_step(
        parameters: Any,
        optimizer_state: Any,
        observations: Any,
        controls: Any,
    ) -> tuple[Any, Any, tuple[Any, Any, Any, Any, Any], Any]:
        (total_loss, components), gradients = value_and_gradient(
            parameters,
            observations,
            controls,
        )
        gradient_norm = optax.tree.norm(gradients)
        finite = jnp.all(
            jnp.isfinite(
                jnp.stack((total_loss, *components, gradient_norm)),
            )
        )
        for leaf in jax.tree_util.tree_leaves(gradients):
            finite = jnp.logical_and(finite, jnp.all(jnp.isfinite(leaf)))
        for leaf in jax.tree_util.tree_leaves(parameters):
            finite = jnp.logical_and(finite, jnp.all(jnp.isfinite(leaf)))
        for leaf in jax.tree_util.tree_leaves(optimizer_state):
            finite = jnp.logical_and(finite, jnp.all(jnp.isfinite(leaf)))

        updates, next_optimizer_state = optimizer.update(
            gradients,
            optimizer_state,
            parameters,
        )
        next_parameters = optax.apply_updates(parameters, updates)
        for leaf in jax.tree_util.tree_leaves(next_parameters):
            finite = jnp.logical_and(finite, jnp.all(jnp.isfinite(leaf)))
        for leaf in jax.tree_util.tree_leaves(next_optimizer_state):
            finite = jnp.logical_and(finite, jnp.all(jnp.isfinite(leaf)))

        parameters, optimizer_state = jax.lax.cond(
            finite,
            lambda _: (next_parameters, next_optimizer_state),
            lambda state: state,
            (parameters, optimizer_state),
        )
        return (
            parameters,
            optimizer_state,
            (total_loss, *components, gradient_norm),
            finite,
        )

    return train_step


def evaluate_validation_rollout(
    adapter: OfficialR2DNAdapter,
    parameters: Any,
    window: R2DNWindowBatch,
    *,
    burn_in_steps: int,
) -> ValidationRolloutMetrics:
    """Measure teacher-forced and autoregressive error on one fixed window batch."""

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as error:
        raise ImportError("validation requires the Phase-6 JAX dependencies") from error

    observations = jnp.asarray(window.observations, dtype=jnp.float32)
    controls = jnp.asarray(window.controls, dtype=jnp.float32)
    horizon = window.transitions - burn_in_steps
    if horizon < 1:
        raise ValueError("validation window must extend beyond burn-in")
    batch_size = window.batch_size
    latent_size = adapter.architecture.state_size

    @jax.jit
    def evaluate(parameters: Any) -> tuple[Any, Any]:
        initial_state = jnp.zeros((batch_size, latent_size), dtype=jnp.float32)
        burned_state, _ = adapter.burn_in(
            parameters,
            initial_state,
            observations[:burn_in_steps],
            controls[:burn_in_steps],
        )
        target = observations[burn_in_steps + 1 :]
        teacher_regressors = jnp.concatenate(
            (observations[burn_in_steps:-1], controls[burn_in_steps:]),
            axis=-1,
        )
        _, one_step = adapter.model.simulate_sequence(
            parameters,
            burned_state,
            teacher_regressors,
        )
        _, free = adapter.free_rollout(
            parameters,
            burned_state,
            observations[burn_in_steps],
            controls[burn_in_steps:],
        )
        return one_step - target, free - target

    one_error, free_error = evaluate(parameters)
    one_error = np.asarray(one_error)
    free_error = np.asarray(free_error)
    finite = bool(np.isfinite(one_error).all() and np.isfinite(free_error).all())
    if finite:
        one_step_nrmse = float(np.sqrt(np.mean(one_error**2)))
        feature_nrmse = np.sqrt(np.mean(free_error**2, axis=(0, 1)))
        free_rollout_nrmse = float(np.sqrt(np.mean(free_error**2)))
        maximum_prediction = float(
            np.max(np.abs(free_error + window.observations[burn_in_steps + 1 :]))
        )
    else:
        one_step_nrmse = math.inf
        feature_nrmse = np.asarray([math.inf, math.inf])
        free_rollout_nrmse = math.inf
        maximum_prediction = math.inf
    return ValidationRolloutMetrics(
        free_rollout_nrmse=free_rollout_nrmse,
        one_step_nrmse=one_step_nrmse,
        current_free_rollout_nrmse=float(feature_nrmse[0]),
        speed_free_rollout_nrmse=float(feature_nrmse[1]),
        maximum_absolute_normalized_prediction=maximum_prediction,
        finite=finite,
        horizon_steps=horizon,
        windows=batch_size,
    )


def save_phase6_checkpoint(
    directory: Path | str,
    *,
    dataset: Phase4Dataset,
    study: Phase6TrainingStudy,
    spec: Phase6Spec | None = None,
    overwrite: bool = False,
) -> Phase6CheckpointManifest:
    """Atomically save the selected model and all evidence needed to audit it."""

    try:
        import flax
        import optax
        from flax import serialization
    except ImportError as error:
        raise ImportError("checkpoint saving requires the Phase-6 dependencies") from error

    spec = spec or load_phase6_spec()
    profile = spec.profile(study.profile_name)
    output = Path(directory)
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"checkpoint directory already exists: {output}; pass overwrite=True explicitly"
        )
    staging = output.parent / f".{output.name}.building"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        parameter_path = staging / "parameters.msgpack"
        parameter_path.write_bytes(serialization.to_bytes(study.selected_run.parameters))
        normalization_path = staging / "normalization.npz"
        dataset.normalization.save(normalization_path)
        history_path = staging / "training_history.json"
        history_payload = study.history_payload()
        history_payload["training_configuration"] = {
            "interface": {
                "observation": list(spec.interface["observation"]),
                "control": list(spec.interface["control"]),
                "forbidden_training_features": list(
                    spec.interface["forbidden_training_features"]
                ),
            },
            "architecture": dict(spec.architecture),
            "loss": dict(spec.loss),
            "optimizer": dict(spec.optimizer),
            "selection": dict(spec.selection),
            "profile": {
                **{
                    key: value
                    for key, value in asdict(profile).items()
                    if key not in {"pilot_stages", "final_stages"}
                },
                "pilot_stages": [asdict(stage) for stage in profile.pilot_stages],
                "final_stages": [asdict(stage) for stage in profile.final_stages],
            },
            "runtime": {
                **inspect_jax_runtime().to_dict(),
                "flax_version": flax.__version__,
                "optax_version": optax.__version__,
            },
        }
        history_path.write_text(
            json.dumps(history_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        phase1 = load_phase1_spec()
        selected = study.selected_run
        manifest = Phase6CheckpointManifest(
            schema_version=1,
            phase=6,
            model_type="ContractingR2DN",
            training_profile=study.profile_name,
            dataset_fingerprint=dataset.fingerprint,
            train_trajectory_ids=dataset.trajectory_ids("train"),
            validation_trajectory_ids=dataset.trajectory_ids("validation"),
            upstream_commit=str(phase1.upstream["commit"]),
            observation_features=tuple(spec.interface["observation"]),
            control_features=tuple(spec.interface["control"]),
            forbidden_training_features=FORBIDDEN_TRAINING_FEATURES,
            latent_size=selected.architecture.state_size,
            feature_size=selected.architecture.features,
            hidden_sizes=selected.architecture.hidden,
            initialization=selected.architecture.init_method,
            polar_parameterization=selected.architecture.do_polar_param,
            seed=selected.seed,
            burn_in_steps=profile.burn_in_steps,
            selection_horizon_steps=profile.selection_horizon_steps,
            validation_window_seed=profile.selection_validation_seed,
            selection_metric=str(spec.selection["metric"]),
            update_count=selected.update_count,
            validation_free_rollout_nrmse=selected.selection_score,
            contractivity_margin=selected.contractivity_margin,
            parameter_sha256=_file_sha256(parameter_path),
            normalization_sha256=_file_sha256(normalization_path),
            training_history_sha256=_file_sha256(history_path),
        )
        manifest.validate(dataset=dataset, spec=spec)
        (staging / "manifest.json").write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output.exists():
            shutil.rmtree(output)
        staging.rename(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return manifest


def load_phase6_checkpoint(
    directory: Path | str,
    *,
    dataset: Phase4Dataset,
    spec: Phase6Spec | None = None,
) -> LoadedPhase6Checkpoint:
    """Load parameters only after every metadata and content binding is verified."""

    try:
        from flax import serialization
    except ImportError as error:
        raise ImportError("checkpoint loading requires the Phase-6 dependencies") from error

    spec = spec or load_phase6_spec()
    root = Path(directory)
    required = tuple(spec.checkpoint["required_files"])
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Phase-6 checkpoint is missing: {', '.join(missing)}")
    manifest = Phase6CheckpointManifest.from_dict(
        json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    )
    manifest.validate(dataset=dataset, spec=spec)
    for path, expected in (
        (root / "parameters.msgpack", manifest.parameter_sha256),
        (root / "normalization.npz", manifest.normalization_sha256),
        (root / "training_history.json", manifest.training_history_sha256),
    ):
        if _file_sha256(path) != expected:
            raise ValueError(f"checkpoint content hash mismatch: {path.name}")

    normalization = NormalizationStatistics.load(root / "normalization.npz")
    _validate_normalization_matches_dataset(normalization, dataset.normalization)
    architecture = R2DNArchitecture(
        input_size=int(spec.architecture["input_size"]),
        state_size=manifest.latent_size,
        features=manifest.feature_size,
        output_size=int(spec.architecture["output_size"]),
        hidden=manifest.hidden_sizes,
        init_method=manifest.initialization,
        do_polar_param=manifest.polar_parameterization,
    )
    adapter = OfficialR2DNAdapter(architecture)
    template, _ = adapter.initialize(seed=manifest.seed, batch_size=1)
    parameters = serialization.from_bytes(
        template,
        (root / "parameters.msgpack").read_bytes(),
    )
    history = json.loads((root / "training_history.json").read_text(encoding="utf-8"))
    if history.get("dataset_fingerprint") != dataset.fingerprint:
        raise ValueError("training history belongs to a different dataset")
    if int(history.get("selected_seed", -1)) != manifest.seed:
        raise ValueError("training history and manifest selected seeds differ")
    configuration = history.get("training_configuration", {})
    if configuration.get("optimizer") != spec.optimizer:
        raise ValueError("training history optimizer configuration mismatch")
    if configuration.get("loss") != spec.loss:
        raise ValueError("training history loss configuration mismatch")
    if configuration.get("selection") != spec.selection:
        raise ValueError("training history selection configuration mismatch")
    return LoadedPhase6Checkpoint(
        adapter=adapter,
        parameters=parameters,
        manifest=manifest,
        normalization=normalization,
        training_history=history,
    )


def _validate_normalization_matches_dataset(
    checkpoint: NormalizationStatistics,
    dataset: NormalizationStatistics,
) -> None:
    scalar_fields = ("observation_count", "control_count", "fit_split")
    if any(getattr(checkpoint, name) != getattr(dataset, name) for name in scalar_fields):
        raise ValueError("checkpoint normalization metadata differs from Phase 4")
    array_fields = (
        "observation_mean",
        "observation_std",
        "control_mean",
        "control_std",
    )
    if any(
        not np.array_equal(getattr(checkpoint, name), getattr(dataset, name))
        for name in array_fields
    ):
        raise ValueError("checkpoint normalization values differ from Phase 4")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
