"""Typed Phase-6 contract for R2DN world-model training."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from r2dn_dc_motor.phase1_spec import Phase1Spec, load_phase1_spec
from r2dn_dc_motor.phase4_spec import Phase4Spec, load_phase4_spec
from r2dn_dc_motor.spec import ExperimentSpec, SpecValidationError, load_phase0_spec

REQUIRED_PHASE6_CHECKS = {
    "training_uses_only_temperature_free_model_view",
    "training_and_selection_splits_are_isolated",
    "normalization_matches_phase4_train_statistics",
    "checkpoint_is_dataset_and_upstream_bound",
    "training_losses_and_gradients_are_finite",
    "contractivity_certificate_is_positive",
    "selection_uses_validation_long_rollout",
    "multiple_final_training_seeds_are_recorded",
}
FORBIDDEN_TRAINING_FEATURES = (
    "winding_temperature_c",
    "load_torque_n_m",
    "angular_speed_reference_rad_s",
    "commanded_armature_voltage_v",
)


@dataclass(frozen=True)
class CurriculumStage:
    """One fixed-shape curriculum stage."""

    name: str
    updates: int
    rollout_steps: int
    batch_size: int
    one_step_weight: float
    rollout_weight: float
    reconstruction_weight: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CurriculumStage:
        return cls(
            name=str(raw["name"]),
            updates=int(raw["updates"]),
            rollout_steps=int(raw["rollout_steps"]),
            batch_size=int(raw["batch_size"]),
            one_step_weight=float(raw["one_step_weight"]),
            rollout_weight=float(raw["rollout_weight"]),
            reconstruction_weight=float(raw["reconstruction_weight"]),
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.name:
            errors.append("curriculum stage name must not be empty")
        if self.updates < 1 or self.rollout_steps < 1 or self.batch_size < 1:
            errors.append(f"{self.name}: updates, rollout steps, and batch size must be positive")
        weights = (
            self.one_step_weight,
            self.rollout_weight,
            self.reconstruction_weight,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in weights):
            errors.append(f"{self.name}: loss weights must be finite and non-negative")
        if self.one_step_weight + self.rollout_weight <= 0.0:
            errors.append(f"{self.name}: predictive loss weight must be positive")
        return errors


@dataclass(frozen=True)
class TrainingProfile:
    """Pilot-search and final-training budget."""

    name: str
    pilot_seed: int
    candidate_latent_sizes: tuple[int, ...]
    training_seeds: tuple[int, ...]
    burn_in_steps: int
    pilot_validation_horizon_steps: int
    selection_horizon_steps: int
    pilot_validation_seed: int
    selection_validation_seed: int
    validation_windows: int
    history_log_interval: int
    pilot_stages: tuple[CurriculumStage, ...]
    final_stages: tuple[CurriculumStage, ...]

    @classmethod
    def from_dict(cls, name: str, raw: dict[str, Any]) -> TrainingProfile:
        return cls(
            name=name,
            pilot_seed=int(raw["pilot_seed"]),
            candidate_latent_sizes=tuple(int(value) for value in raw["candidate_latent_sizes"]),
            training_seeds=tuple(int(value) for value in raw["training_seeds"]),
            burn_in_steps=int(raw["burn_in_steps"]),
            pilot_validation_horizon_steps=int(
                raw["pilot_validation_horizon_steps"]
            ),
            selection_horizon_steps=int(raw["selection_horizon_steps"]),
            pilot_validation_seed=int(raw["pilot_validation_seed"]),
            selection_validation_seed=int(raw["selection_validation_seed"]),
            validation_windows=int(raw["validation_windows"]),
            history_log_interval=int(raw["history_log_interval"]),
            pilot_stages=tuple(
                CurriculumStage.from_dict(stage) for stage in raw["pilot_stages"]
            ),
            final_stages=tuple(
                CurriculumStage.from_dict(stage) for stage in raw["final_stages"]
            ),
        )

    def validate(self, minimum_latent_size: int) -> list[str]:
        errors: list[str] = []
        if self.pilot_seed < 0 or any(seed < 0 for seed in self.training_seeds):
            errors.append(f"{self.name}: training seeds must be non-negative")
        if len(set(self.training_seeds)) != len(self.training_seeds):
            errors.append(f"{self.name}: training seeds must be unique")
        if not self.candidate_latent_sizes:
            errors.append(f"{self.name}: at least one latent size is required")
        if len(set(self.candidate_latent_sizes)) != len(self.candidate_latent_sizes):
            errors.append(f"{self.name}: candidate latent sizes must be unique")
        if any(value < minimum_latent_size for value in self.candidate_latent_sizes):
            errors.append(f"{self.name}: candidate latent size is below the Phase-1 minimum")
        if (
            self.burn_in_steps < 1
            or self.pilot_validation_horizon_steps < 1
            or self.selection_horizon_steps < self.pilot_validation_horizon_steps
            or self.pilot_validation_seed < 0
            or self.selection_validation_seed < 0
            or self.pilot_validation_seed == self.selection_validation_seed
            or self.validation_windows < 1
            or self.history_log_interval < 1
        ):
            errors.append(f"{self.name}: invalid burn-in, validation, or logging budget")
        if not self.pilot_stages or not self.final_stages:
            errors.append(f"{self.name}: pilot and final curricula must be non-empty")
        for stage in (*self.pilot_stages, *self.final_stages):
            errors.extend(stage.validate())
        return errors


@dataclass(frozen=True)
class Phase6Spec:
    """Executable source of truth for Phase-6 training."""

    schema_version: int
    phase: dict[str, Any]
    interface: dict[str, Any]
    architecture: dict[str, Any]
    loss: dict[str, Any]
    optimizer: dict[str, Any]
    selection: dict[str, Any]
    checkpoint: dict[str, Any]
    profiles: dict[str, TrainingProfile]
    validation: dict[str, Any]

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
        *,
        phase0: ExperimentSpec | None = None,
        phase1: Phase1Spec | None = None,
        phase4: Phase4Spec | None = None,
    ) -> Phase6Spec:
        spec = cls(
            schema_version=int(raw["schema_version"]),
            phase=dict(raw["phase"]),
            interface=dict(raw["interface"]),
            architecture=dict(raw["architecture"]),
            loss=dict(raw["loss"]),
            optimizer=dict(raw["optimizer"]),
            selection=dict(raw["selection"]),
            checkpoint=dict(raw["checkpoint"]),
            profiles={
                name: TrainingProfile.from_dict(name, values)
                for name, values in raw["profiles"].items()
            },
            validation=dict(raw["validation"]),
        )
        phase0 = phase0 or load_phase0_spec()
        phase1 = phase1 or load_phase1_spec(phase0=phase0)
        phase4 = phase4 or load_phase4_spec(phase0=phase0)
        spec.validate(phase0=phase0, phase1=phase1, phase4=phase4)
        return spec

    def profile(self, name: str) -> TrainingProfile:
        try:
            return self.profiles[name]
        except KeyError as error:
            choices = ", ".join(sorted(self.profiles))
            raise SpecValidationError(
                f"unknown Phase-6 profile {name!r}; choose one of: {choices}"
            ) from error

    def validate(
        self,
        *,
        phase0: ExperimentSpec,
        phase1: Phase1Spec,
        phase4: Phase4Spec,
    ) -> None:
        errors: list[str] = []
        if self.schema_version != 1:
            errors.append("schema_version must be 1")
        if self.phase.get("number") != 6 or self.phase.get("status") != "implemented":
            errors.append("Phase 6 must be numbered 6 and marked implemented")

        expected_observation = tuple(phase4.features["model_observation"])
        expected_control = tuple(phase4.features["model_control"])
        if tuple(self.interface.get("observation", ())) != expected_observation:
            errors.append("Phase-6 observation order drifted from Phase 4")
        if tuple(self.interface.get("control", ())) != expected_control:
            errors.append("Phase-6 control order drifted from Phase 4")
        if tuple(self.interface.get("regressor", ())) != (
            *expected_observation,
            *expected_control,
        ):
            errors.append("Phase-6 regressor must be [current, speed, voltage]")
        if tuple(self.interface.get("target", ())) != expected_observation:
            errors.append("Phase-6 target must be the next measured observation")
        if tuple(self.interface.get("forbidden_training_features", ())) != (
            FORBIDDEN_TRAINING_FEATURES
        ):
            errors.append("forbidden Phase-6 training feature catalog changed")
        if phase0.signals.hidden_state[0] not in FORBIDDEN_TRAINING_FEATURES:
            errors.append("hidden temperature must be explicitly forbidden")
        if self.interface.get("normalization") != "phase4_train_only":
            errors.append("Phase 6 must reuse Phase-4 train-only normalization")
        if self.interface.get("sequence_layout") != "time_batch_feature":
            errors.append("R2DN training sequences must remain time-major")

        errors.extend(self._validate_architecture(phase1))
        errors.extend(self._validate_loss_and_optimizer())
        errors.extend(self._validate_selection())
        errors.extend(self._validate_checkpoint(phase1))

        if set(self.profiles) != {"ci", "final"}:
            errors.append("Phase 6 must define exactly ci and final profiles")
        else:
            minimum_latent = int(phase1.interface["minimum_latent_size"])
            for profile in self.profiles.values():
                errors.extend(profile.validate(minimum_latent))
            final = self.profiles["final"]
            if len(final.candidate_latent_sizes) < 3:
                errors.append("final pilot must compare at least three latent sizes")
            if len(final.training_seeds) < 3:
                errors.append("final training must use at least three seeds")
            if final.burn_in_steps != 250:
                errors.append("final burn-in must remain the Phase-3 selected 250 ms")
            pilot_horizons = tuple(
                stage.rollout_steps for stage in final.pilot_stages
            )
            if pilot_horizons != tuple(sorted(pilot_horizons)):
                errors.append("final pilot rollout horizons must be non-decreasing")
            if (
                not pilot_horizons
                or pilot_horizons[0] != 1
                or max(pilot_horizons) < 100
                or not any(1 < horizon <= 10 for horizon in pilot_horizons)
            ):
                errors.append(
                    "final pilot must bridge one-step and 100-step rollout "
                    "training with a horizon of at most 10 steps"
                )
            if max(stage.rollout_steps for stage in final.final_stages) < 1000:
                errors.append("final curriculum must reach a one-second free rollout")

        if set(self.validation.get("required_checks", ())) != REQUIRED_PHASE6_CHECKS:
            errors.append("Phase-6 validation check catalog changed")
        prediction_limit = float(
            self.validation.get("maximum_absolute_normalized_prediction", math.nan)
        )
        if not math.isfinite(prediction_limit) or prediction_limit <= 0.0:
            errors.append("Phase-6 normalized prediction guard must be positive")
        if self.validation.get("phase6_does_not_claim_phase7_gate") is not True:
            errors.append("Phase 6 must not claim the clean Phase-7 comparison gate")
        if errors:
            raise SpecValidationError("\n".join(f"- {error}" for error in errors))

    def _validate_architecture(self, phase1: Phase1Spec) -> list[str]:
        errors: list[str] = []
        if self.architecture.get("backend") != phase1.interface["backend"]:
            errors.append("Phase-6 backend drifted from the pinned Phase-1 backend")
        for key in ("input_size", "output_size"):
            if int(self.architecture.get(key, -1)) != int(phase1.interface[key]):
                errors.append(f"Phase-6 {key} drifted from Phase 1")
        if int(self.architecture.get("feature_size", 0)) < 1:
            errors.append("R2DN feature size must be positive")
        hidden = tuple(int(value) for value in self.architecture.get("hidden_sizes", ()))
        if not hidden or min(hidden) < 1:
            errors.append("R2DN hidden sizes must be positive")
        candidates = tuple(
            int(value) for value in self.architecture.get("candidate_latent_sizes", ())
        )
        if candidates != (4, 6, 8):
            errors.append("locked pilot latent-size catalog must be [4, 6, 8]")
        if self.architecture.get("initialization") != "long_memory":
            errors.append("Phase-6 R2DN must use long-memory initialization")
        if self.architecture.get("polar_parameterization") is not True:
            errors.append("contracting polar parameterization must remain enabled")
        return errors

    def _validate_loss_and_optimizer(self) -> list[str]:
        errors: list[str] = []
        if self.loss.get("teacher_forcing_after_burn_in") is not False:
            errors.append("free-rollout loss must not teacher-force after burn-in")
        if self.loss.get("target_weighting") != "equal_after_train_normalization":
            errors.append("current and speed must be weighted after train normalization")
        if self.optimizer.get("name") != "adamw":
            errors.append("locked Phase-6 optimizer must be AdamW")
        for key in ("learning_rate", "gradient_clip_norm"):
            value = float(self.optimizer.get(key, math.nan))
            if not math.isfinite(value) or value <= 0.0:
                errors.append(f"optimizer {key} must be finite and positive")
        decay = float(self.optimizer.get("weight_decay", math.nan))
        if not math.isfinite(decay) or decay < 0.0:
            errors.append("optimizer weight decay must be finite and non-negative")
        return errors

    def _validate_selection(self) -> list[str]:
        errors: list[str] = []
        if self.selection.get("fit_split") != "train":
            errors.append("R2DN may only fit on the train split")
        if self.selection.get("selection_split") != "validation":
            errors.append("R2DN checkpoint selection must use validation")
        if self.selection.get("id_test_used_for_selection") is not False:
            errors.append("ID test must not select a Phase-6 checkpoint")
        if self.selection.get("ood_test_used_for_selection") is not False:
            errors.append("OOD test must not select a Phase-6 checkpoint")
        if self.selection.get("metric") != "validation_free_rollout_nrmse":
            errors.append("checkpoint selection must use validation free-rollout NRMSE")
        if self.selection.get("mode") != "min":
            errors.append("validation free-rollout NRMSE must be minimized")
        return errors

    def _validate_checkpoint(self, phase1: Phase1Spec) -> list[str]:
        errors: list[str] = []
        if tuple(self.checkpoint.get("required_files", ())) != tuple(
            phase1.checkpoint["required_files"]
        ) + ("training_history.json",):
            errors.append("Phase-6 checkpoint file contract changed")
        if self.checkpoint.get("dataset_bound") is not True:
            errors.append("Phase-6 checkpoint must be dataset-bound")
        if self.checkpoint.get("upstream_commit_bound") is not True:
            errors.append("Phase-6 checkpoint must be upstream-commit-bound")
        return errors

    def summary(self, profile_name: str = "final") -> str:
        profile = self.profile(profile_name)
        return "\n".join(
            [
                "PHASE 6 SPEC: PASS",
                "model: pinned official ContractingR2DN",
                "training view: normalized current, speed, applied voltage",
                "forbidden: temperature, load, reference, commanded voltage",
                f"profile: {profile.name}",
                f"burn-in: {profile.burn_in_steps} steps",
                f"latent candidates: {list(profile.candidate_latent_sizes)}",
                f"final seeds: {list(profile.training_seeds)}",
                (
                    "selection: validation free rollout, "
                    f"{profile.selection_horizon_steps} steps"
                ),
                "Phase-7 Gate is not claimed by this training phase",
            ]
        )


def load_phase6_spec(
    path: Path | str | None = None,
    *,
    phase0: ExperimentSpec | None = None,
    phase1: Phase1Spec | None = None,
    phase4: Phase4Spec | None = None,
) -> Phase6Spec:
    """Load and validate the canonical Phase-6 TOML contract."""

    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "phase6.toml"
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    return Phase6Spec.from_dict(
        raw,
        phase0=phase0,
        phase1=phase1,
        phase4=phase4,
    )
