"""Phase-7 pure-R2DN accuracy study and versioned checkpoint support."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from r2dn_dc_motor.data import NormalizationStatistics, Phase4Dataset
from r2dn_dc_motor.data.phase4_dataset import canonical_sha256
from r2dn_dc_motor.models.r2dn_adapter import OfficialR2DNAdapter, R2DNArchitecture
from r2dn_dc_motor.models.r2dn_training import (
    R2DNRunResult,
    R2DNTrainingObjective,
    ValidationRolloutMetrics,
    compute_train_increment_std,
    train_r2dn_run_with_objective,
)
from r2dn_dc_motor.phase6_spec import Phase6Spec, TrainingProfile, load_phase6_spec
from r2dn_dc_motor.phase7_spec import (
    Phase7Profile,
    Phase7Spec,
    Phase7Variant,
    load_phase7_spec,
)

PHASE7_ALGORITHM_VERSION = "phase7_broadband_delta_multiscale_v1"
ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class Phase7RunEvidence:
    variant: str
    run: R2DNRunResult

    def to_dict(self) -> dict[str, Any]:
        return {"variant": self.variant, **self.run.summary()}


@dataclass(frozen=True)
class Phase7VariantAggregate:
    variant: str
    seeds: tuple[int, ...]
    validation_scores: tuple[float, ...]
    median_validation_nrmse: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase7Study:
    profile_name: str
    dataset_fingerprint: str
    protocol_sha256: str
    increment_std_normalized: tuple[float, float]
    runs: tuple[Phase7RunEvidence, ...]
    aggregates: tuple[Phase7VariantAggregate, ...]
    selected_variant: str
    selected_run: Phase7RunEvidence
    target_combined_nrmse: float

    @property
    def selected_aggregate(self) -> Phase7VariantAggregate:
        return next(value for value in self.aggregates if value.variant == self.selected_variant)

    @property
    def target_met(self) -> bool:
        return self.selected_aggregate.median_validation_nrmse <= self.target_combined_nrmse

    def history_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "phase": 7,
            "algorithm_version": PHASE7_ALGORITHM_VERSION,
            "profile": self.profile_name,
            "dataset_fingerprint": self.dataset_fingerprint,
            "protocol_sha256": self.protocol_sha256,
            "increment_std_normalized": list(self.increment_std_normalized),
            "selection_split": "validation",
            "selection_metric": "median_validation_free_rollout_nrmse",
            "runs": [value.to_dict() for value in self.runs],
            "aggregates": [value.to_dict() for value in self.aggregates],
            "selected_variant": self.selected_variant,
            "selected_seed": self.selected_run.run.seed,
            "selected_run_history": list(self.selected_run.run.history),
            "target_combined_nrmse": self.target_combined_nrmse,
            "target_met": self.target_met,
        }


@dataclass(frozen=True)
class Phase7CheckpointManifest:
    schema_version: int
    phase: int
    model_type: str
    training_profile: str
    selected_variant: str
    dataset_fingerprint: str
    dataset_id: str
    dataset_version: str
    protocol_sha256: str
    latent_size: int
    feature_size: int
    hidden_sizes: tuple[int, ...]
    initialization: str
    polar_parameterization: bool
    absolute_state_output_retained: bool
    seed: int
    burn_in_steps: int
    selection_horizon_steps: int
    selection_metric: str
    variant_median_validation_nrmse: float
    validation_free_rollout_nrmse: float
    current_validation_nrmse: float
    speed_validation_nrmse: float
    contractivity_margin: float
    update_count: int
    delta_weight: float
    increment_std_normalized: tuple[float, float]
    rollout_horizon_weights: tuple[tuple[int, float], ...]
    target_combined_nrmse: float
    target_met: bool
    parameter_sha256: str
    normalization_sha256: str
    study_history_sha256: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Phase7CheckpointManifest:
        values = dict(raw)
        values["hidden_sizes"] = tuple(int(value) for value in raw["hidden_sizes"])
        values["increment_std_normalized"] = tuple(
            float(value) for value in raw["increment_std_normalized"]
        )
        values["rollout_horizon_weights"] = tuple(
            (int(item[0]), float(item[1])) for item in raw["rollout_horizon_weights"]
        )
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(
        self,
        *,
        dataset: Phase4Dataset,
        phase7: Phase7Spec,
        phase6: Phase6Spec,
    ) -> None:
        errors: list[str] = []
        profile = phase7.profile(self.training_profile)
        variants = {value.name: value for value in phase7.variants}
        variant = variants.get(self.selected_variant)
        if self.schema_version != 1 or self.phase != 7:
            errors.append("Phase-7 checkpoint schema or phase mismatch")
        if self.model_type != "ContractingR2DN":
            errors.append("Phase-7 checkpoint model type mismatch")
        if self.dataset_fingerprint != dataset.fingerprint:
            errors.append("Phase-7 checkpoint belongs to another dataset")
        if self.dataset_id != dataset.manifest.get("dataset_id"):
            errors.append("Phase-7 checkpoint dataset ID mismatch")
        if self.dataset_version != dataset.manifest.get("dataset_version"):
            errors.append("Phase-7 checkpoint dataset version mismatch")
        if self.dataset_version != phase7.dataset["required_version"]:
            errors.append("Phase-7 checkpoint was not trained on broadband v2 data")
        if self.protocol_sha256 != phase7_protocol_sha256(phase7, phase6):
            errors.append("Phase-7 checkpoint protocol hash mismatch")
        if variant is None:
            errors.append("Phase-7 selected variant is not declared")
        else:
            if self.feature_size != variant.feature_size:
                errors.append("Phase-7 checkpoint feature width mismatch")
            if self.hidden_sizes != variant.hidden_sizes:
                errors.append("Phase-7 checkpoint hidden widths mismatch")
            if not math.isclose(self.delta_weight, variant.delta_weight):
                errors.append("Phase-7 checkpoint delta weight mismatch")
            expected_updates = sum(
                stage.updates for stage in profile.stages(variant.curriculum)
            )
            if self.update_count != expected_updates:
                errors.append("Phase-7 checkpoint did not complete its curriculum")
            expected_horizons = tuple(
                zip(
                    (
                        int(value)
                        for value in phase7.objective["rollout_horizon_steps"]
                    ),
                    (
                        float(value)
                        for value in phase7.objective["rollout_horizon_weights"]
                    ),
                    strict=True,
                )
            )
            if not variant.use_multihorizon_loss:
                expected_horizons = ()
            if self.rollout_horizon_weights != expected_horizons:
                errors.append("Phase-7 checkpoint multi-horizon objective mismatch")
        if self.latent_size != int(phase7.training["latent_size"]):
            errors.append("Phase-7 checkpoint latent size mismatch")
        if self.initialization != phase6.architecture["initialization"]:
            errors.append("Phase-7 checkpoint initialization mismatch")
        if not self.polar_parameterization or not self.absolute_state_output_retained:
            errors.append("Phase-7 checkpoint discarded a required stability constraint")
        if self.seed not in profile.seeds or self.burn_in_steps != profile.burn_in_steps:
            errors.append("Phase-7 checkpoint seed or burn-in mismatch")
        if self.selection_horizon_steps != profile.validation_horizon_steps:
            errors.append("Phase-7 checkpoint selection horizon mismatch")
        if self.selection_metric != phase7.training["selection_metric"]:
            errors.append("Phase-7 checkpoint selection metric mismatch")
        if any(value <= 0.0 for value in self.increment_std_normalized):
            errors.append("Phase-7 checkpoint increment scales are invalid")
        numeric = (
            self.variant_median_validation_nrmse,
            self.validation_free_rollout_nrmse,
            self.current_validation_nrmse,
            self.speed_validation_nrmse,
            self.contractivity_margin,
            self.target_combined_nrmse,
        )
        if not all(math.isfinite(value) for value in numeric):
            errors.append("Phase-7 checkpoint contains non-finite metrics")
        if min(numeric[:4]) < 0.0 or self.contractivity_margin <= 0.0:
            errors.append("Phase-7 checkpoint accuracy or contraction margin is invalid")
        if self.target_met != (
            self.variant_median_validation_nrmse <= self.target_combined_nrmse
        ):
            errors.append("Phase-7 checkpoint target flag is inconsistent")
        for name, digest in (
            ("parameter", self.parameter_sha256),
            ("normalization", self.normalization_sha256),
            ("study history", self.study_history_sha256),
        ):
            if len(digest) != 64:
                errors.append(f"{name} SHA-256 is malformed")
        if errors:
            raise ValueError("\n".join(errors))


@dataclass(frozen=True)
class LoadedPhase7Checkpoint:
    adapter: OfficialR2DNAdapter
    parameters: Any
    manifest: Phase7CheckpointManifest
    normalization: NormalizationStatistics
    study_history: dict[str, Any]


def phase7_protocol_sha256(phase7: Phase7Spec, phase6: Phase6Spec) -> str:
    return canonical_sha256(
        {
            "algorithm_version": PHASE7_ALGORITHM_VERSION,
            "phase7": asdict(phase7),
            "phase6_backend": phase6.architecture["backend"],
            "phase6_input_size": phase6.architecture["input_size"],
            "phase6_output_size": phase6.architecture["output_size"],
            "phase6_optimizer": phase6.optimizer,
        }
    )


def _training_profile(profile: Phase7Profile, variant: Phase7Variant) -> TrainingProfile:
    stages = profile.stages(variant.curriculum)
    return TrainingProfile(
        name=f"phase7-{profile.name}-{variant.name}",
        pilot_seed=profile.seeds[0],
        candidate_latent_sizes=(16,),
        training_seeds=profile.seeds,
        burn_in_steps=profile.burn_in_steps,
        pilot_validation_horizon_steps=min(100, profile.validation_horizon_steps),
        selection_horizon_steps=profile.validation_horizon_steps,
        pilot_validation_seed=profile.validation_window_seed + 1,
        selection_validation_seed=profile.validation_window_seed,
        validation_windows=profile.validation_windows,
        history_log_interval=profile.history_log_interval,
        pilot_stages=stages,
        final_stages=stages,
    )


def train_phase7_study(
    dataset: Phase4Dataset,
    *,
    phase7: Phase7Spec | None = None,
    phase6: Phase6Spec | None = None,
    profile_name: str = "final",
    cache_directory: Path | str = Path("checkpoints/phase7/run-cache-v1"),
    overwrite_cache: bool = False,
    progress: ProgressCallback | None = None,
) -> Phase7Study:
    phase6 = phase6 or load_phase6_spec()
    phase7 = phase7 or load_phase7_spec()
    profile = phase7.profile(profile_name)
    _validate_broadband_dataset(dataset, phase7, profile_name=profile_name)
    progress = progress or (lambda _: None)
    protocol = phase7_protocol_sha256(phase7, phase6)
    increment_std = compute_train_increment_std(
        dataset,
        floor=float(phase7.objective["increment_standard_deviation_floor"]),
    )
    horizons = tuple(
        zip(
            (int(value) for value in phase7.objective["rollout_horizon_steps"]),
            (float(value) for value in phase7.objective["rollout_horizon_weights"]),
            strict=True,
        )
    )
    cache = Path(cache_directory)
    evidences: list[Phase7RunEvidence] = []
    aggregates: list[Phase7VariantAggregate] = []
    for variant in phase7.variants:
        variant_runs: list[Phase7RunEvidence] = []
        training_profile = _training_profile(profile, variant)
        objective = R2DNTrainingObjective(
            delta_weight=variant.delta_weight,
            delta_std_normalized=increment_std,
            rollout_horizon_weights=(horizons if variant.use_multihorizon_loss else ()),
        )
        for seed in profile.seeds:
            role = f"phase7_{variant.name}"
            path = cache / profile.name / variant.name / f"seed-{seed}"
            if path.is_dir() and not overwrite_cache:
                run = _load_run_cache(
                    path,
                    dataset=dataset,
                    protocol_sha256=protocol,
                    variant=variant,
                    seed=seed,
                    role=role,
                    validation_horizon_steps=profile.validation_horizon_steps,
                )
                progress(
                    f"cache hit {variant.name}, seed={seed}: "
                    f"validation NRMSE={run.selection_score:.6g}"
                )
            else:
                progress(
                    f"train {variant.name}, seed={seed}, "
                    f"updates={sum(stage.updates for stage in training_profile.final_stages)}"
                )
                run = train_r2dn_run_with_objective(
                    dataset,
                    spec=phase6,
                    profile=training_profile,
                    latent_size=int(phase7.training["latent_size"]),
                    seed=seed,
                    stages=training_profile.final_stages,
                    validation_horizon_steps=profile.validation_horizon_steps,
                    validation_window_seed=profile.validation_window_seed,
                    role=role,
                    objective=objective,
                    architecture={
                        "feature_size": variant.feature_size,
                        "hidden_sizes": variant.hidden_sizes,
                    },
                    progress=progress,
                )
                _save_run_cache(
                    path,
                    run=run,
                    dataset=dataset,
                    protocol_sha256=protocol,
                    variant=variant,
                    overwrite=overwrite_cache,
                )
            evidence = Phase7RunEvidence(variant=variant.name, run=run)
            variant_runs.append(evidence)
            evidences.append(evidence)
        scores = tuple(value.run.selection_score for value in variant_runs)
        aggregates.append(
            Phase7VariantAggregate(
                variant=variant.name,
                seeds=profile.seeds,
                validation_scores=scores,
                median_validation_nrmse=float(median(scores)),
            )
        )
    selected_variant = _select_variant(
        phase7,
        aggregates,
        relative_tolerance=float(phase7.training["near_tie_relative_tolerance"]),
    )
    selected_run = min(
        (value for value in evidences if value.variant == selected_variant),
        key=lambda value: value.run.selection_score,
    )
    progress(
        f"selected {selected_variant}, seed={selected_run.run.seed}, "
        f"validation NRMSE={selected_run.run.selection_score:.6g}"
    )
    return Phase7Study(
        profile_name=profile_name,
        dataset_fingerprint=dataset.fingerprint,
        protocol_sha256=protocol,
        increment_std_normalized=increment_std,
        runs=tuple(evidences),
        aggregates=tuple(aggregates),
        selected_variant=selected_variant,
        selected_run=selected_run,
        target_combined_nrmse=float(phase7.training["target_combined_nrmse"]),
    )


def save_phase7_checkpoint(
    directory: Path | str,
    *,
    dataset: Phase4Dataset,
    study: Phase7Study,
    phase7: Phase7Spec | None = None,
    phase6: Phase6Spec | None = None,
    overwrite: bool = False,
) -> Phase7CheckpointManifest:
    try:
        from flax import serialization
    except ImportError as error:
        raise ImportError("Phase-7 checkpoint saving requires Phase-6 dependencies") from error
    phase6 = phase6 or load_phase6_spec()
    phase7 = phase7 or load_phase7_spec()
    profile = phase7.profile(study.profile_name)
    variant = next(value for value in phase7.variants if value.name == study.selected_variant)
    objective_horizons = tuple(
        zip(
            (int(value) for value in phase7.objective["rollout_horizon_steps"]),
            (float(value) for value in phase7.objective["rollout_horizon_weights"]),
            strict=True,
        )
    )
    if not variant.use_multihorizon_loss:
        objective_horizons = ()
    selected = study.selected_run.run
    output = Path(directory)
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Phase-7 checkpoint already exists: {output}; pass overwrite=True explicitly"
        )
    staging = output.parent / f".{output.name}.building"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        parameter_path = staging / "parameters.msgpack"
        parameter_path.write_bytes(serialization.to_bytes(selected.parameters))
        normalization_path = staging / "normalization.npz"
        dataset.normalization.save(normalization_path)
        history_path = staging / "study_history.json"
        history_path.write_text(
            json.dumps(study.history_payload(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = Phase7CheckpointManifest(
            schema_version=1,
            phase=7,
            model_type="ContractingR2DN",
            training_profile=study.profile_name,
            selected_variant=study.selected_variant,
            dataset_fingerprint=dataset.fingerprint,
            dataset_id=str(dataset.manifest["dataset_id"]),
            dataset_version=str(dataset.manifest["dataset_version"]),
            protocol_sha256=study.protocol_sha256,
            latent_size=selected.architecture.state_size,
            feature_size=selected.architecture.features,
            hidden_sizes=selected.architecture.hidden,
            initialization=selected.architecture.init_method,
            polar_parameterization=selected.architecture.do_polar_param,
            absolute_state_output_retained=True,
            seed=selected.seed,
            burn_in_steps=profile.burn_in_steps,
            selection_horizon_steps=profile.validation_horizon_steps,
            selection_metric=str(phase7.training["selection_metric"]),
            variant_median_validation_nrmse=(
                study.selected_aggregate.median_validation_nrmse
            ),
            validation_free_rollout_nrmse=selected.validation.free_rollout_nrmse,
            current_validation_nrmse=selected.validation.current_free_rollout_nrmse,
            speed_validation_nrmse=selected.validation.speed_free_rollout_nrmse,
            contractivity_margin=selected.contractivity_margin,
            update_count=selected.update_count,
            delta_weight=variant.delta_weight,
            increment_std_normalized=study.increment_std_normalized,
            rollout_horizon_weights=objective_horizons,
            target_combined_nrmse=study.target_combined_nrmse,
            target_met=study.target_met,
            parameter_sha256=_file_sha256(parameter_path),
            normalization_sha256=_file_sha256(normalization_path),
            study_history_sha256=_file_sha256(history_path),
        )
        manifest.validate(dataset=dataset, phase7=phase7, phase6=phase6)
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


def load_phase7_checkpoint(
    directory: Path | str,
    *,
    dataset: Phase4Dataset,
    phase7: Phase7Spec | None = None,
    phase6: Phase6Spec | None = None,
) -> LoadedPhase7Checkpoint:
    try:
        from flax import serialization
    except ImportError as error:
        raise ImportError("Phase-7 checkpoint loading requires Phase-6 dependencies") from error
    phase6 = phase6 or load_phase6_spec()
    phase7 = phase7 or load_phase7_spec()
    root = Path(directory)
    required = ("manifest.json", "parameters.msgpack", "normalization.npz", "study_history.json")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Phase-7 checkpoint is missing: {', '.join(missing)}")
    manifest = Phase7CheckpointManifest.from_dict(
        json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    )
    manifest.validate(dataset=dataset, phase7=phase7, phase6=phase6)
    for path, expected in (
        (root / "parameters.msgpack", manifest.parameter_sha256),
        (root / "normalization.npz", manifest.normalization_sha256),
        (root / "study_history.json", manifest.study_history_sha256),
    ):
        if _file_sha256(path) != expected:
            raise ValueError(f"Phase-7 checkpoint content hash mismatch: {path.name}")
    normalization = NormalizationStatistics.load(root / "normalization.npz")
    _validate_normalization(normalization, dataset.normalization)
    architecture = R2DNArchitecture(
        input_size=int(phase6.architecture["input_size"]),
        state_size=manifest.latent_size,
        features=manifest.feature_size,
        output_size=int(phase6.architecture["output_size"]),
        hidden=manifest.hidden_sizes,
        init_method=manifest.initialization,
        do_polar_param=manifest.polar_parameterization,
    )
    adapter = OfficialR2DNAdapter(architecture)
    template, _ = adapter.initialize(seed=manifest.seed, batch_size=1)
    parameters = serialization.from_bytes(template, (root / "parameters.msgpack").read_bytes())
    history = json.loads((root / "study_history.json").read_text(encoding="utf-8"))
    if history.get("protocol_sha256") != manifest.protocol_sha256:
        raise ValueError("Phase-7 history protocol hash mismatch")
    if history.get("dataset_fingerprint") != dataset.fingerprint:
        raise ValueError("Phase-7 history belongs to another dataset")
    if history.get("selected_variant") != manifest.selected_variant:
        raise ValueError("Phase-7 history selected variant mismatch")
    return LoadedPhase7Checkpoint(
        adapter=adapter,
        parameters=parameters,
        manifest=manifest,
        normalization=normalization,
        study_history=history,
    )


def _select_variant(
    phase7: Phase7Spec,
    aggregates: list[Phase7VariantAggregate],
    *,
    relative_tolerance: float,
) -> str:
    scores = {value.variant: value.median_validation_nrmse for value in aggregates}
    best = min(scores.values())
    threshold = best * (1.0 + relative_tolerance)
    return next(value.name for value in phase7.variants if scores[value.name] <= threshold)


def _validate_broadband_dataset(
    dataset: Phase4Dataset,
    phase7: Phase7Spec,
    *,
    profile_name: str,
) -> None:
    errors: list[str] = []
    if dataset.manifest.get("dataset_id") != phase7.dataset["required_dataset_id"]:
        errors.append("Phase-7 training requires the broadband dataset ID")
    if dataset.manifest.get("dataset_version") != phase7.dataset["required_version"]:
        errors.append("Phase-7 training requires broadband dataset version 2.0.0")
    if dataset.manifest.get("simulator", {}).get("name") != "FULL":
        errors.append("Phase-7 training data must come only from FULL")
    if dataset.manifest.get("normalization", {}).get("temperature_used") is not False:
        errors.append("Phase-7 dataset normalization leaked temperature")
    if profile_name == "final" and dataset.manifest.get("profile") != "final":
        errors.append("final Phase-7 training requires the final broadband dataset")
    if errors:
        raise ValueError("\n".join(errors))


def _save_run_cache(
    path: Path,
    *,
    run: R2DNRunResult,
    dataset: Phase4Dataset,
    protocol_sha256: str,
    variant: Phase7Variant,
    overwrite: bool,
) -> None:
    from flax import serialization

    if path.exists() and not overwrite:
        raise FileExistsError(f"run cache already exists: {path}")
    staging = path.parent / f".{path.name}.building"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        parameter_path = staging / "parameters.msgpack"
        parameter_path.write_bytes(serialization.to_bytes(run.parameters))
        payload = {
            "schema_version": 1,
            "dataset_fingerprint": dataset.fingerprint,
            "protocol_sha256": protocol_sha256,
            "variant": asdict(variant),
            "architecture": asdict(run.architecture),
            "summary": run.summary(),
            "history": list(run.history),
            "parameter_sha256": _file_sha256(parameter_path),
        }
        (staging / "run.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if path.exists():
            shutil.rmtree(path)
        staging.rename(path)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _load_run_cache(
    path: Path,
    *,
    dataset: Phase4Dataset,
    protocol_sha256: str,
    variant: Phase7Variant,
    seed: int,
    role: str,
    validation_horizon_steps: int,
) -> R2DNRunResult:
    from flax import serialization

    payload = json.loads((path / "run.json").read_text(encoding="utf-8"))
    summary = payload["summary"]
    if payload.get("dataset_fingerprint") != dataset.fingerprint:
        raise ValueError(f"run cache belongs to another dataset: {path}")
    if payload.get("protocol_sha256") != protocol_sha256:
        raise ValueError(f"run cache protocol mismatch: {path}")
    if canonical_sha256(payload.get("variant")) != canonical_sha256(asdict(variant)):
        raise ValueError(f"run cache variant mismatch: {path}")
    if int(summary.get("seed", -1)) != seed or summary.get("role") != role:
        raise ValueError(f"run cache seed or role mismatch: {path}")
    validation = ValidationRolloutMetrics(**summary["validation"])
    if validation.horizon_steps != validation_horizon_steps:
        raise ValueError(f"run cache validation horizon mismatch: {path}")
    parameter_path = path / "parameters.msgpack"
    if _file_sha256(parameter_path) != payload["parameter_sha256"]:
        raise ValueError(f"run cache parameter hash mismatch: {path}")
    raw = payload["architecture"]
    architecture = R2DNArchitecture(
        input_size=int(raw["input_size"]),
        state_size=int(raw["state_size"]),
        features=int(raw["features"]),
        output_size=int(raw["output_size"]),
        hidden=tuple(int(value) for value in raw["hidden"]),
        init_method=str(raw["init_method"]),
        do_polar_param=bool(raw["do_polar_param"]),
    )
    if architecture.features != variant.feature_size or architecture.hidden != variant.hidden_sizes:
        raise ValueError(f"run cache architecture drift: {path}")
    adapter = OfficialR2DNAdapter(architecture)
    template, _ = adapter.initialize(seed=seed, batch_size=1)
    parameters = serialization.from_bytes(template, parameter_path.read_bytes())
    return R2DNRunResult(
        parameters=parameters,
        architecture=architecture,
        seed=seed,
        role=role,
        update_count=int(summary["update_count"]),
        contractivity_margin=float(summary["contractivity_margin"]),
        validation=validation,
        history=tuple(payload["history"]),
    )


def _validate_normalization(
    checkpoint: NormalizationStatistics,
    dataset: NormalizationStatistics,
) -> None:
    scalar_fields = ("observation_count", "control_count", "fit_split")
    array_fields = ("observation_mean", "observation_std", "control_mean", "control_std")
    if any(getattr(checkpoint, name) != getattr(dataset, name) for name in scalar_fields):
        raise ValueError("Phase-7 normalization metadata differs from its training dataset")
    if any(
        not np.array_equal(getattr(checkpoint, name), getattr(dataset, name))
        for name in array_fields
    ):
        raise ValueError("Phase-7 normalization values differ from its training dataset")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
