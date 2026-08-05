"""Controlled Phase-6D accuracy ablations, cache, and checkpoint."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from r2dn_dc_motor.data import (
    NormalizationStatistics,
    Phase4Dataset,
    R2DNWindowBatch,
    R2DNWindowSampler,
)
from r2dn_dc_motor.data.phase4_dataset import canonical_sha256
from r2dn_dc_motor.models.r2dn_adapter import OfficialR2DNAdapter, R2DNArchitecture
from r2dn_dc_motor.models.r2dn_phase6b import _load_run_cache, _save_run_cache
from r2dn_dc_motor.models.r2dn_training import (
    R2DNRunResult,
    evaluate_validation_rollout,
    train_r2dn_run,
)
from r2dn_dc_motor.phase6_spec import Phase6Spec, load_phase6_spec
from r2dn_dc_motor.phase6d_spec import (
    AccuracyVariant,
    Phase6DSpec,
    load_phase6d_spec,
)

PHASE6D_ALGORITHM_VERSION = "phase6d_training_v1_aligned_validation_targets"


@dataclass(frozen=True)
class VariantAggregate:
    """Three-seed accuracy evidence for one controlled variant."""

    name: str
    latent_size: int
    burn_in_steps: int
    curriculum: str
    seeds: tuple[int, ...]
    scores: tuple[float, ...]
    median_validation_free_rollout_nrmse: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase6DStudy:
    """All accuracy-ablation runs and the locked validation selection."""

    profile_name: str
    dataset_fingerprint: str
    protocol_sha256: str
    runs: tuple[R2DNRunResult, ...]
    aggregates: tuple[VariantAggregate, ...]
    selected_variant: AccuracyVariant
    selected_run: R2DNRunResult
    target_combined_nrmse: float

    @property
    def target_met(self) -> bool:
        return self.selected_run.selection_score <= self.target_combined_nrmse

    def history_payload(self, *, phase6d: Phase6DSpec) -> dict[str, Any]:
        profile = phase6d.profile(self.profile_name)
        return {
            "schema_version": 1,
            "phase": "6D",
            "algorithm_version": PHASE6D_ALGORITHM_VERSION,
            "profile": self.profile_name,
            "dataset_fingerprint": self.dataset_fingerprint,
            "protocol_sha256": self.protocol_sha256,
            "selection_split": "validation",
            "selection_metric": "validation_free_rollout_nrmse",
            "selection_aggregation": "median_across_seeds",
            "tie_relative_tolerance": profile.tie_relative_tolerance,
            "validation_window_seed": profile.validation_window_seed,
            "selection_horizon_steps": profile.selection_horizon_steps,
            "validation_windows": profile.validation_windows,
            "variants": [value.to_dict() for value in self.aggregates],
            "runs": [value.summary() for value in self.runs],
            "selected_variant": self.selected_variant.name,
            "selected_seed": self.selected_run.seed,
            "selected_run_history": list(self.selected_run.history),
            "target_combined_nrmse": self.target_combined_nrmse,
            "target_met": self.target_met,
        }


@dataclass(frozen=True)
class Phase6DCheckpointManifest:
    """Hash-bound selected Phase-6D checkpoint metadata."""

    schema_version: int
    phase: str
    model_type: str
    training_profile: str
    selected_variant: str
    dataset_fingerprint: str
    protocol_sha256: str
    latent_size: int
    feature_size: int
    hidden_sizes: tuple[int, ...]
    initialization: str
    polar_parameterization: bool
    seed: int
    burn_in_steps: int
    selection_horizon_steps: int
    selection_validation_seed: int
    selection_metric: str
    variant_median_validation_nrmse: float
    validation_free_rollout_nrmse: float
    contractivity_margin: float
    update_count: int
    target_combined_nrmse: float
    target_met: bool
    parameter_sha256: str
    normalization_sha256: str
    study_history_sha256: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Phase6DCheckpointManifest:
        return cls(
            schema_version=int(raw["schema_version"]),
            phase=str(raw["phase"]),
            model_type=str(raw["model_type"]),
            training_profile=str(raw["training_profile"]),
            selected_variant=str(raw["selected_variant"]),
            dataset_fingerprint=str(raw["dataset_fingerprint"]),
            protocol_sha256=str(raw["protocol_sha256"]),
            latent_size=int(raw["latent_size"]),
            feature_size=int(raw["feature_size"]),
            hidden_sizes=tuple(int(value) for value in raw["hidden_sizes"]),
            initialization=str(raw["initialization"]),
            polar_parameterization=bool(raw["polar_parameterization"]),
            seed=int(raw["seed"]),
            burn_in_steps=int(raw["burn_in_steps"]),
            selection_horizon_steps=int(raw["selection_horizon_steps"]),
            selection_validation_seed=int(raw["selection_validation_seed"]),
            selection_metric=str(raw["selection_metric"]),
            variant_median_validation_nrmse=float(
                raw["variant_median_validation_nrmse"]
            ),
            validation_free_rollout_nrmse=float(
                raw["validation_free_rollout_nrmse"]
            ),
            contractivity_margin=float(raw["contractivity_margin"]),
            update_count=int(raw["update_count"]),
            target_combined_nrmse=float(raw["target_combined_nrmse"]),
            target_met=bool(raw["target_met"]),
            parameter_sha256=str(raw["parameter_sha256"]),
            normalization_sha256=str(raw["normalization_sha256"]),
            study_history_sha256=str(raw["study_history_sha256"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(
        self,
        *,
        dataset: Phase4Dataset,
        phase6: Phase6Spec,
        phase6d: Phase6DSpec,
    ) -> None:
        errors: list[str] = []
        if self.schema_version != 1 or self.phase != "6D":
            errors.append("Phase-6D checkpoint schema or phase mismatch")
        if self.model_type != "ContractingR2DN":
            errors.append("Phase-6D checkpoint model type mismatch")
        if self.dataset_fingerprint != dataset.fingerprint:
            errors.append("Phase-6D checkpoint belongs to another dataset")
        expected_protocol = phase6d_protocol_sha256(phase6d, phase6)
        if self.protocol_sha256 != expected_protocol:
            errors.append("Phase-6D checkpoint protocol hash mismatch")
        profile = phase6d.profile(self.training_profile)
        variants = {value.name: value for value in profile.variants}
        variant = variants.get(self.selected_variant)
        if variant is None:
            errors.append("selected Phase-6D variant is not declared")
        else:
            if self.latent_size != variant.latent_size:
                errors.append("Phase-6D checkpoint latent mismatch")
            if self.burn_in_steps != variant.burn_in_steps:
                errors.append("Phase-6D checkpoint burn-in mismatch")
        if self.seed not in profile.seeds:
            errors.append("Phase-6D checkpoint seed is not declared")
        if self.feature_size != int(phase6.architecture["feature_size"]):
            errors.append("Phase-6D checkpoint feature width mismatch")
        if self.hidden_sizes != tuple(phase6.architecture["hidden_sizes"]):
            errors.append("Phase-6D checkpoint hidden widths mismatch")
        if self.initialization != phase6.architecture["initialization"]:
            errors.append("Phase-6D checkpoint initialization mismatch")
        if self.polar_parameterization is not True:
            errors.append("Phase-6D checkpoint disabled contraction parameterization")
        if self.selection_horizon_steps != profile.selection_horizon_steps:
            errors.append("Phase-6D checkpoint validation horizon mismatch")
        if self.selection_validation_seed != profile.validation_window_seed:
            errors.append("Phase-6D checkpoint validation seed mismatch")
        if self.selection_metric != "validation_free_rollout_nrmse":
            errors.append("Phase-6D checkpoint selection metric mismatch")
        numeric = (
            self.variant_median_validation_nrmse,
            self.validation_free_rollout_nrmse,
            self.contractivity_margin,
            self.target_combined_nrmse,
        )
        if not all(math.isfinite(value) for value in numeric):
            errors.append("Phase-6D checkpoint contains non-finite metrics")
        if min(self.variant_median_validation_nrmse, self.validation_free_rollout_nrmse) < 0:
            errors.append("Phase-6D validation NRMSE must be non-negative")
        if self.contractivity_margin <= 0.0:
            errors.append("Phase-6D contractivity margin is not positive")
        if self.target_met != (
            self.validation_free_rollout_nrmse <= self.target_combined_nrmse
        ):
            errors.append("Phase-6D target flag is inconsistent")
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
class LoadedPhase6DCheckpoint:
    """Validated selected Phase-6D parameters and evidence."""

    adapter: OfficialR2DNAdapter
    parameters: Any
    manifest: Phase6DCheckpointManifest
    normalization: NormalizationStatistics
    study_history: dict[str, Any]


def phase6d_protocol_sha256(phase6d: Phase6DSpec, phase6: Phase6Spec) -> str:
    """Hash every rule that changes Phase-6D training or selection."""

    return canonical_sha256(
        {
            "algorithm_version": PHASE6D_ALGORITHM_VERSION,
            "phase6": {
                "interface": phase6.interface,
                "architecture": phase6.architecture,
                "loss": phase6.loss,
                "optimizer": phase6.optimizer,
            },
            "phase6d": {
                "phase": phase6d.phase,
                "selection": phase6d.selection,
                "profiles": {
                    name: asdict(profile) for name, profile in phase6d.profiles.items()
                },
                "curricula": {
                    name: [asdict(stage) for stage in stages]
                    for name, stages in phase6d.curricula.items()
                },
                "validation": phase6d.validation,
            },
        }
    )


def aligned_validation_window(
    maximum_burn_in_window: R2DNWindowBatch,
    *,
    maximum_burn_in_steps: int,
    burn_in_steps: int,
    rollout_steps: int,
) -> R2DNWindowBatch:
    """Shorten only history so every burn-in compares the identical targets."""

    if not 1 <= burn_in_steps <= maximum_burn_in_steps:
        raise ValueError("burn-in must lie in [1, maximum_burn_in_steps]")
    required = maximum_burn_in_steps + rollout_steps
    if maximum_burn_in_window.transitions != required:
        raise ValueError("maximum-burn-in window has an unexpected horizon")
    offset = maximum_burn_in_steps - burn_in_steps
    return R2DNWindowBatch(
        observations=maximum_burn_in_window.observations[offset:],
        controls=maximum_burn_in_window.controls[offset:],
        trajectory_ids=maximum_burn_in_window.trajectory_ids,
        start_steps=tuple(
            int(value) + offset for value in maximum_burn_in_window.start_steps
        ),
        split=maximum_burn_in_window.split,
    )


def select_variant_from_medians(
    aggregates: tuple[VariantAggregate, ...] | list[VariantAggregate],
    *,
    relative_tolerance: float,
) -> str:
    """Select the first, simpler declared variant within tolerance of the best."""

    if not aggregates:
        raise ValueError("at least one Phase-6D aggregate is required")
    if not 0.0 <= relative_tolerance < 1.0:
        raise ValueError("relative tolerance must lie in [0, 1)")
    scores = [value.median_validation_free_rollout_nrmse for value in aggregates]
    if any(not math.isfinite(value) or value < 0.0 for value in scores):
        raise ValueError("variant medians must be finite and non-negative")
    threshold = min(scores) * (1.0 + relative_tolerance)
    return next(
        value.name
        for value in aggregates
        if value.median_validation_free_rollout_nrmse <= threshold
    )


def train_phase6d_study(
    dataset: Phase4Dataset,
    *,
    phase6d: Phase6DSpec | None = None,
    phase6: Phase6Spec | None = None,
    profile_name: str = "ci",
    cache_directory: Path | str = Path("checkpoints/phase6d/run-cache-v1"),
    overwrite_cache: bool = False,
    progress: Any | None = None,
) -> Phase6DStudy:
    """Train/resume every declared variant and select on aligned targets."""

    phase6 = phase6 or load_phase6_spec()
    phase6d = phase6d or load_phase6d_spec(phase6=phase6)
    profile = phase6d.profile(profile_name)
    base_profile = phase6.profile(profile.base_phase6_profile)
    protocol = phase6d_protocol_sha256(phase6d, phase6)
    cache = Path(cache_directory)
    progress = progress or (lambda _: None)
    maximum_burn_in = max(value.burn_in_steps for value in profile.variants)
    sampler = R2DNWindowSampler(dataset, split="validation", seed=0)
    maximum_window = sampler.fixed_validation_windows(
        count=profile.validation_windows,
        burn_in_steps=maximum_burn_in,
        rollout_steps=profile.selection_horizon_steps,
        seed=profile.validation_window_seed,
    )

    all_runs: list[R2DNRunResult] = []
    aggregates: list[VariantAggregate] = []
    for variant in profile.variants:
        role = f"phase6d_{variant.name}"
        validation_window = aligned_validation_window(
            maximum_window,
            maximum_burn_in_steps=maximum_burn_in,
            burn_in_steps=variant.burn_in_steps,
            rollout_steps=profile.selection_horizon_steps,
        )
        variant_profile = replace(
            base_profile,
            name=f"phase6d-{profile.name}-{variant.name}",
            candidate_latent_sizes=(variant.latent_size,),
            training_seeds=profile.seeds,
            burn_in_steps=variant.burn_in_steps,
            selection_horizon_steps=profile.selection_horizon_steps,
            selection_validation_seed=profile.validation_window_seed,
            validation_windows=profile.validation_windows,
        )
        variant_runs: list[R2DNRunResult] = []
        for seed in profile.seeds:
            path = cache / profile.name / variant.name / f"seed-{seed}"
            if path.is_dir() and not overwrite_cache:
                run = _load_run_cache(
                    path,
                    dataset=dataset,
                    phase6=phase6,
                    protocol_sha256=protocol,
                    latent_size=variant.latent_size,
                    seed=seed,
                    role=role,
                    validation_horizon_steps=profile.selection_horizon_steps,
                )
                progress(
                    f"cache hit {variant.name} seed={seed}: "
                    f"validation NRMSE={run.selection_score:.6g}"
                )
            else:
                stages = phase6d.stages(variant)
                progress(
                    f"train {variant.name} latent={variant.latent_size} "
                    f"burn-in={variant.burn_in_steps} seed={seed}, "
                    f"updates={sum(stage.updates for stage in stages)}"
                )
                run = train_r2dn_run(
                    dataset,
                    spec=phase6,
                    profile=variant_profile,
                    latent_size=variant.latent_size,
                    seed=seed,
                    stages=stages,
                    validation_horizon_steps=profile.selection_horizon_steps,
                    validation_window_seed=profile.validation_window_seed,
                    role=role,
                    progress=progress,
                )
                adapter = OfficialR2DNAdapter(run.architecture)
                aligned_metrics = evaluate_validation_rollout(
                    adapter,
                    run.parameters,
                    validation_window,
                    burn_in_steps=variant.burn_in_steps,
                )
                run = replace(run, validation=aligned_metrics)
                _save_run_cache(
                    path,
                    run=run,
                    dataset=dataset,
                    protocol_sha256=protocol,
                    validation_window_seed=profile.validation_window_seed,
                    overwrite=overwrite_cache,
                )
            variant_runs.append(run)
            all_runs.append(run)
        scores = tuple(value.selection_score for value in variant_runs)
        aggregate = VariantAggregate(
            name=variant.name,
            latent_size=variant.latent_size,
            burn_in_steps=variant.burn_in_steps,
            curriculum=variant.curriculum,
            seeds=profile.seeds,
            scores=scores,
            median_validation_free_rollout_nrmse=float(median(scores)),
        )
        aggregates.append(aggregate)
        progress(
            f"aggregate {variant.name}: median validation "
            f"NRMSE={aggregate.median_validation_free_rollout_nrmse:.6g}"
        )

    selected_name = select_variant_from_medians(
        aggregates,
        relative_tolerance=profile.tie_relative_tolerance,
    )
    selected_variant = next(value for value in profile.variants if value.name == selected_name)
    selected_runs = [
        value for value in all_runs if value.role == f"phase6d_{selected_name}"
    ]
    selected_run = min(selected_runs, key=lambda value: value.selection_score)
    progress(
        f"selected {selected_name}, seed={selected_run.seed}, "
        f"validation NRMSE={selected_run.selection_score:.6g}"
    )
    return Phase6DStudy(
        profile_name=profile_name,
        dataset_fingerprint=dataset.fingerprint,
        protocol_sha256=protocol,
        runs=tuple(all_runs),
        aggregates=tuple(aggregates),
        selected_variant=selected_variant,
        selected_run=selected_run,
        target_combined_nrmse=profile.target_combined_nrmse,
    )


def save_phase6d_checkpoint(
    directory: Path | str,
    *,
    dataset: Phase4Dataset,
    study: Phase6DStudy,
    phase6d: Phase6DSpec | None = None,
    phase6: Phase6Spec | None = None,
    overwrite: bool = False,
) -> Phase6DCheckpointManifest:
    """Atomically save the selected Phase-6D run and complete evidence."""

    try:
        from flax import serialization
    except ImportError as error:
        raise ImportError("Phase-6D checkpoint saving requires Phase-6 dependencies") from error
    phase6 = phase6 or load_phase6_spec()
    phase6d = phase6d or load_phase6d_spec(phase6=phase6)
    profile = phase6d.profile(study.profile_name)
    aggregate = next(
        value for value in study.aggregates if value.name == study.selected_variant.name
    )
    output = Path(directory)
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Phase-6D checkpoint already exists: {output}; pass overwrite=True explicitly"
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
        history_path = staging / "study_history.json"
        history_path.write_text(
            json.dumps(
                study.history_payload(phase6d=phase6d),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        selected = study.selected_run
        architecture = selected.architecture
        manifest = Phase6DCheckpointManifest(
            schema_version=1,
            phase="6D",
            model_type="ContractingR2DN",
            training_profile=study.profile_name,
            selected_variant=study.selected_variant.name,
            dataset_fingerprint=dataset.fingerprint,
            protocol_sha256=study.protocol_sha256,
            latent_size=architecture.state_size,
            feature_size=architecture.features,
            hidden_sizes=architecture.hidden,
            initialization=architecture.init_method,
            polar_parameterization=architecture.do_polar_param,
            seed=selected.seed,
            burn_in_steps=study.selected_variant.burn_in_steps,
            selection_horizon_steps=profile.selection_horizon_steps,
            selection_validation_seed=profile.validation_window_seed,
            selection_metric="validation_free_rollout_nrmse",
            variant_median_validation_nrmse=(
                aggregate.median_validation_free_rollout_nrmse
            ),
            validation_free_rollout_nrmse=selected.selection_score,
            contractivity_margin=selected.contractivity_margin,
            update_count=selected.update_count,
            target_combined_nrmse=study.target_combined_nrmse,
            target_met=study.target_met,
            parameter_sha256=_file_sha256(parameter_path),
            normalization_sha256=_file_sha256(normalization_path),
            study_history_sha256=_file_sha256(history_path),
        )
        manifest.validate(dataset=dataset, phase6=phase6, phase6d=phase6d)
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


def load_phase6d_checkpoint(
    directory: Path | str,
    *,
    dataset: Phase4Dataset,
    phase6d: Phase6DSpec | None = None,
    phase6: Phase6Spec | None = None,
) -> LoadedPhase6DCheckpoint:
    """Load a Phase-6D checkpoint after validating provenance and hashes."""

    try:
        from flax import serialization
    except ImportError as error:
        raise ImportError("Phase-6D checkpoint loading requires Phase-6 dependencies") from error
    phase6 = phase6 or load_phase6_spec()
    phase6d = phase6d or load_phase6d_spec(phase6=phase6)
    root = Path(directory)
    required = (
        "manifest.json",
        "parameters.msgpack",
        "normalization.npz",
        "study_history.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Phase-6D checkpoint is missing: {', '.join(missing)}")
    manifest = Phase6DCheckpointManifest.from_dict(
        json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    )
    manifest.validate(dataset=dataset, phase6=phase6, phase6d=phase6d)
    for path, expected in (
        (root / "parameters.msgpack", manifest.parameter_sha256),
        (root / "normalization.npz", manifest.normalization_sha256),
        (root / "study_history.json", manifest.study_history_sha256),
    ):
        if _file_sha256(path) != expected:
            raise ValueError(f"Phase-6D checkpoint content hash mismatch: {path.name}")
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
    parameters = serialization.from_bytes(
        template,
        (root / "parameters.msgpack").read_bytes(),
    )
    history = json.loads((root / "study_history.json").read_text(encoding="utf-8"))
    if history.get("protocol_sha256") != manifest.protocol_sha256:
        raise ValueError("Phase-6D history protocol hash mismatch")
    if history.get("dataset_fingerprint") != dataset.fingerprint:
        raise ValueError("Phase-6D history belongs to another dataset")
    if history.get("selected_variant") != manifest.selected_variant:
        raise ValueError("Phase-6D history selected variant mismatch")
    if int(history.get("selected_seed", -1)) != manifest.seed:
        raise ValueError("Phase-6D history selected seed mismatch")
    return LoadedPhase6DCheckpoint(
        adapter=adapter,
        parameters=parameters,
        manifest=manifest,
        normalization=normalization,
        study_history=history,
    )


def _validate_normalization(
    checkpoint: NormalizationStatistics,
    dataset: NormalizationStatistics,
) -> None:
    scalar_fields = ("observation_count", "control_count", "fit_split")
    array_fields = (
        "observation_mean",
        "observation_std",
        "control_mean",
        "control_std",
    )
    if any(getattr(checkpoint, name) != getattr(dataset, name) for name in scalar_fields):
        raise ValueError("Phase-6D normalization metadata differs from Phase 4")
    if any(
        not np.array_equal(getattr(checkpoint, name), getattr(dataset, name))
        for name in array_fields
    ):
        raise ValueError("Phase-6D normalization values differ from Phase 4")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
