"""Phase-6B repeated latent search, resumable runs, and selected checkpoint."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from r2dn_dc_motor.data import NormalizationStatistics, Phase4Dataset
from r2dn_dc_motor.data.phase4_dataset import canonical_sha256
from r2dn_dc_motor.models.jax_runtime import inspect_jax_runtime
from r2dn_dc_motor.models.r2dn_adapter import OfficialR2DNAdapter, R2DNArchitecture
from r2dn_dc_motor.models.r2dn_training import (
    LoadedPhase6Checkpoint,
    R2DNRunResult,
    ValidationRolloutMetrics,
    load_phase6_checkpoint,
    train_r2dn_run,
)
from r2dn_dc_motor.phase1_spec import load_phase1_spec
from r2dn_dc_motor.phase6_spec import Phase6Spec, load_phase6_spec
from r2dn_dc_motor.phase6b_spec import Phase6BProfile, Phase6BSpec, load_phase6b_spec

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class PilotAggregate:
    """Three-seed pilot evidence for one latent dimension."""

    latent_size: int
    seeds: tuple[int, ...]
    scores: tuple[float, ...]
    median_validation_free_rollout_nrmse: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase6BStudy:
    """Repeated pilot search followed by final checkpoint selection or reuse."""

    profile_name: str
    dataset_fingerprint: str
    protocol_sha256: str
    evaluated_latent_sizes: tuple[int, ...]
    adaptive_candidate_triggered: bool
    pilot_runs: tuple[R2DNRunResult, ...]
    pilot_aggregates: tuple[PilotAggregate, ...]
    selected_latent_size: int
    final_run_summaries: tuple[dict[str, Any], ...]
    selected_run: R2DNRunResult
    selected_run_source: str

    def history_payload(
        self,
        *,
        phase6: Phase6Spec,
        phase6b: Phase6BSpec,
    ) -> dict[str, Any]:
        profile = phase6b.profile(self.profile_name)
        base = phase6.profile(profile.base_phase6_profile)
        return {
            "schema_version": 1,
            "phase": "6B",
            "profile": self.profile_name,
            "dataset_fingerprint": self.dataset_fingerprint,
            "protocol_sha256": self.protocol_sha256,
            "selection_split": "validation",
            "selection_metric": "median_validation_free_rollout_nrmse",
            "selection_mode": "min",
            "near_tie_policy": "smallest_within_relative_tolerance",
            "tie_relative_tolerance": profile.tie_relative_tolerance,
            "evaluated_latent_sizes": list(self.evaluated_latent_sizes),
            "adaptive_candidate_triggered": self.adaptive_candidate_triggered,
            "pilot_runs": [run.summary() for run in self.pilot_runs],
            "pilot_aggregates": [
                aggregate.to_dict() for aggregate in self.pilot_aggregates
            ],
            "selected_latent_size": self.selected_latent_size,
            "final_selection_metric": "validation_free_rollout_nrmse",
            "final_runs": list(self.final_run_summaries),
            "selected_seed": self.selected_run.seed,
            "selected_run_source": self.selected_run_source,
            "selected_run_history": list(self.selected_run.history),
            "training_configuration": {
                "phase6_architecture": dict(phase6.architecture),
                "phase6_loss": dict(phase6.loss),
                "phase6_optimizer": dict(phase6.optimizer),
                "phase6_selection": dict(phase6.selection),
                "phase6_pilot_stages": [asdict(stage) for stage in base.pilot_stages],
                "phase6_final_stages": [asdict(stage) for stage in base.final_stages],
                "phase6b_profile": asdict(profile),
                "phase6b_search": dict(phase6b.search),
                "phase6b_stress": {
                    key: value
                    for key, value in phase6b.stress.items()
                    if key != "scenarios"
                },
                "stress_scenarios": list(phase6b.scenarios),
                "runtime": inspect_jax_runtime().to_dict(),
            },
        }


@dataclass(frozen=True)
class Phase6BCheckpointManifest:
    """Bindings for the selected extended-search checkpoint."""

    schema_version: int
    phase: str
    model_type: str
    training_profile: str
    selected_run_source: str
    dataset_fingerprint: str
    protocol_sha256: str
    upstream_commit: str
    evaluated_latent_sizes: tuple[int, ...]
    pilot_seeds: tuple[int, ...]
    tie_relative_tolerance: float
    adaptive_candidate_triggered: bool
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
    pilot_median_validation_nrmse: float
    validation_free_rollout_nrmse: float
    contractivity_margin: float
    parameter_sha256: str
    normalization_sha256: str
    study_history_sha256: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Phase6BCheckpointManifest:
        return cls(
            schema_version=int(raw["schema_version"]),
            phase=str(raw["phase"]),
            model_type=str(raw["model_type"]),
            training_profile=str(raw["training_profile"]),
            selected_run_source=str(raw["selected_run_source"]),
            dataset_fingerprint=str(raw["dataset_fingerprint"]),
            protocol_sha256=str(raw["protocol_sha256"]),
            upstream_commit=str(raw["upstream_commit"]),
            evaluated_latent_sizes=tuple(int(value) for value in raw["evaluated_latent_sizes"]),
            pilot_seeds=tuple(int(value) for value in raw["pilot_seeds"]),
            tie_relative_tolerance=float(raw["tie_relative_tolerance"]),
            adaptive_candidate_triggered=bool(raw["adaptive_candidate_triggered"]),
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
            pilot_median_validation_nrmse=float(
                raw["pilot_median_validation_nrmse"]
            ),
            validation_free_rollout_nrmse=float(
                raw["validation_free_rollout_nrmse"]
            ),
            contractivity_margin=float(raw["contractivity_margin"]),
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
        phase6b: Phase6BSpec,
    ) -> None:
        profile = phase6b.profile(self.training_profile)
        base = phase6.profile(profile.base_phase6_profile)
        errors: list[str] = []
        if self.schema_version != 1 or self.phase != "6B":
            errors.append("Phase-6B checkpoint schema or phase mismatch")
        if self.model_type != "ContractingR2DN":
            errors.append("Phase-6B checkpoint model type changed")
        if self.selected_run_source not in {
            "phase6b_final_training",
            "phase6_checkpoint_reuse",
        }:
            errors.append("Phase-6B checkpoint source is unsupported")
        if self.dataset_fingerprint != dataset.fingerprint:
            errors.append("Phase-6B checkpoint belongs to another dataset")
        expected_protocol = phase6b_protocol_sha256(phase6b, phase6)
        if self.protocol_sha256 != expected_protocol:
            errors.append("Phase-6B checkpoint protocol hash mismatch")
        if self.upstream_commit != load_phase1_spec().upstream["commit"]:
            errors.append("Phase-6B checkpoint upstream commit mismatch")
        expected_latents = list(profile.candidate_latent_sizes)
        if profile.adaptive_enabled and self.adaptive_candidate_triggered:
            expected_latents.append(profile.adaptive_candidate_latent)
        if self.evaluated_latent_sizes != tuple(expected_latents):
            errors.append("Phase-6B evaluated latent catalog is incomplete")
        if self.pilot_seeds != profile.pilot_seeds:
            errors.append("Phase-6B pilot seed catalog mismatch")
        if not math.isclose(
            self.tie_relative_tolerance,
            profile.tie_relative_tolerance,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            errors.append("Phase-6B tie rule mismatch")
        if self.latent_size not in self.evaluated_latent_sizes:
            errors.append("selected Phase-6B latent was not evaluated")
        if self.feature_size != int(phase6.architecture["feature_size"]):
            errors.append("Phase-6B feature size drifted from Phase 6")
        if self.hidden_sizes != tuple(phase6.architecture["hidden_sizes"]):
            errors.append("Phase-6B hidden widths drifted from Phase 6")
        if self.initialization != phase6.architecture["initialization"]:
            errors.append("Phase-6B initialization drifted from Phase 6")
        if self.polar_parameterization is not True:
            errors.append("Phase-6B checkpoint disabled polar parameterization")
        if self.seed not in profile.final_training_seeds:
            errors.append("Phase-6B selected seed is not a locked final seed")
        if self.burn_in_steps != base.burn_in_steps:
            errors.append("Phase-6B burn-in mismatch")
        if self.selection_horizon_steps != profile.selection_horizon_steps:
            errors.append("Phase-6B final selection horizon mismatch")
        if self.selection_validation_seed != profile.selection_validation_seed:
            errors.append("Phase-6B final validation windows mismatch")
        if self.selection_metric != "validation_free_rollout_nrmse":
            errors.append("Phase-6B final selection metric changed")
        numeric = (
            self.pilot_median_validation_nrmse,
            self.validation_free_rollout_nrmse,
            self.contractivity_margin,
        )
        if not all(math.isfinite(value) for value in numeric):
            errors.append("Phase-6B checkpoint metrics are non-finite")
        if min(
            self.pilot_median_validation_nrmse,
            self.validation_free_rollout_nrmse,
        ) < 0.0:
            errors.append("Phase-6B NRMSE values must be non-negative")
        if self.contractivity_margin <= 0.0:
            errors.append("Phase-6B contractivity certificate is not positive")
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
class LoadedPhase6BCheckpoint:
    """Validated selected parameters plus all Phase-6B search evidence."""

    adapter: OfficialR2DNAdapter
    parameters: Any
    manifest: Phase6BCheckpointManifest
    normalization: NormalizationStatistics
    study_history: dict[str, Any]


def phase6b_protocol_sha256(phase6b: Phase6BSpec, phase6: Phase6Spec) -> str:
    """Hash every rule that can affect Phase-6B training or evaluation."""

    payload = {
        "phase6": {
            "interface": phase6.interface,
            "architecture": phase6.architecture,
            "loss": phase6.loss,
            "optimizer": phase6.optimizer,
            "selection": phase6.selection,
            "profiles": {
                name: asdict(profile) for name, profile in phase6.profiles.items()
            },
        },
        "phase6b": {
            "phase": phase6b.phase,
            "search": phase6b.search,
            "stress": phase6b.stress,
            "profiles": {
                name: asdict(profile) for name, profile in phase6b.profiles.items()
            },
            "validation": phase6b.validation,
        },
    }
    return canonical_sha256(payload)


def aggregate_pilot_runs(
    runs: tuple[R2DNRunResult, ...] | list[R2DNRunResult],
) -> tuple[PilotAggregate, ...]:
    """Aggregate fixed-seed pilot scores by latent size using the median."""

    grouped: dict[int, list[R2DNRunResult]] = {}
    for run in runs:
        grouped.setdefault(run.architecture.state_size, []).append(run)
    aggregates = []
    for latent_size in sorted(grouped):
        latent_runs = sorted(grouped[latent_size], key=lambda run: run.seed)
        scores = tuple(run.selection_score for run in latent_runs)
        aggregates.append(
            PilotAggregate(
                latent_size=latent_size,
                seeds=tuple(run.seed for run in latent_runs),
                scores=scores,
                median_validation_free_rollout_nrmse=float(median(scores)),
            )
        )
    return tuple(aggregates)


def select_latent_from_medians(
    medians: Mapping[int, float],
    *,
    relative_tolerance: float,
) -> int:
    """Choose the smallest latent whose median is within tolerance of the best."""

    if not medians:
        raise ValueError("at least one latent median is required")
    if not 0.0 <= relative_tolerance < 1.0:
        raise ValueError("relative tolerance must lie in [0, 1)")
    if any(not math.isfinite(score) or score < 0.0 for score in medians.values()):
        raise ValueError("latent medians must be finite and non-negative")
    best = min(medians.values())
    threshold = best * (1.0 + relative_tolerance)
    return min(latent for latent, score in medians.items() if score <= threshold)


def adaptive_candidate_required(
    medians: Mapping[int, float],
    *,
    reference_latent: int,
    boundary_latent: int,
    improvement_threshold: float,
) -> bool:
    """Apply the predeclared boundary-improvement trigger."""

    reference = float(medians[reference_latent])
    boundary = float(medians[boundary_latent])
    return boundary < reference * (1.0 - improvement_threshold)


def train_phase6b_study(
    dataset: Phase4Dataset,
    *,
    phase6b: Phase6BSpec | None = None,
    phase6: Phase6Spec | None = None,
    profile_name: str = "ci",
    cache_directory: Path | str = Path("checkpoints/phase6b/run-cache-v1"),
    reusable_phase6_checkpoint: Path | str | None = None,
    overwrite_cache: bool = False,
    progress: ProgressCallback | None = None,
) -> Phase6BStudy:
    """Run or resume repeated pilots, then train or reuse the selected final model."""

    phase6b = phase6b or load_phase6b_spec()
    phase6 = phase6 or load_phase6_spec()
    profile = phase6b.profile(profile_name)
    base = phase6.profile(profile.base_phase6_profile)
    protocol = phase6b_protocol_sha256(phase6b, phase6)
    cache = Path(cache_directory)
    progress = progress or (lambda _: None)

    pilot_runs: list[R2DNRunResult] = []
    for latent_size in profile.candidate_latent_sizes:
        for seed in profile.pilot_seeds:
            run_path = cache / "pilot" / f"latent-{latent_size:03d}" / f"seed-{seed}"
            pilot_runs.append(
                _load_or_train_run(
                    run_path,
                    dataset=dataset,
                    phase6=phase6,
                    base_profile=base,
                    protocol_sha256=protocol,
                    latent_size=latent_size,
                    seed=seed,
                    role="phase6b_pilot",
                    stages=base.pilot_stages,
                    validation_horizon_steps=profile.pilot_validation_horizon_steps,
                    validation_window_seed=profile.pilot_validation_seed,
                    overwrite=overwrite_cache,
                    progress=progress,
                )
            )

    aggregates = aggregate_pilot_runs(pilot_runs)
    median_map = {
        aggregate.latent_size: aggregate.median_validation_free_rollout_nrmse
        for aggregate in aggregates
    }
    adaptive_triggered = False
    if profile.adaptive_enabled:
        adaptive_triggered = adaptive_candidate_required(
            median_map,
            reference_latent=profile.adaptive_reference_latent,
            boundary_latent=profile.adaptive_boundary_latent,
            improvement_threshold=profile.adaptive_improvement_threshold,
        )
        progress(
            "adaptive latent "
            f"{profile.adaptive_candidate_latent}: "
            f"{'TRIGGERED' if adaptive_triggered else 'not triggered'}"
        )
        if adaptive_triggered:
            latent_size = profile.adaptive_candidate_latent
            for seed in profile.pilot_seeds:
                run_path = (
                    cache / "pilot" / f"latent-{latent_size:03d}" / f"seed-{seed}"
                )
                pilot_runs.append(
                    _load_or_train_run(
                        run_path,
                        dataset=dataset,
                        phase6=phase6,
                        base_profile=base,
                        protocol_sha256=protocol,
                        latent_size=latent_size,
                        seed=seed,
                        role="phase6b_pilot",
                        stages=base.pilot_stages,
                        validation_horizon_steps=profile.pilot_validation_horizon_steps,
                        validation_window_seed=profile.pilot_validation_seed,
                        overwrite=overwrite_cache,
                        progress=progress,
                    )
                )
            aggregates = aggregate_pilot_runs(pilot_runs)
            median_map = {
                aggregate.latent_size: aggregate.median_validation_free_rollout_nrmse
                for aggregate in aggregates
            }

    selected_latent = select_latent_from_medians(
        median_map,
        relative_tolerance=profile.tie_relative_tolerance,
    )
    progress(
        f"pilot selected latent={selected_latent}, "
        f"median validation NRMSE={median_map[selected_latent]:.6g}"
    )

    selected_run: R2DNRunResult | None = None
    final_summaries: tuple[dict[str, Any], ...] = ()
    source = "phase6b_final_training"
    reusable_path = (
        Path(reusable_phase6_checkpoint)
        if reusable_phase6_checkpoint is not None
        else None
    )
    if (
        reusable_path is not None
        and reusable_path.is_dir()
        and phase6b.search["reuse_phase6_checkpoint_when_selected_latent_matches"]
    ):
        reusable = load_phase6_checkpoint(
            reusable_path,
            dataset=dataset,
            spec=phase6,
        )
        if reusable.manifest.latent_size == selected_latent:
            selected_run, final_summaries = _reuse_phase6_final(reusable, profile)
            source = "phase6_checkpoint_reuse"
            progress(
                f"reused validated Phase-6 latent={selected_latent}, "
                f"seed={selected_run.seed}"
            )

    if selected_run is None:
        final_runs: list[R2DNRunResult] = []
        for seed in profile.final_training_seeds:
            run_path = (
                cache / "final" / f"latent-{selected_latent:03d}" / f"seed-{seed}"
            )
            final_runs.append(
                _load_or_train_run(
                    run_path,
                    dataset=dataset,
                    phase6=phase6,
                    base_profile=base,
                    protocol_sha256=protocol,
                    latent_size=selected_latent,
                    seed=seed,
                    role="phase6b_final",
                    stages=base.final_stages,
                    validation_horizon_steps=profile.selection_horizon_steps,
                    validation_window_seed=profile.selection_validation_seed,
                    overwrite=overwrite_cache,
                    progress=progress,
                )
            )
        selected_run = min(final_runs, key=lambda run: run.selection_score)
        final_summaries = tuple(run.summary() for run in final_runs)

    progress(
        f"selected final latent={selected_latent}, seed={selected_run.seed}, "
        f"validation NRMSE={selected_run.selection_score:.6g}"
    )
    evaluated_latents = tuple(aggregate.latent_size for aggregate in aggregates)
    return Phase6BStudy(
        profile_name=profile_name,
        dataset_fingerprint=dataset.fingerprint,
        protocol_sha256=protocol,
        evaluated_latent_sizes=evaluated_latents,
        adaptive_candidate_triggered=adaptive_triggered,
        pilot_runs=tuple(pilot_runs),
        pilot_aggregates=aggregates,
        selected_latent_size=selected_latent,
        final_run_summaries=final_summaries,
        selected_run=selected_run,
        selected_run_source=source,
    )


def save_phase6b_checkpoint(
    directory: Path | str,
    *,
    dataset: Phase4Dataset,
    study: Phase6BStudy,
    phase6b: Phase6BSpec | None = None,
    phase6: Phase6Spec | None = None,
    overwrite: bool = False,
) -> Phase6BCheckpointManifest:
    """Atomically save the selected model and complete Phase-6B search evidence."""

    try:
        from flax import serialization
    except ImportError as error:
        raise ImportError("Phase-6B checkpoint saving requires Phase-6 dependencies") from error

    phase6b = phase6b or load_phase6b_spec()
    phase6 = phase6 or load_phase6_spec()
    profile = phase6b.profile(study.profile_name)
    base = phase6.profile(profile.base_phase6_profile)
    output = Path(directory)
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Phase-6B checkpoint already exists: {output}; "
            "pass overwrite=True explicitly"
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
        history = study.history_payload(phase6=phase6, phase6b=phase6b)
        history_path.write_text(
            json.dumps(history, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        selected_median = next(
            value.median_validation_free_rollout_nrmse
            for value in study.pilot_aggregates
            if value.latent_size == study.selected_latent_size
        )
        selected = study.selected_run
        manifest = Phase6BCheckpointManifest(
            schema_version=1,
            phase="6B",
            model_type="ContractingR2DN",
            training_profile=study.profile_name,
            selected_run_source=study.selected_run_source,
            dataset_fingerprint=dataset.fingerprint,
            protocol_sha256=study.protocol_sha256,
            upstream_commit=str(load_phase1_spec().upstream["commit"]),
            evaluated_latent_sizes=study.evaluated_latent_sizes,
            pilot_seeds=profile.pilot_seeds,
            tie_relative_tolerance=profile.tie_relative_tolerance,
            adaptive_candidate_triggered=study.adaptive_candidate_triggered,
            latent_size=selected.architecture.state_size,
            feature_size=selected.architecture.features,
            hidden_sizes=selected.architecture.hidden,
            initialization=selected.architecture.init_method,
            polar_parameterization=selected.architecture.do_polar_param,
            seed=selected.seed,
            burn_in_steps=base.burn_in_steps,
            selection_horizon_steps=profile.selection_horizon_steps,
            selection_validation_seed=profile.selection_validation_seed,
            selection_metric="validation_free_rollout_nrmse",
            pilot_median_validation_nrmse=selected_median,
            validation_free_rollout_nrmse=selected.selection_score,
            contractivity_margin=selected.contractivity_margin,
            parameter_sha256=_file_sha256(parameter_path),
            normalization_sha256=_file_sha256(normalization_path),
            study_history_sha256=_file_sha256(history_path),
        )
        manifest.validate(
            dataset=dataset,
            phase6=phase6,
            phase6b=phase6b,
        )
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


def load_phase6b_checkpoint(
    directory: Path | str,
    *,
    dataset: Phase4Dataset,
    phase6b: Phase6BSpec | None = None,
    phase6: Phase6Spec | None = None,
) -> LoadedPhase6BCheckpoint:
    """Load the extended-search checkpoint only after all bindings pass."""

    try:
        from flax import serialization
    except ImportError as error:
        raise ImportError("Phase-6B checkpoint loading requires Phase-6 dependencies") from error

    phase6b = phase6b or load_phase6b_spec()
    phase6 = phase6 or load_phase6_spec()
    root = Path(directory)
    required = (
        "manifest.json",
        "parameters.msgpack",
        "normalization.npz",
        "study_history.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Phase-6B checkpoint is missing: {', '.join(missing)}")
    manifest = Phase6BCheckpointManifest.from_dict(
        json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    )
    manifest.validate(dataset=dataset, phase6=phase6, phase6b=phase6b)
    for path, expected in (
        (root / "parameters.msgpack", manifest.parameter_sha256),
        (root / "normalization.npz", manifest.normalization_sha256),
        (root / "study_history.json", manifest.study_history_sha256),
    ):
        if _file_sha256(path) != expected:
            raise ValueError(f"Phase-6B checkpoint content hash mismatch: {path.name}")
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
        raise ValueError("Phase-6B history protocol hash mismatch")
    if history.get("dataset_fingerprint") != dataset.fingerprint:
        raise ValueError("Phase-6B history belongs to another dataset")
    if int(history.get("selected_latent_size", -1)) != manifest.latent_size:
        raise ValueError("Phase-6B history selected latent mismatch")
    if int(history.get("selected_seed", -1)) != manifest.seed:
        raise ValueError("Phase-6B history selected seed mismatch")
    return LoadedPhase6BCheckpoint(
        adapter=adapter,
        parameters=parameters,
        manifest=manifest,
        normalization=normalization,
        study_history=history,
    )


def _load_or_train_run(
    path: Path,
    *,
    dataset: Phase4Dataset,
    phase6: Phase6Spec,
    base_profile: Any,
    protocol_sha256: str,
    latent_size: int,
    seed: int,
    role: str,
    stages: Any,
    validation_horizon_steps: int,
    validation_window_seed: int,
    overwrite: bool,
    progress: ProgressCallback,
) -> R2DNRunResult:
    if path.is_dir() and not overwrite:
        run = _load_run_cache(
            path,
            dataset=dataset,
            phase6=phase6,
            protocol_sha256=protocol_sha256,
            latent_size=latent_size,
            seed=seed,
            role=role,
            validation_horizon_steps=validation_horizon_steps,
        )
        progress(
            f"cache hit {role} latent={latent_size} seed={seed}: "
            f"validation NRMSE={run.selection_score:.6g}"
        )
        return run
    progress(
        f"train {role} latent={latent_size} seed={seed}, "
        f"updates={sum(stage.updates for stage in stages)}"
    )
    run = train_r2dn_run(
        dataset,
        spec=phase6,
        profile=base_profile,
        latent_size=latent_size,
        seed=seed,
        stages=stages,
        validation_horizon_steps=validation_horizon_steps,
        validation_window_seed=validation_window_seed,
        role=role,
        progress=progress,
    )
    _save_run_cache(
        path,
        run=run,
        dataset=dataset,
        protocol_sha256=protocol_sha256,
        validation_window_seed=validation_window_seed,
        overwrite=overwrite,
    )
    return run


def _save_run_cache(
    path: Path,
    *,
    run: R2DNRunResult,
    dataset: Phase4Dataset,
    protocol_sha256: str,
    validation_window_seed: int,
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
            "validation_window_seed": validation_window_seed,
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
    phase6: Phase6Spec,
    protocol_sha256: str,
    latent_size: int,
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
    if int(summary.get("latent_size", -1)) != latent_size:
        raise ValueError(f"run cache latent mismatch: {path}")
    if int(summary.get("seed", -1)) != seed or summary.get("role") != role:
        raise ValueError(f"run cache seed or role mismatch: {path}")
    validation = ValidationRolloutMetrics(**summary["validation"])
    if validation.horizon_steps != validation_horizon_steps:
        raise ValueError(f"run cache validation horizon mismatch: {path}")
    parameter_path = path / "parameters.msgpack"
    if _file_sha256(parameter_path) != payload["parameter_sha256"]:
        raise ValueError(f"run cache parameter hash mismatch: {path}")
    raw_architecture = payload["architecture"]
    architecture = R2DNArchitecture(
        input_size=int(raw_architecture["input_size"]),
        state_size=int(raw_architecture["state_size"]),
        features=int(raw_architecture["features"]),
        output_size=int(raw_architecture["output_size"]),
        hidden=tuple(int(value) for value in raw_architecture["hidden"]),
        init_method=str(raw_architecture["init_method"]),
        do_polar_param=bool(raw_architecture["do_polar_param"]),
    )
    if architecture.features != int(phase6.architecture["feature_size"]):
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


def _reuse_phase6_final(
    checkpoint: LoadedPhase6Checkpoint,
    profile: Phase6BProfile,
) -> tuple[R2DNRunResult, tuple[dict[str, Any], ...]]:
    summaries = tuple(checkpoint.training_history.get("final_runs", ()))
    recorded_seeds = tuple(int(value["seed"]) for value in summaries)
    if recorded_seeds != profile.final_training_seeds:
        raise ValueError("reusable Phase-6 checkpoint final seed catalog mismatch")
    selected_summary = next(
        (value for value in summaries if int(value["seed"]) == checkpoint.manifest.seed),
        None,
    )
    if selected_summary is None:
        raise ValueError("reusable Phase-6 checkpoint lacks selected-run summary")
    scores = tuple(float(value["validation"]["free_rollout_nrmse"]) for value in summaries)
    if not math.isclose(
        float(selected_summary["validation"]["free_rollout_nrmse"]),
        min(scores),
        rel_tol=1e-6,
        abs_tol=1e-8,
    ):
        raise ValueError("reusable Phase-6 checkpoint is not the best recorded final seed")
    validation = ValidationRolloutMetrics(**selected_summary["validation"])
    run = R2DNRunResult(
        parameters=checkpoint.parameters,
        architecture=checkpoint.adapter.architecture,
        seed=checkpoint.manifest.seed,
        role="phase6_checkpoint_reuse",
        update_count=checkpoint.manifest.update_count,
        contractivity_margin=checkpoint.manifest.contractivity_margin,
        validation=validation,
        history=tuple(checkpoint.training_history["selected_run_history"]),
    )
    return run, summaries


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
        raise ValueError("Phase-6B normalization metadata differs from Phase 4")
    if any(
        not np.array_equal(getattr(checkpoint, name), getattr(dataset, name))
        for name in array_fields
    ):
        raise ValueError("Phase-6B normalization values differ from Phase 4")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
