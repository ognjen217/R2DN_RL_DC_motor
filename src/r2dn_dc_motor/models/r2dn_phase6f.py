"""Phase-6F optimizer-floor ablation and checkpoint support."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from r2dn_dc_motor.data import NormalizationStatistics, Phase4Dataset
from r2dn_dc_motor.data.phase4_dataset import canonical_sha256
from r2dn_dc_motor.models.r2dn_adapter import OfficialR2DNAdapter, R2DNArchitecture
from r2dn_dc_motor.models.r2dn_phase6b import _load_run_cache, _save_run_cache
from r2dn_dc_motor.models.r2dn_phase6e import (
    LatentRunEvidence,
    LoadedPhase6ECheckpoint,
    build_selection_cases,
    evaluate_run_on_selection_cases,
    load_phase6e_checkpoint,
)
from r2dn_dc_motor.models.r2dn_training import (
    R2DNRunResult,
    ValidationRolloutMetrics,
    train_r2dn_run_with_optimizer,
)
from r2dn_dc_motor.phase2_spec import Phase2Spec, load_phase2_spec
from r2dn_dc_motor.phase6_spec import Phase6Spec, load_phase6_spec
from r2dn_dc_motor.phase6e_spec import Phase6ESpec, load_phase6e_spec
from r2dn_dc_motor.phase6f_spec import (
    OptimizerVariant,
    Phase6FSpec,
    load_phase6f_spec,
    phase6f_protocol_payload,
)

PHASE6F_ALGORITHM_VERSION = "phase6f_optimizer_floor_ablation_v1"


@dataclass(frozen=True)
class OptimizerVariantEvidence:
    """One baseline or trained optimizer candidate plus selection evidence."""

    variant: OptimizerVariant
    run: R2DNRunResult
    selection: LatentRunEvidence

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.variant.name,
            "source": self.variant.source,
            "schedule": self.variant.schedule,
            "initial_learning_rate": self.variant.initial_learning_rate,
            "final_learning_rate": self.variant.final_learning_rate,
            "stage_update_multiplier": self.variant.stage_update_multiplier,
            **self.run.summary(),
            "phase4_validation_free_rollout_nrmse": self.run.selection_score,
            "selection_multisine_median_combined_nrmse": (
                self.selection.median_combined_nrmse
            ),
            "selection_multisine_median_current_nrmse": (
                self.selection.median_current_nrmse
            ),
            "selection_multisine_median_speed_nrmse": (
                self.selection.median_speed_nrmse
            ),
            "multisine_metrics": [
                asdict(value) for value in self.selection.multisine_metrics
            ],
        }


@dataclass(frozen=True)
class Phase6FStudy:
    """Complete baseline/schedule/budget comparison."""

    profile_name: str
    dataset_fingerprint: str
    protocol_sha256: str
    phase6b_report_sha256: str
    phase6e_manifest_sha256: str
    variants: tuple[OptimizerVariantEvidence, ...]
    selected_variant_name: str
    target_combined_nrmse: float
    tie_relative_tolerance: float

    @property
    def selected(self) -> OptimizerVariantEvidence:
        return next(
            value for value in self.variants if value.variant.name == self.selected_variant_name
        )

    @property
    def baseline(self) -> OptimizerVariantEvidence:
        return next(
            value
            for value in self.variants
            if value.variant.source == "phase6e_checkpoint"
        )

    @property
    def target_met(self) -> bool:
        return self.selected.selection.median_combined_nrmse <= self.target_combined_nrmse

    @property
    def relative_improvement_over_baseline(self) -> float:
        baseline = self.baseline.selection.median_combined_nrmse
        selected = self.selected.selection.median_combined_nrmse
        return (baseline - selected) / baseline

    def history_payload(self, *, phase6f: Phase6FSpec) -> dict[str, Any]:
        profile = phase6f.profile(self.profile_name)
        return {
            "schema_version": 1,
            "phase": "6F",
            "algorithm_version": PHASE6F_ALGORITHM_VERSION,
            "profile": self.profile_name,
            "dataset_fingerprint": self.dataset_fingerprint,
            "protocol_sha256": self.protocol_sha256,
            "phase6b_report_sha256": self.phase6b_report_sha256,
            "phase6e_manifest_sha256": self.phase6e_manifest_sha256,
            "selection_metric": phase6f.selection["run_metric"],
            "tie_relative_tolerance": self.tie_relative_tolerance,
            "selection_duration_s": profile.selection_duration_s,
            "selection_split_anchor_indices": list(profile.selection_anchor_indices),
            "selection_scenarios": [
                value.to_dict() for value in profile.multisine_scenarios
            ],
            "variants": [value.summary() for value in self.variants],
            "selected_variant": self.selected_variant_name,
            "selected_run_history": list(self.selected.run.history),
            "relative_improvement_over_baseline": self.relative_improvement_over_baseline,
            "target_combined_nrmse": self.target_combined_nrmse,
            "target_met": self.target_met,
        }


@dataclass(frozen=True)
class Phase6FCheckpointManifest:
    """Hash-bound metadata for the selected optimizer candidate."""

    schema_version: int
    phase: str
    model_type: str
    training_profile: str
    selected_variant: str
    selected_source: str
    dataset_fingerprint: str
    protocol_sha256: str
    phase6b_report_sha256: str
    phase6e_manifest_sha256: str
    latent_size: int
    feature_size: int
    hidden_sizes: tuple[int, ...]
    initialization: str
    polar_parameterization: bool
    seed: int
    burn_in_steps: int
    selection_duration_s: float
    selection_metric: str
    learning_rate_schedule: str
    initial_learning_rate: float
    final_learning_rate: float
    stage_update_multiplier: int
    validation_free_rollout_nrmse: float
    current_validation_nrmse: float
    speed_validation_nrmse: float
    phase4_validation_nrmse: float
    baseline_validation_nrmse: float
    relative_improvement_over_baseline: float
    contractivity_margin: float
    update_count: int
    target_combined_nrmse: float
    target_met: bool
    parameter_sha256: str
    normalization_sha256: str
    study_history_sha256: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Phase6FCheckpointManifest:
        payload = dict(raw)
        payload["hidden_sizes"] = tuple(int(value) for value in raw["hidden_sizes"])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(
        self,
        *,
        dataset: Phase4Dataset,
        phase6f: Phase6FSpec,
        phase6: Phase6Spec,
    ) -> None:
        errors: list[str] = []
        profile = phase6f.profile(self.training_profile)
        base = phase6.profile(profile.base_phase6_profile)
        variants = {value.name: value for value in profile.variants}
        variant = variants.get(self.selected_variant)
        if self.schema_version != 1 or self.phase != "6F":
            errors.append("Phase-6F checkpoint schema or phase mismatch")
        if self.model_type != "ContractingR2DN":
            errors.append("Phase-6F checkpoint model type mismatch")
        if self.dataset_fingerprint != dataset.fingerprint:
            errors.append("Phase-6F checkpoint belongs to another dataset")
        if self.protocol_sha256 != phase6f_protocol_sha256(phase6f, phase6):
            errors.append("Phase-6F checkpoint protocol hash mismatch")
        if variant is None:
            errors.append("Phase-6F selected optimizer variant is undeclared")
        else:
            if self.selected_source != variant.source:
                errors.append("Phase-6F selected source mismatch")
            if self.learning_rate_schedule != variant.schedule:
                errors.append("Phase-6F learning-rate schedule mismatch")
            if not math.isclose(self.initial_learning_rate, variant.initial_learning_rate):
                errors.append("Phase-6F initial learning rate mismatch")
            if not math.isclose(self.final_learning_rate, variant.final_learning_rate):
                errors.append("Phase-6F final learning rate mismatch")
            if self.stage_update_multiplier != variant.stage_update_multiplier:
                errors.append("Phase-6F update multiplier mismatch")
            expected_updates = sum(stage.updates for stage in base.final_stages) * (
                variant.stage_update_multiplier
            )
            if self.update_count != expected_updates:
                errors.append("Phase-6F checkpoint update count mismatch")
        if self.latent_size != profile.latent_size or self.seed != profile.seed:
            errors.append("Phase-6F latent or seed drifted")
        if self.feature_size != int(phase6.architecture["feature_size"]):
            errors.append("Phase-6F feature width mismatch")
        if self.hidden_sizes != tuple(phase6.architecture["hidden_sizes"]):
            errors.append("Phase-6F hidden widths mismatch")
        if self.initialization != phase6.architecture["initialization"]:
            errors.append("Phase-6F initialization mismatch")
        if self.polar_parameterization is not True:
            errors.append("Phase-6F contraction parameterization was disabled")
        if self.burn_in_steps != profile.burn_in_steps:
            errors.append("Phase-6F burn-in mismatch")
        if not math.isclose(self.selection_duration_s, profile.selection_duration_s):
            errors.append("Phase-6F selection horizon mismatch")
        if self.selection_metric != phase6f.selection["run_metric"]:
            errors.append("Phase-6F selection metric mismatch")
        numeric = (
            self.validation_free_rollout_nrmse,
            self.current_validation_nrmse,
            self.speed_validation_nrmse,
            self.phase4_validation_nrmse,
            self.baseline_validation_nrmse,
            self.relative_improvement_over_baseline,
            self.contractivity_margin,
            self.target_combined_nrmse,
        )
        if not all(math.isfinite(value) for value in numeric):
            errors.append("Phase-6F checkpoint contains non-finite metrics")
        if self.contractivity_margin <= 0.0:
            errors.append("Phase-6F contractivity margin is not positive")
        if self.target_met != (
            self.validation_free_rollout_nrmse <= self.target_combined_nrmse
        ):
            errors.append("Phase-6F target flag is inconsistent")
        for name, digest in (
            ("Phase-6B report", self.phase6b_report_sha256),
            ("Phase-6E manifest", self.phase6e_manifest_sha256),
            ("parameter", self.parameter_sha256),
            ("normalization", self.normalization_sha256),
            ("study history", self.study_history_sha256),
        ):
            if len(digest) != 64:
                errors.append(f"{name} SHA-256 is malformed")
        if errors:
            raise ValueError("\n".join(errors))


@dataclass(frozen=True)
class LoadedPhase6FCheckpoint:
    """Validated Phase-6F parameters and optimization evidence."""

    adapter: OfficialR2DNAdapter
    parameters: Any
    manifest: Phase6FCheckpointManifest
    normalization: NormalizationStatistics
    study_history: dict[str, Any]


def phase6f_protocol_sha256(
    phase6f: Phase6FSpec,
    phase6: Phase6Spec,
    phase6e: Phase6ESpec | None = None,
) -> str:
    """Hash every rule that changes Phase-6F training or selection."""

    phase6e = phase6e or load_phase6e_spec(phase6=phase6)
    return canonical_sha256(
        {
            "algorithm_version": PHASE6F_ALGORITHM_VERSION,
            **phase6f_protocol_payload(phase6f, phase6, phase6e),
        }
    )


def select_optimizer_variant(
    scores: dict[str, float],
    ordered_names: tuple[str, ...],
    *,
    relative_tolerance: float,
) -> str:
    """Prefer the lowest training budget whose score is near the best."""

    if set(scores) != set(ordered_names) or not scores:
        raise ValueError("optimizer selection scores do not match the declared catalog")
    if not all(math.isfinite(value) and value >= 0.0 for value in scores.values()):
        raise ValueError("optimizer selection scores must be finite and non-negative")
    best = min(scores.values())
    threshold = best * (1.0 + relative_tolerance)
    return next(name for name in ordered_names if scores[name] <= threshold)


def _baseline_run(checkpoint: LoadedPhase6ECheckpoint) -> R2DNRunResult:
    manifest = checkpoint.manifest
    summary = next(
        (
            value
            for value in checkpoint.study_history["runs"]
            if int(value["latent_size"]) == manifest.latent_size
            and int(value["seed"]) == manifest.seed
        ),
        None,
    )
    if summary is None:
        raise ValueError("Phase-6E checkpoint history lacks its selected run")
    validation = ValidationRolloutMetrics(**summary["validation"])
    return R2DNRunResult(
        parameters=checkpoint.parameters,
        architecture=checkpoint.adapter.architecture,
        seed=manifest.seed,
        role="baseline_phase6e",
        update_count=manifest.update_count,
        contractivity_margin=manifest.contractivity_margin,
        validation=validation,
        history=tuple(checkpoint.study_history["selected_run_history"]),
    )


def train_phase6f_study(
    dataset: Phase4Dataset,
    phase6b_report_path: Path | str,
    phase6e_checkpoint_directory: Path | str,
    *,
    phase6f: Phase6FSpec | None = None,
    phase6e: Phase6ESpec | None = None,
    phase6: Phase6Spec | None = None,
    phase2: Phase2Spec | None = None,
    profile_name: str = "final",
    cache_directory: Path | str = Path("checkpoints/phase6f/run-cache-v1"),
    overwrite_cache: bool = False,
    progress: Any | None = None,
) -> Phase6FStudy:
    """Evaluate the Phase-6E baseline and train the two controlled variants."""

    phase2 = phase2 or load_phase2_spec()
    phase6 = phase6 or load_phase6_spec()
    phase6e = phase6e or load_phase6e_spec(phase2=phase2, phase6=phase6)
    phase6f = phase6f or load_phase6f_spec(
        phase2=phase2,
        phase6=phase6,
        phase6e=phase6e,
    )
    profile = phase6f.profile(profile_name)
    base = phase6.profile(profile.base_phase6_profile)
    protocol = phase6f_protocol_sha256(phase6f, phase6, phase6e)
    progress = progress or (lambda _: None)
    checkpoint_directory = Path(phase6e_checkpoint_directory)
    baseline_checkpoint = load_phase6e_checkpoint(
        checkpoint_directory,
        dataset=dataset,
        phase6e=phase6e,
        phase6=phase6,
    )
    if (
        baseline_checkpoint.manifest.latent_size != profile.latent_size
        or baseline_checkpoint.manifest.seed != profile.seed
    ):
        raise ValueError("Phase-6F baseline must be the latent-16/seed-43 Phase-6E winner")
    cases = build_selection_cases(
        dataset,
        phase6b_report_path,
        profile=profile,
        phase2=phase2,
        progress=progress,
    )
    evidences: list[OptimizerVariantEvidence] = []
    for variant in profile.variants:
        if variant.source == "phase6e_checkpoint":
            run = _baseline_run(baseline_checkpoint)
            progress(
                f"reuse {variant.name}: updates={run.update_count}, "
                f"Phase-4 validation NRMSE={run.selection_score:.6g}"
            )
        else:
            stages = tuple(
                replace(
                    stage,
                    updates=stage.updates * variant.stage_update_multiplier,
                )
                for stage in base.final_stages
            )
            expected_updates = sum(stage.updates for stage in stages)
            role = f"phase6f_{variant.name}"
            path = Path(cache_directory) / profile.name / variant.name / f"seed-{profile.seed}"
            if path.is_dir() and not overwrite_cache:
                run = _load_run_cache(
                    path,
                    dataset=dataset,
                    phase6=phase6,
                    protocol_sha256=protocol,
                    latent_size=profile.latent_size,
                    seed=profile.seed,
                    role=role,
                    validation_horizon_steps=base.selection_horizon_steps,
                )
                if run.update_count != expected_updates:
                    raise ValueError(f"Phase-6F cache update-count mismatch: {path}")
                progress(f"cache hit {variant.name}: NRMSE={run.selection_score:.6g}")
            else:
                training_profile = replace(
                    base,
                    name=f"phase6f-{profile.name}-{variant.name}",
                    candidate_latent_sizes=(profile.latent_size,),
                    training_seeds=(profile.seed,),
                )
                progress(
                    f"train {variant.name}: latent={profile.latent_size}, "
                    f"seed={profile.seed}, updates={expected_updates}, "
                    f"LR={variant.initial_learning_rate:g}->{variant.final_learning_rate:g}"
                )
                run = train_r2dn_run_with_optimizer(
                    dataset,
                    spec=phase6,
                    profile=training_profile,
                    latent_size=profile.latent_size,
                    seed=profile.seed,
                    stages=stages,
                    validation_horizon_steps=base.selection_horizon_steps,
                    validation_window_seed=base.selection_validation_seed,
                    role=role,
                    optimizer=variant.optimizer_payload(phase6),
                    progress=progress,
                )
                _save_run_cache(
                    path,
                    run=run,
                    dataset=dataset,
                    protocol_sha256=protocol,
                    validation_window_seed=base.selection_validation_seed,
                    overwrite=overwrite_cache,
                )
        selection = evaluate_run_on_selection_cases(
            run,
            dataset,
            cases,
            profile=profile,
        )
        evidence = OptimizerVariantEvidence(
            variant=variant,
            run=run,
            selection=selection,
        )
        evidences.append(evidence)
        progress(
            f"selection {variant.name}: combined/current/speed="
            f"{selection.median_combined_nrmse:.6g}/"
            f"{selection.median_current_nrmse:.6g}/"
            f"{selection.median_speed_nrmse:.6g}"
        )
    ordered = tuple(value.name for value in profile.variants)
    selected = select_optimizer_variant(
        {
            value.variant.name: value.selection.median_combined_nrmse
            for value in evidences
        },
        ordered,
        relative_tolerance=profile.tie_relative_tolerance,
    )
    progress(f"selected optimizer variant: {selected}")
    return Phase6FStudy(
        profile_name=profile_name,
        dataset_fingerprint=dataset.fingerprint,
        protocol_sha256=protocol,
        phase6b_report_sha256=_file_sha256(Path(phase6b_report_path)),
        phase6e_manifest_sha256=_file_sha256(checkpoint_directory / "manifest.json"),
        variants=tuple(evidences),
        selected_variant_name=selected,
        target_combined_nrmse=profile.target_combined_nrmse,
        tie_relative_tolerance=profile.tie_relative_tolerance,
    )


def save_phase6f_checkpoint(
    directory: Path | str,
    *,
    dataset: Phase4Dataset,
    study: Phase6FStudy,
    phase6f: Phase6FSpec | None = None,
    phase6e: Phase6ESpec | None = None,
    phase6: Phase6Spec | None = None,
    overwrite: bool = False,
) -> Phase6FCheckpointManifest:
    """Atomically save the selected optimizer candidate and complete evidence."""

    try:
        from flax import serialization
    except ImportError as error:
        raise ImportError("Phase-6F checkpoint saving requires Phase-6 dependencies") from error
    phase6 = phase6 or load_phase6_spec()
    phase6e = phase6e or load_phase6e_spec(phase6=phase6)
    phase6f = phase6f or load_phase6f_spec(phase6=phase6, phase6e=phase6e)
    profile = phase6f.profile(study.profile_name)
    selected = study.selected
    output = Path(directory)
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Phase-6F checkpoint already exists: {output}; pass overwrite=True explicitly"
        )
    staging = output.parent / f".{output.name}.building"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        parameter_path = staging / "parameters.msgpack"
        parameter_path.write_bytes(serialization.to_bytes(selected.run.parameters))
        normalization_path = staging / "normalization.npz"
        dataset.normalization.save(normalization_path)
        history_path = staging / "study_history.json"
        history_path.write_text(
            json.dumps(study.history_payload(phase6f=phase6f), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        architecture = selected.run.architecture
        manifest = Phase6FCheckpointManifest(
            schema_version=1,
            phase="6F",
            model_type="ContractingR2DN",
            training_profile=study.profile_name,
            selected_variant=selected.variant.name,
            selected_source=selected.variant.source,
            dataset_fingerprint=dataset.fingerprint,
            protocol_sha256=study.protocol_sha256,
            phase6b_report_sha256=study.phase6b_report_sha256,
            phase6e_manifest_sha256=study.phase6e_manifest_sha256,
            latent_size=architecture.state_size,
            feature_size=architecture.features,
            hidden_sizes=architecture.hidden,
            initialization=architecture.init_method,
            polar_parameterization=architecture.do_polar_param,
            seed=selected.run.seed,
            burn_in_steps=profile.burn_in_steps,
            selection_duration_s=profile.selection_duration_s,
            selection_metric=str(phase6f.selection["run_metric"]),
            learning_rate_schedule=selected.variant.schedule,
            initial_learning_rate=selected.variant.initial_learning_rate,
            final_learning_rate=selected.variant.final_learning_rate,
            stage_update_multiplier=selected.variant.stage_update_multiplier,
            validation_free_rollout_nrmse=selected.selection.median_combined_nrmse,
            current_validation_nrmse=selected.selection.median_current_nrmse,
            speed_validation_nrmse=selected.selection.median_speed_nrmse,
            phase4_validation_nrmse=selected.run.selection_score,
            baseline_validation_nrmse=study.baseline.selection.median_combined_nrmse,
            relative_improvement_over_baseline=study.relative_improvement_over_baseline,
            contractivity_margin=selected.run.contractivity_margin,
            update_count=selected.run.update_count,
            target_combined_nrmse=study.target_combined_nrmse,
            target_met=study.target_met,
            parameter_sha256=_file_sha256(parameter_path),
            normalization_sha256=_file_sha256(normalization_path),
            study_history_sha256=_file_sha256(history_path),
        )
        manifest.validate(dataset=dataset, phase6f=phase6f, phase6=phase6)
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


def load_phase6f_checkpoint(
    directory: Path | str,
    *,
    dataset: Phase4Dataset,
    phase6f: Phase6FSpec | None = None,
    phase6e: Phase6ESpec | None = None,
    phase6: Phase6Spec | None = None,
) -> LoadedPhase6FCheckpoint:
    """Load a Phase-6F checkpoint after validating provenance and hashes."""

    try:
        from flax import serialization
    except ImportError as error:
        raise ImportError("Phase-6F checkpoint loading requires Phase-6 dependencies") from error
    phase6 = phase6 or load_phase6_spec()
    phase6e = phase6e or load_phase6e_spec(phase6=phase6)
    phase6f = phase6f or load_phase6f_spec(phase6=phase6, phase6e=phase6e)
    root = Path(directory)
    required = (
        "manifest.json",
        "parameters.msgpack",
        "normalization.npz",
        "study_history.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Phase-6F checkpoint is missing: {', '.join(missing)}")
    manifest = Phase6FCheckpointManifest.from_dict(
        json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    )
    manifest.validate(dataset=dataset, phase6f=phase6f, phase6=phase6)
    for path, expected in (
        (root / "parameters.msgpack", manifest.parameter_sha256),
        (root / "normalization.npz", manifest.normalization_sha256),
        (root / "study_history.json", manifest.study_history_sha256),
    ):
        if _file_sha256(path) != expected:
            raise ValueError(f"Phase-6F checkpoint content hash mismatch: {path.name}")
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
        raise ValueError("Phase-6F history protocol hash mismatch")
    if history.get("dataset_fingerprint") != dataset.fingerprint:
        raise ValueError("Phase-6F history belongs to another dataset")
    if history.get("selected_variant") != manifest.selected_variant:
        raise ValueError("Phase-6F history selected variant mismatch")
    return LoadedPhase6FCheckpoint(
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
        raise ValueError("Phase-6F normalization metadata differs from Phase 4")
    if any(
        not np.array_equal(getattr(checkpoint, name), getattr(dataset, name))
        for name in array_fields
    ):
        raise ValueError("Phase-6F normalization values differ from Phase 4")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
