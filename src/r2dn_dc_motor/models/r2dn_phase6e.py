"""Phase-6E full-curriculum larger-latent search and checkpoint support."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from typing import Any

import numpy as np

from r2dn_dc_motor.data import NormalizationStatistics, Phase4Dataset
from r2dn_dc_motor.data.phase4_dataset import canonical_sha256
from r2dn_dc_motor.models.r2dn_adapter import OfficialR2DNAdapter, R2DNArchitecture
from r2dn_dc_motor.models.r2dn_phase6b import (
    _load_run_cache,
    _save_run_cache,
    select_latent_from_medians,
)
from r2dn_dc_motor.models.r2dn_training import R2DNRunResult, train_r2dn_run
from r2dn_dc_motor.phase2_spec import Phase2Spec, load_phase2_spec
from r2dn_dc_motor.phase6_spec import Phase6Spec, load_phase6_spec
from r2dn_dc_motor.phase6e_spec import (
    Phase6EProfile,
    Phase6ESpec,
    load_phase6e_spec,
    phase6e_protocol_payload,
)
from r2dn_dc_motor.validation.r2dn_rk4_benchmark import (
    BenchmarkAnchorData,
    RK4Trace,
    build_voltage_trace,
    calculate_accuracy,
    load_benchmark_anchor,
    run_r2dn_trace,
    run_rk4_trace,
)

PHASE6E_ALGORITHM_VERSION = "phase6e_full_curriculum_multisine_selection_v1"
ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class SelectionCase:
    """One fixed multisine, anchor, voltage trace, and FULL/RK4 reference."""

    name: str
    scenario: dict[str, Any]
    anchor_index: int
    anchor: BenchmarkAnchorData
    voltages_v: np.ndarray
    rk4: RK4Trace


@dataclass(frozen=True)
class MultisineMetric:
    """One trained model evaluated against one held-out FULL/RK4 trace."""

    scenario_name: str
    anchor_index: int
    duration_s: float
    combined_nrmse: float
    current_nrmse: float
    speed_nrmse: float
    current_rmse_a: float
    speed_rmse_rad_s: float
    maximum_absolute_current_error_a: float
    maximum_absolute_speed_error_rad_s: float
    r2dn_warm_wall_time_s: float
    rk4_wall_time_s: float
    finite: bool


@dataclass(frozen=True)
class LatentRunEvidence:
    """One full-curriculum training run plus all selection rollouts."""

    run: R2DNRunResult
    multisine_metrics: tuple[MultisineMetric, ...]
    median_combined_nrmse: float
    median_current_nrmse: float
    median_speed_nrmse: float

    def summary(self) -> dict[str, Any]:
        return {
            **self.run.summary(),
            "phase4_validation_free_rollout_nrmse": self.run.selection_score,
            "selection_multisine_median_combined_nrmse": self.median_combined_nrmse,
            "selection_multisine_median_current_nrmse": self.median_current_nrmse,
            "selection_multisine_median_speed_nrmse": self.median_speed_nrmse,
            "multisine_metrics": [asdict(value) for value in self.multisine_metrics],
        }


@dataclass(frozen=True)
class LatentAggregate:
    """Three-seed selection evidence for one latent dimension."""

    latent_size: int
    seeds: tuple[int, ...]
    combined_scores: tuple[float, ...]
    current_scores: tuple[float, ...]
    speed_scores: tuple[float, ...]
    median_combined_nrmse: float
    median_current_nrmse: float
    median_speed_nrmse: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase6EStudy:
    """All larger-latent runs and the locked median/tie selection."""

    profile_name: str
    dataset_fingerprint: str
    protocol_sha256: str
    phase6b_report_sha256: str
    runs: tuple[LatentRunEvidence, ...]
    aggregates: tuple[LatentAggregate, ...]
    selected_latent_size: int
    selected_run: LatentRunEvidence
    target_combined_nrmse: float

    @property
    def selected_aggregate(self) -> LatentAggregate:
        return next(
            value for value in self.aggregates if value.latent_size == self.selected_latent_size
        )

    @property
    def target_met(self) -> bool:
        return self.selected_aggregate.median_combined_nrmse <= self.target_combined_nrmse

    def history_payload(self, *, phase6e: Phase6ESpec) -> dict[str, Any]:
        profile = phase6e.profile(self.profile_name)
        return {
            "schema_version": 1,
            "phase": "6E",
            "algorithm_version": PHASE6E_ALGORITHM_VERSION,
            "profile": self.profile_name,
            "dataset_fingerprint": self.dataset_fingerprint,
            "protocol_sha256": self.protocol_sha256,
            "phase6b_report_sha256": self.phase6b_report_sha256,
            "selection_split": phase6e.selection["selection_split"],
            "selection_reference": phase6e.selection["selection_reference"],
            "selection_metric": phase6e.selection["run_metric"],
            "latent_aggregation": phase6e.selection["latent_aggregation"],
            "tie_relative_tolerance": profile.tie_relative_tolerance,
            "selection_duration_s": profile.selection_duration_s,
            "selection_split_anchor_indices": list(profile.selection_anchor_indices),
            "selection_scenarios": [
                value.to_dict() for value in profile.multisine_scenarios
            ],
            "aggregates": [value.to_dict() for value in self.aggregates],
            "runs": [value.summary() for value in self.runs],
            "selected_latent_size": self.selected_latent_size,
            "selected_seed": self.selected_run.run.seed,
            "selected_run_history": list(self.selected_run.run.history),
            "target_combined_nrmse": self.target_combined_nrmse,
            "target_met": self.target_met,
        }


@dataclass(frozen=True)
class Phase6ECheckpointManifest:
    """Hash-bound metadata for the selected larger-latent checkpoint."""

    schema_version: int
    phase: str
    model_type: str
    training_profile: str
    selected_variant: str
    dataset_fingerprint: str
    protocol_sha256: str
    phase6b_report_sha256: str
    latent_size: int
    feature_size: int
    hidden_sizes: tuple[int, ...]
    initialization: str
    polar_parameterization: bool
    seed: int
    burn_in_steps: int
    selection_duration_s: float
    selection_metric: str
    latent_median_validation_nrmse: float
    validation_free_rollout_nrmse: float
    current_validation_nrmse: float
    speed_validation_nrmse: float
    phase4_validation_nrmse: float
    contractivity_margin: float
    update_count: int
    target_combined_nrmse: float
    target_met: bool
    parameter_sha256: str
    normalization_sha256: str
    study_history_sha256: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Phase6ECheckpointManifest:
        return cls(
            schema_version=int(raw["schema_version"]),
            phase=str(raw["phase"]),
            model_type=str(raw["model_type"]),
            training_profile=str(raw["training_profile"]),
            selected_variant=str(raw["selected_variant"]),
            dataset_fingerprint=str(raw["dataset_fingerprint"]),
            protocol_sha256=str(raw["protocol_sha256"]),
            phase6b_report_sha256=str(raw["phase6b_report_sha256"]),
            latent_size=int(raw["latent_size"]),
            feature_size=int(raw["feature_size"]),
            hidden_sizes=tuple(int(value) for value in raw["hidden_sizes"]),
            initialization=str(raw["initialization"]),
            polar_parameterization=bool(raw["polar_parameterization"]),
            seed=int(raw["seed"]),
            burn_in_steps=int(raw["burn_in_steps"]),
            selection_duration_s=float(raw["selection_duration_s"]),
            selection_metric=str(raw["selection_metric"]),
            latent_median_validation_nrmse=float(
                raw["latent_median_validation_nrmse"]
            ),
            validation_free_rollout_nrmse=float(raw["validation_free_rollout_nrmse"]),
            current_validation_nrmse=float(raw["current_validation_nrmse"]),
            speed_validation_nrmse=float(raw["speed_validation_nrmse"]),
            phase4_validation_nrmse=float(raw["phase4_validation_nrmse"]),
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
        phase6e: Phase6ESpec,
        phase6: Phase6Spec,
    ) -> None:
        errors: list[str] = []
        profile = phase6e.profile(self.training_profile)
        base = phase6.profile(profile.base_phase6_profile)
        if self.schema_version != 1 or self.phase != "6E":
            errors.append("Phase-6E checkpoint schema or phase mismatch")
        if self.model_type != "ContractingR2DN":
            errors.append("Phase-6E checkpoint model type mismatch")
        if self.dataset_fingerprint != dataset.fingerprint:
            errors.append("Phase-6E checkpoint belongs to another dataset")
        if self.protocol_sha256 != phase6e_protocol_sha256(phase6e, phase6):
            errors.append("Phase-6E checkpoint protocol hash mismatch")
        if self.latent_size not in profile.latent_sizes:
            errors.append("selected Phase-6E latent was not declared")
        if self.selected_variant != f"latent{self.latent_size}":
            errors.append("Phase-6E selected variant label mismatch")
        if self.feature_size != int(phase6.architecture["feature_size"]):
            errors.append("Phase-6E checkpoint feature width mismatch")
        if self.hidden_sizes != tuple(phase6.architecture["hidden_sizes"]):
            errors.append("Phase-6E checkpoint hidden widths mismatch")
        if self.initialization != phase6.architecture["initialization"]:
            errors.append("Phase-6E checkpoint initialization mismatch")
        if self.polar_parameterization is not True:
            errors.append("Phase-6E checkpoint disabled contraction parameterization")
        if self.seed not in profile.seeds:
            errors.append("Phase-6E checkpoint seed is not declared")
        if self.burn_in_steps != profile.burn_in_steps:
            errors.append("Phase-6E checkpoint burn-in mismatch")
        if not math.isclose(self.selection_duration_s, profile.selection_duration_s):
            errors.append("Phase-6E checkpoint selection horizon mismatch")
        if self.selection_metric != phase6e.selection["run_metric"]:
            errors.append("Phase-6E checkpoint selection metric mismatch")
        if self.update_count != sum(stage.updates for stage in base.final_stages):
            errors.append("Phase-6E checkpoint did not use the complete final curriculum")
        numeric = (
            self.latent_median_validation_nrmse,
            self.validation_free_rollout_nrmse,
            self.current_validation_nrmse,
            self.speed_validation_nrmse,
            self.phase4_validation_nrmse,
            self.contractivity_margin,
            self.target_combined_nrmse,
        )
        if not all(math.isfinite(value) for value in numeric):
            errors.append("Phase-6E checkpoint contains non-finite metrics")
        if min(numeric[:-2]) < 0.0:
            errors.append("Phase-6E NRMSE values must be non-negative")
        if self.contractivity_margin <= 0.0:
            errors.append("Phase-6E contractivity margin is not positive")
        if self.target_met != (
            self.latent_median_validation_nrmse <= self.target_combined_nrmse
        ):
            errors.append("Phase-6E target flag is inconsistent")
        for name, digest in (
            ("Phase-6B report", self.phase6b_report_sha256),
            ("parameter", self.parameter_sha256),
            ("normalization", self.normalization_sha256),
            ("study history", self.study_history_sha256),
        ):
            if len(digest) != 64:
                errors.append(f"{name} SHA-256 is malformed")
        if errors:
            raise ValueError("\n".join(errors))


@dataclass(frozen=True)
class LoadedPhase6ECheckpoint:
    """Validated Phase-6E parameters plus complete selection evidence."""

    adapter: OfficialR2DNAdapter
    parameters: Any
    manifest: Phase6ECheckpointManifest
    normalization: NormalizationStatistics
    study_history: dict[str, Any]


def phase6e_protocol_sha256(phase6e: Phase6ESpec, phase6: Phase6Spec) -> str:
    """Hash every rule that changes Phase-6E training or selection."""

    return canonical_sha256(
        {
            "algorithm_version": PHASE6E_ALGORITHM_VERSION,
            **phase6e_protocol_payload(phase6e, phase6),
        }
    )


def build_selection_cases(
    dataset: Phase4Dataset,
    phase6b_report_path: Path | str,
    *,
    profile: Phase6EProfile,
    phase2: Phase2Spec,
    progress: ProgressCallback | None = None,
) -> tuple[SelectionCase, ...]:
    """Materialize the three model-independent FULL/RK4 selection references."""

    progress = progress or (lambda _: None)
    placeholder = SimpleNamespace(
        normalization=dataset.normalization,
        manifest=SimpleNamespace(
            latent_size=-1,
            seed=-1,
            burn_in_steps=profile.burn_in_steps,
        ),
    )
    cases: list[SelectionCase] = []
    dt = phase2.integration_settings.control_period_s
    for scenario, anchor_index in zip(
        profile.multisine_scenarios,
        profile.selection_anchor_indices,
        strict=True,
    ):
        anchor, report = load_benchmark_anchor(
            dataset,
            phase6b_report_path,
            split=profile.selection_split,
            anchor_index=anchor_index,
            checkpoint=placeholder,
            require_checkpoint_match=False,
        )
        if not bool(report.get("passed", False)):
            raise ValueError("Phase-6E requires a passing Phase-6B report")
        payload = scenario.to_dict()
        voltages = build_voltage_trace(
            payload,
            duration_s=profile.selection_duration_s,
            control_period_s=dt,
        )
        progress(
            f"FULL/RK4 selection reference {scenario.name}, "
            f"anchor={anchor_index}, steps={voltages.size}"
        )
        rk4 = run_rk4_trace(
            phase2,
            anchor.initial_full_state,
            voltages,
            duration_s=profile.selection_duration_s,
        )
        if rk4.terminated or rk4.observations.shape[0] != voltages.size:
            raise RuntimeError(
                f"FULL/RK4 selection reference terminated for {scenario.name}: "
                f"{rk4.termination_reason}"
            )
        cases.append(
            SelectionCase(
                name=scenario.name,
                scenario=payload,
                anchor_index=anchor_index,
                anchor=anchor,
                voltages_v=voltages,
                rk4=rk4,
            )
        )
    return tuple(cases)


def evaluate_run_on_selection_cases(
    run: R2DNRunResult,
    dataset: Phase4Dataset,
    cases: tuple[SelectionCase, ...],
    *,
    profile: Phase6EProfile,
) -> LatentRunEvidence:
    """Evaluate one trained run on every locked held-out multisine case."""

    adapter = OfficialR2DNAdapter(run.architecture)
    checkpoint = SimpleNamespace(
        adapter=adapter,
        parameters=run.parameters,
        normalization=dataset.normalization,
        manifest=SimpleNamespace(
            latent_size=run.architecture.state_size,
            seed=run.seed,
            burn_in_steps=profile.burn_in_steps,
        ),
    )
    metrics: list[MultisineMetric] = []
    for case in cases:
        trace = run_r2dn_trace(
            checkpoint,
            case.anchor,
            case.voltages_v,
            duration_s=profile.selection_duration_s,
            chunk_steps=profile.selection_chunk_steps,
        )
        accuracy = calculate_accuracy(
            trace.observations,
            case.rk4.observations,
            dataset.normalization.observation_std,
            control_period_s=(
                profile.selection_duration_s / case.voltages_v.size
            ),
            requested_horizons_s=(profile.selection_duration_s,),
        )
        finite = bool(
            np.isfinite(trace.observations).all()
            and math.isfinite(accuracy.combined_nrmse)
        )
        metrics.append(
            MultisineMetric(
                scenario_name=case.name,
                anchor_index=case.anchor_index,
                duration_s=profile.selection_duration_s,
                combined_nrmse=accuracy.combined_nrmse,
                current_nrmse=accuracy.current_nrmse,
                speed_nrmse=accuracy.speed_nrmse,
                current_rmse_a=accuracy.current_rmse_a,
                speed_rmse_rad_s=accuracy.speed_rmse_rad_s,
                maximum_absolute_current_error_a=(
                    accuracy.maximum_absolute_current_error_a
                ),
                maximum_absolute_speed_error_rad_s=(
                    accuracy.maximum_absolute_speed_error_rad_s
                ),
                r2dn_warm_wall_time_s=trace.timing.warm_wall_time_s,
                rk4_wall_time_s=case.rk4.timing.wall_time_s,
                finite=finite,
            )
        )
    return LatentRunEvidence(
        run=run,
        multisine_metrics=tuple(metrics),
        median_combined_nrmse=float(median(value.combined_nrmse for value in metrics)),
        median_current_nrmse=float(median(value.current_nrmse for value in metrics)),
        median_speed_nrmse=float(median(value.speed_nrmse for value in metrics)),
    )


def train_phase6e_study(
    dataset: Phase4Dataset,
    phase6b_report_path: Path | str,
    *,
    phase6e: Phase6ESpec | None = None,
    phase6: Phase6Spec | None = None,
    phase2: Phase2Spec | None = None,
    profile_name: str = "final",
    cache_directory: Path | str = Path("checkpoints/phase6e/run-cache-v1"),
    overwrite_cache: bool = False,
    progress: ProgressCallback | None = None,
) -> Phase6EStudy:
    """Train/resume every latent/seed and select on held-out multisine rollouts."""

    phase2 = phase2 or load_phase2_spec()
    phase6 = phase6 or load_phase6_spec()
    phase6e = phase6e or load_phase6e_spec(phase2=phase2, phase6=phase6)
    profile = phase6e.profile(profile_name)
    base = phase6.profile(profile.base_phase6_profile)
    protocol = phase6e_protocol_sha256(phase6e, phase6)
    cache = Path(cache_directory)
    progress = progress or (lambda _: None)
    cases = build_selection_cases(
        dataset,
        phase6b_report_path,
        profile=profile,
        phase2=phase2,
        progress=progress,
    )
    report_sha256 = _file_sha256(Path(phase6b_report_path))
    evidences: list[LatentRunEvidence] = []
    aggregates: list[LatentAggregate] = []
    for latent_size in profile.latent_sizes:
        latent_runs: list[LatentRunEvidence] = []
        for seed in profile.seeds:
            role = f"phase6e_latent{latent_size}"
            path = cache / profile.name / f"latent-{latent_size:03d}" / f"seed-{seed}"
            if path.is_dir() and not overwrite_cache:
                run = _load_run_cache(
                    path,
                    dataset=dataset,
                    phase6=phase6,
                    protocol_sha256=protocol,
                    latent_size=latent_size,
                    seed=seed,
                    role=role,
                    validation_horizon_steps=base.selection_horizon_steps,
                )
                progress(
                    f"cache hit latent={latent_size}, seed={seed}: "
                    f"Phase-4 validation NRMSE={run.selection_score:.6g}"
                )
            else:
                training_profile = replace(
                    base,
                    name=f"phase6e-{profile.name}-latent{latent_size}",
                    candidate_latent_sizes=(latent_size,),
                    training_seeds=profile.seeds,
                )
                progress(
                    f"train latent={latent_size}, seed={seed}, burn-in={profile.burn_in_steps}, "
                    f"updates={sum(stage.updates for stage in base.final_stages)}"
                )
                run = train_r2dn_run(
                    dataset,
                    spec=phase6,
                    profile=training_profile,
                    latent_size=latent_size,
                    seed=seed,
                    stages=base.final_stages,
                    validation_horizon_steps=base.selection_horizon_steps,
                    validation_window_seed=base.selection_validation_seed,
                    role=role,
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
            progress(f"evaluate latent={latent_size}, seed={seed} on held-out multisine")
            evidence = evaluate_run_on_selection_cases(
                run,
                dataset,
                cases,
                profile=profile,
            )
            progress(
                f"selection latent={latent_size}, seed={seed}: "
                f"combined/current/speed={evidence.median_combined_nrmse:.6g}/"
                f"{evidence.median_current_nrmse:.6g}/"
                f"{evidence.median_speed_nrmse:.6g}"
            )
            latent_runs.append(evidence)
            evidences.append(evidence)
        combined = tuple(value.median_combined_nrmse for value in latent_runs)
        current = tuple(value.median_current_nrmse for value in latent_runs)
        speed = tuple(value.median_speed_nrmse for value in latent_runs)
        aggregate = LatentAggregate(
            latent_size=latent_size,
            seeds=profile.seeds,
            combined_scores=combined,
            current_scores=current,
            speed_scores=speed,
            median_combined_nrmse=float(median(combined)),
            median_current_nrmse=float(median(current)),
            median_speed_nrmse=float(median(speed)),
        )
        aggregates.append(aggregate)
        progress(
            f"aggregate latent={latent_size}: combined/current/speed medians="
            f"{aggregate.median_combined_nrmse:.6g}/"
            f"{aggregate.median_current_nrmse:.6g}/"
            f"{aggregate.median_speed_nrmse:.6g}"
        )
    selected_latent = select_latent_from_medians(
        {value.latent_size: value.median_combined_nrmse for value in aggregates},
        relative_tolerance=profile.tie_relative_tolerance,
    )
    selected_run = min(
        (value for value in evidences if value.run.architecture.state_size == selected_latent),
        key=lambda value: value.median_combined_nrmse,
    )
    progress(
        f"selected latent={selected_latent}, seed={selected_run.run.seed}, "
        f"multisine NRMSE={selected_run.median_combined_nrmse:.6g}"
    )
    return Phase6EStudy(
        profile_name=profile_name,
        dataset_fingerprint=dataset.fingerprint,
        protocol_sha256=protocol,
        phase6b_report_sha256=report_sha256,
        runs=tuple(evidences),
        aggregates=tuple(aggregates),
        selected_latent_size=selected_latent,
        selected_run=selected_run,
        target_combined_nrmse=profile.target_combined_nrmse,
    )


def save_phase6e_checkpoint(
    directory: Path | str,
    *,
    dataset: Phase4Dataset,
    study: Phase6EStudy,
    phase6e: Phase6ESpec | None = None,
    phase6: Phase6Spec | None = None,
    overwrite: bool = False,
) -> Phase6ECheckpointManifest:
    """Atomically save the selected Phase-6E run and all evidence."""

    try:
        from flax import serialization
    except ImportError as error:
        raise ImportError("Phase-6E checkpoint saving requires Phase-6 dependencies") from error
    phase6 = phase6 or load_phase6_spec()
    phase6e = phase6e or load_phase6e_spec(phase6=phase6)
    profile = phase6e.profile(study.profile_name)
    base = phase6.profile(profile.base_phase6_profile)
    selected = study.selected_run
    selected_run = selected.run
    aggregate = study.selected_aggregate
    output = Path(directory)
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Phase-6E checkpoint already exists: {output}; pass overwrite=True explicitly"
        )
    staging = output.parent / f".{output.name}.building"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        parameter_path = staging / "parameters.msgpack"
        parameter_path.write_bytes(serialization.to_bytes(selected_run.parameters))
        normalization_path = staging / "normalization.npz"
        dataset.normalization.save(normalization_path)
        history_path = staging / "study_history.json"
        history_path.write_text(
            json.dumps(
                study.history_payload(phase6e=phase6e),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        architecture = selected_run.architecture
        manifest = Phase6ECheckpointManifest(
            schema_version=1,
            phase="6E",
            model_type="ContractingR2DN",
            training_profile=study.profile_name,
            selected_variant=f"latent{study.selected_latent_size}",
            dataset_fingerprint=dataset.fingerprint,
            protocol_sha256=study.protocol_sha256,
            phase6b_report_sha256=study.phase6b_report_sha256,
            latent_size=architecture.state_size,
            feature_size=architecture.features,
            hidden_sizes=architecture.hidden,
            initialization=architecture.init_method,
            polar_parameterization=architecture.do_polar_param,
            seed=selected_run.seed,
            burn_in_steps=profile.burn_in_steps,
            selection_duration_s=profile.selection_duration_s,
            selection_metric=str(phase6e.selection["run_metric"]),
            latent_median_validation_nrmse=aggregate.median_combined_nrmse,
            validation_free_rollout_nrmse=selected.median_combined_nrmse,
            current_validation_nrmse=selected.median_current_nrmse,
            speed_validation_nrmse=selected.median_speed_nrmse,
            phase4_validation_nrmse=selected_run.selection_score,
            contractivity_margin=selected_run.contractivity_margin,
            update_count=sum(stage.updates for stage in base.final_stages),
            target_combined_nrmse=study.target_combined_nrmse,
            target_met=study.target_met,
            parameter_sha256=_file_sha256(parameter_path),
            normalization_sha256=_file_sha256(normalization_path),
            study_history_sha256=_file_sha256(history_path),
        )
        manifest.validate(dataset=dataset, phase6e=phase6e, phase6=phase6)
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


def load_phase6e_checkpoint(
    directory: Path | str,
    *,
    dataset: Phase4Dataset,
    phase6e: Phase6ESpec | None = None,
    phase6: Phase6Spec | None = None,
) -> LoadedPhase6ECheckpoint:
    """Load a Phase-6E checkpoint after validating provenance and hashes."""

    try:
        from flax import serialization
    except ImportError as error:
        raise ImportError("Phase-6E checkpoint loading requires Phase-6 dependencies") from error
    phase6 = phase6 or load_phase6_spec()
    phase6e = phase6e or load_phase6e_spec(phase6=phase6)
    root = Path(directory)
    required = (
        "manifest.json",
        "parameters.msgpack",
        "normalization.npz",
        "study_history.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Phase-6E checkpoint is missing: {', '.join(missing)}")
    manifest = Phase6ECheckpointManifest.from_dict(
        json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    )
    manifest.validate(dataset=dataset, phase6e=phase6e, phase6=phase6)
    for path, expected in (
        (root / "parameters.msgpack", manifest.parameter_sha256),
        (root / "normalization.npz", manifest.normalization_sha256),
        (root / "study_history.json", manifest.study_history_sha256),
    ):
        if _file_sha256(path) != expected:
            raise ValueError(f"Phase-6E checkpoint content hash mismatch: {path.name}")
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
        raise ValueError("Phase-6E history protocol hash mismatch")
    if history.get("dataset_fingerprint") != dataset.fingerprint:
        raise ValueError("Phase-6E history belongs to another dataset")
    if int(history.get("selected_latent_size", -1)) != manifest.latent_size:
        raise ValueError("Phase-6E history selected latent mismatch")
    if int(history.get("selected_seed", -1)) != manifest.seed:
        raise ValueError("Phase-6E history selected seed mismatch")
    return LoadedPhase6ECheckpoint(
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
        raise ValueError("Phase-6E normalization metadata differs from Phase 4")
    if any(
        not np.array_equal(getattr(checkpoint, name), getattr(dataset, name))
        for name in array_fields
    ):
        raise ValueError("Phase-6E normalization values differ from Phase 4")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
