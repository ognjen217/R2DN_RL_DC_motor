"""Typed Phase-1 R2DN interface specification and invariant checks."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from r2dn_dc_motor.spec import ExperimentSpec, SpecValidationError, load_phase0_spec


@dataclass(frozen=True)
class Phase1Spec:
    """Frozen design contract for the R2DN world-model boundary."""

    schema_version: int
    phase: dict[str, Any]
    interface: dict[str, Any]
    initialization: dict[str, Any]
    rollout: dict[str, Any]
    contractivity: dict[str, Any]
    data: dict[str, Any]
    checkpoint: dict[str, Any]
    upstream: dict[str, Any]

    @classmethod
    def from_dict(
        cls,
        raw: dict[str, Any],
        phase0: ExperimentSpec | None = None,
    ) -> Phase1Spec:
        spec = cls(
            schema_version=int(raw["schema_version"]),
            phase=dict(raw["phase"]),
            interface=dict(raw["interface"]),
            initialization=dict(raw["initialization"]),
            rollout=dict(raw["rollout"]),
            contractivity=dict(raw["contractivity"]),
            data=dict(raw["data"]),
            checkpoint=dict(raw["checkpoint"]),
            upstream=dict(raw["upstream"]),
        )
        spec.validate(phase0 or load_phase0_spec())
        return spec

    def validate(self, phase0: ExperimentSpec) -> None:
        """Validate Phase-1 decisions against the frozen Phase-0 experiment."""

        errors: list[str] = []
        interface = self.interface
        hidden = set(phase0.signals.hidden_state)

        if self.schema_version != 1:
            errors.append("schema_version must be 1")
        if self.phase.get("number") != 1:
            errors.append("phase.number must be 1")
        if self.phase.get("status") != "locked":
            errors.append("Phase 1 must be explicitly locked")

        observation_features = tuple(interface.get("observation_features", ()))
        control_features = tuple(interface.get("control_features", ()))
        regressor_features = tuple(interface.get("regressor_features", ()))
        target_features = tuple(interface.get("target_features", ()))

        if observation_features != phase0.signals.plant_output:
            errors.append("observation features must match the frozen plant output order")
        if control_features != (phase0.signals.control,):
            errors.append("control features must contain only the frozen control signal")
        if regressor_features != phase0.signals.world_model_input:
            errors.append("regressor features must match the frozen world-model input order")
        if target_features != phase0.signals.world_model_output:
            errors.append("target features must match the frozen world-model output order")

        if hidden & set(regressor_features):
            errors.append("hidden temperature leaked into the Phase-1 regressor")
        if hidden & set(target_features):
            errors.append("hidden temperature leaked into the Phase-1 targets")

        if interface.get("input_size") != len(regressor_features):
            errors.append("input_size must equal the number of regressor features")
        if interface.get("output_size") != len(target_features):
            errors.append("output_size must equal the number of target features")

        minimum_latent = int(interface.get("minimum_latent_size", 0))
        pilot_latent = int(interface.get("pilot_latent_size", 0))
        if minimum_latent <= len(observation_features):
            errors.append("minimum latent size must exceed the observation dimension")
        if pilot_latent < minimum_latent:
            errors.append("pilot latent size must not be smaller than the minimum")

        if (
            interface.get("temporal_representation")
            != phase0.temporal_representation["r2dn"]
        ):
            errors.append("Phase-1 temporal representation drifted from Phase 0")
        if interface.get("sequence_layout") != "time_batch_feature":
            errors.append("upstream R2DN sequence layout must be time-batch-feature")
        if (
            interface.get("step_semantics")
            != "regressor_at_k_predicts_observation_at_k_plus_1"
        ):
            errors.append("one-step timing semantics changed")

        if self.initialization.get("method") != "zero_carry_then_observation_burn_in":
            errors.append("R2DN initialization must use observation burn-in")
        if self.initialization.get("burn_in_steps_policy") != (
            "derive_after_phase2_time_constant_analysis"
        ):
            errors.append("burn-in length must remain deferred until time constants are known")
        if int(self.initialization.get("minimum_burn_in_steps", 0)) < 1:
            errors.append("burn-in must contain at least one transition")

        if self.rollout.get("mode") != "autoregressive":
            errors.append("free rollout must be autoregressive")
        if self.rollout.get("teacher_forcing_after_burn_in") is not False:
            errors.append("free rollout must not use teacher forcing")
        if self.rollout.get("future_observation_source") != "model_prediction_only":
            errors.append("future true observations leaked into free rollout")

        expected_scope = "latent_state_for_identical_external_regressor_sequences"
        if self.contractivity.get("formal_scope") != expected_scope:
            errors.append("formal contractivity scope is misstated")
        if self.contractivity.get("parameterization_test") != (
            "positive_definite_residual_of_equation_20"
        ):
            errors.append("the Equation-20 certificate test must remain required")
        if self.contractivity.get("autoregressive_scope") != (
            "requires_empirical_rollout_test_not_claimed_by_same_input_certificate"
        ):
            errors.append("autoregressive stability must not be presented as formally certified")

        if self.data.get("temperature_policy") != (
            "raw_and_evaluation_only_never_model_view"
        ):
            errors.append("temperature policy changed or became ambiguous")
        model_data_fields = (
            str(self.data.get("observation_array", "")),
            str(self.data.get("control_array", "")),
            str(self.data.get("regressor_array", "")),
            str(self.data.get("target_array", "")),
        )
        if any("temperature" in field.lower() for field in model_data_fields):
            errors.append("temperature leaked into the model data view")

        expected_files = {
            "manifest.json",
            "parameters.msgpack",
            "normalization.npz",
        }
        if set(self.checkpoint.get("required_files", ())) != expected_files:
            errors.append("checkpoint file contract changed")
        if self.checkpoint.get("manifest_schema_version") != 1:
            errors.append("checkpoint manifest schema version must be 1")

        upstream = self.upstream
        phase0_reference = phase0.r2dn_reference
        for phase1_key, phase0_key in (
            ("repository", "repository"),
            ("commit", "commit"),
            ("class_name", "class_name"),
            ("license", "license"),
        ):
            if upstream.get(phase1_key) != phase0_reference.get(phase0_key):
                errors.append(f"upstream {phase1_key} drifted from Phase 0")

        required_methods = {
            "initialize_carry",
            "simulate_sequence",
            "direct_to_explicit",
            "explicit_call",
        }
        if set(upstream.get("required_methods", ())) != required_methods:
            errors.append("upstream API audit method set changed")

        if errors:
            raise SpecValidationError("\n".join(f"- {error}" for error in errors))

    def summary(self) -> str:
        """Return a compact human-readable summary after validation."""

        return "\n".join(
            [
                "PHASE 1: PASS",
                f"backend: {self.interface['backend']}",
                f"dynamics: {self.interface['temporal_representation']}",
                f"step: {self.interface['step_semantics']}",
                f"regressor: {', '.join(self.interface['regressor_features'])}",
                f"target: {', '.join(self.interface['target_features'])}",
                f"pilot latent size: {self.interface['pilot_latent_size']}",
                f"burn-in: {self.initialization['method']}",
                f"free rollout: {self.rollout['mode']}",
                f"formal contractivity scope: {self.contractivity['formal_scope']}",
                f"R2DN commit: {self.upstream['commit']}",
            ]
        )


def load_phase1_spec(
    path: Path | str | None = None,
    phase0: ExperimentSpec | None = None,
) -> Phase1Spec:
    """Load and validate the canonical Phase-1 TOML file."""

    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "phase1.toml"
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    return Phase1Spec.from_dict(raw, phase0=phase0)
