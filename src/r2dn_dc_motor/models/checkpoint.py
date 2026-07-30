"""Versioned metadata contract for future Flax R2DN checkpoints."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from r2dn_dc_motor.phase1_spec import Phase1Spec, load_phase1_spec
from r2dn_dc_motor.spec import SpecValidationError


@dataclass(frozen=True)
class R2DNCheckpointManifest:
    """Metadata stored next to ``parameters.msgpack`` and ``normalization.npz``."""

    schema_version: int
    model_type: str
    seed: int
    upstream_commit: str
    observation_features: tuple[str, ...]
    control_features: tuple[str, ...]
    latent_size: int
    feature_size: int
    hidden_sizes: tuple[int, ...]
    control_period_s: float

    @classmethod
    def for_phase1_pilot(
        cls,
        *,
        seed: int,
        feature_size: int = 8,
        hidden_sizes: tuple[int, ...] = (8, 8),
        spec: Phase1Spec | None = None,
    ) -> R2DNCheckpointManifest:
        phase1 = spec or load_phase1_spec()
        return cls(
            schema_version=int(phase1.checkpoint["manifest_schema_version"]),
            model_type="ContractingR2DN",
            seed=seed,
            upstream_commit=str(phase1.upstream["commit"]),
            observation_features=tuple(phase1.interface["observation_features"]),
            control_features=tuple(phase1.interface["control_features"]),
            latent_size=int(phase1.interface["pilot_latent_size"]),
            feature_size=feature_size,
            hidden_sizes=hidden_sizes,
            control_period_s=0.001,
        )

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> R2DNCheckpointManifest:
        return cls(
            schema_version=int(raw["schema_version"]),
            model_type=str(raw["model_type"]),
            seed=int(raw["seed"]),
            upstream_commit=str(raw["upstream_commit"]),
            observation_features=tuple(raw["observation_features"]),
            control_features=tuple(raw["control_features"]),
            latent_size=int(raw["latent_size"]),
            feature_size=int(raw["feature_size"]),
            hidden_sizes=tuple(int(value) for value in raw["hidden_sizes"]),
            control_period_s=float(raw["control_period_s"]),
        )

    def validate(self, spec: Phase1Spec | None = None) -> None:
        phase1 = spec or load_phase1_spec()
        errors: list[str] = []

        if self.schema_version != phase1.checkpoint["manifest_schema_version"]:
            errors.append("checkpoint schema version mismatch")
        if self.model_type != phase1.upstream["class_name"]:
            errors.append("checkpoint model type is not ContractingR2DN")
        if self.upstream_commit != phase1.upstream["commit"]:
            errors.append("checkpoint was not produced with the pinned upstream commit")
        if self.observation_features != tuple(phase1.interface["observation_features"]):
            errors.append("checkpoint observation feature order mismatch")
        if self.control_features != tuple(phase1.interface["control_features"]):
            errors.append("checkpoint control feature order mismatch")
        if self.latent_size < int(phase1.interface["minimum_latent_size"]):
            errors.append("checkpoint latent size is below the Phase-1 minimum")
        if self.feature_size < 1 or not self.hidden_sizes or min(self.hidden_sizes) < 1:
            errors.append("checkpoint network widths must be positive")
        if self.control_period_s <= 0.0:
            errors.append("checkpoint control period must be positive")

        if errors:
            raise SpecValidationError("\n".join(f"- {error}" for error in errors))

    def to_json(self, *, indent: int = 2) -> str:
        self.validate()
        return json.dumps(asdict(self), indent=indent, sort_keys=True) + "\n"

    def save(self, path: Path | str) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> R2DNCheckpointManifest:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        manifest = cls.from_dict(raw)
        manifest.validate()
        return manifest
