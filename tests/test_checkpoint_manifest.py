from dataclasses import replace

import pytest

from r2dn_dc_motor.models.checkpoint import R2DNCheckpointManifest
from r2dn_dc_motor.spec import SpecValidationError


def test_phase1_pilot_manifest_is_valid_and_round_trips(tmp_path):
    manifest = R2DNCheckpointManifest.for_phase1_pilot(seed=17)
    path = tmp_path / "manifest.json"

    manifest.save(path)
    restored = R2DNCheckpointManifest.load(path)

    assert restored == manifest
    assert restored.latent_size == 4
    assert restored.observation_features == (
        "armature_current_a",
        "angular_speed_rad_s",
    )


def test_unpinned_upstream_checkpoint_is_rejected():
    manifest = R2DNCheckpointManifest.for_phase1_pilot(seed=17)
    changed = replace(manifest, upstream_commit="0" * 40)

    with pytest.raises(SpecValidationError, match="pinned upstream commit"):
        changed.validate()


def test_checkpoint_with_too_small_latent_state_is_rejected():
    manifest = R2DNCheckpointManifest.for_phase1_pilot(seed=17)
    changed = replace(manifest, latent_size=2)

    with pytest.raises(SpecValidationError, match="below the Phase-1 minimum"):
        changed.validate()
