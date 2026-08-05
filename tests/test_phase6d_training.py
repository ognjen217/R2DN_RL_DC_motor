from __future__ import annotations

import numpy as np
import pytest

from r2dn_dc_motor.data import R2DNWindowBatch
from r2dn_dc_motor.models.r2dn_phase6d import (
    VariantAggregate,
    aligned_validation_window,
    select_variant_from_medians,
)
from r2dn_dc_motor.validate_phase6d import build_parser


def _aggregate(name: str, score: float) -> VariantAggregate:
    return VariantAggregate(
        name=name,
        latent_size=4,
        burn_in_steps=250,
        curriculum="baseline",
        seeds=(17, 29, 43),
        scores=(score, score, score),
        median_validation_free_rollout_nrmse=score,
    )


def test_aligned_windows_keep_identical_prediction_targets() -> None:
    maximum_burn = 10
    rollout = 6
    observations = np.arange((maximum_burn + rollout + 1) * 2, dtype=np.float32)
    observations = observations.reshape(maximum_burn + rollout + 1, 1, 2)
    controls = np.arange(maximum_burn + rollout, dtype=np.float32).reshape(-1, 1, 1)
    maximum = R2DNWindowBatch(
        observations=observations,
        controls=controls,
        trajectory_ids=("validation-example",),
        start_steps=(100,),
        split="validation",
    )
    short = aligned_validation_window(
        maximum,
        maximum_burn_in_steps=maximum_burn,
        burn_in_steps=4,
        rollout_steps=rollout,
    )

    np.testing.assert_array_equal(
        short.observations[5:],
        maximum.observations[maximum_burn + 1 :],
    )
    assert short.start_steps == (106,)
    assert short.transitions == 4 + rollout


def test_variant_selection_uses_declared_simplicity_order_for_near_ties() -> None:
    aggregates = (
        _aggregate("A_control", 0.204),
        _aggregate("B_latent8", 0.200),
        _aggregate("C_burnin1000", 0.199),
    )
    assert select_variant_from_medians(aggregates, relative_tolerance=0.03) == "A_control"


def test_variant_selection_chooses_material_improvement() -> None:
    aggregates = (
        _aggregate("A_control", 0.30),
        _aggregate("B_latent8", 0.19),
        _aggregate("C_burnin1000", 0.20),
    )
    assert select_variant_from_medians(aggregates, relative_tolerance=0.03) == "B_latent8"


def test_variant_selection_rejects_nonfinite_scores() -> None:
    with pytest.raises(ValueError, match="finite"):
        select_variant_from_medians((_aggregate("A_control", np.inf),), relative_tolerance=0.03)


def test_phase6d_cli_exposes_screen_before_training() -> None:
    parser = build_parser()
    args = parser.parse_args(["--screen", "--profile", "final"])
    assert args.screen is True
    assert args.train is False
    assert args.profile == "final"
