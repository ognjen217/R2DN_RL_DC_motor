from __future__ import annotations

from r2dn_dc_motor.models.r2dn_phase6b import select_latent_from_medians
from r2dn_dc_motor.validate_phase6e import build_parser


def test_larger_latent_selection_prefers_smaller_model_within_three_percent() -> None:
    selected = select_latent_from_medians(
        {8: 0.204, 12: 0.200, 16: 0.199},
        relative_tolerance=0.03,
    )

    assert selected == 8


def test_larger_latent_selection_accepts_material_improvement() -> None:
    selected = select_latent_from_medians(
        {8: 0.31, 12: 0.23, 16: 0.19},
        relative_tolerance=0.03,
    )

    assert selected == 16


def test_phase6e_cli_defaults_to_resumable_final_study() -> None:
    args = build_parser().parse_args(["--train"])

    assert args.train is True
    assert args.profile == "final"
    assert str(args.cache_dir) == "checkpoints/phase6e/run-cache-v1"
    assert str(args.checkpoint_dir) == "checkpoints/phase6e/r2dn-v1"
