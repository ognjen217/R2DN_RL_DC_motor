from __future__ import annotations

from r2dn_dc_motor.models.r2dn_phase6f import select_optimizer_variant
from r2dn_dc_motor.validate_phase6f import build_parser


def test_optimizer_selection_keeps_baseline_inside_two_percent() -> None:
    selected = select_optimizer_variant(
        {
            "baseline_phase6e": 0.151,
            "cosine_1x": 0.150,
            "cosine_2x": 0.149,
        },
        ("baseline_phase6e", "cosine_1x", "cosine_2x"),
        relative_tolerance=0.02,
    )

    assert selected == "baseline_phase6e"


def test_optimizer_selection_accepts_material_schedule_improvement() -> None:
    selected = select_optimizer_variant(
        {
            "baseline_phase6e": 0.190,
            "cosine_1x": 0.160,
            "cosine_2x": 0.145,
        },
        ("baseline_phase6e", "cosine_1x", "cosine_2x"),
        relative_tolerance=0.02,
    )

    assert selected == "cosine_2x"


def test_phase6f_cli_defaults_to_existing_phase6e_baseline() -> None:
    args = build_parser().parse_args(["--train"])

    assert args.train is True
    assert args.profile == "final"
    assert str(args.phase6e_checkpoint) == "checkpoints/phase6e/r2dn-v1"
    assert str(args.cache_dir) == "checkpoints/phase6f/run-cache-v1"
    assert str(args.checkpoint_dir) == "checkpoints/phase6f/r2dn-v1"
