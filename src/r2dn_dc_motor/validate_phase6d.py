"""CLI for Phase-6D R2DN accuracy screening and controlled ablations."""

from __future__ import annotations

import argparse
from pathlib import Path

from r2dn_dc_motor.data import Phase4Dataset
from r2dn_dc_motor.models.jax_runtime import inspect_jax_runtime
from r2dn_dc_motor.models.r2dn_phase6b import load_phase6b_checkpoint
from r2dn_dc_motor.models.r2dn_phase6d import (
    save_phase6d_checkpoint,
    train_phase6d_study,
)
from r2dn_dc_motor.models.r2dn_training import load_phase6_checkpoint
from r2dn_dc_motor.phase6_spec import load_phase6_spec
from r2dn_dc_motor.phase6b_spec import load_phase6b_spec
from r2dn_dc_motor.phase6d_spec import load_phase6d_spec
from r2dn_dc_motor.validation.phase6d import (
    generate_phase6d_artifacts,
    generate_screen_artifacts,
    screen_existing_checkpoints,
    validate_phase6d_study,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Screen existing R2DN checkpoints, then optionally run the locked "
            "latent/burn-in/rollout accuracy ablation."
        )
    )
    parser.add_argument(
        "--spec-only",
        action="store_true",
        help="validate and print the Phase-6D protocol, then exit",
    )
    parser.add_argument(
        "--runtime-only",
        action="store_true",
        help="validate specification and JAX runtime, then exit",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="fail unless JAX executes on NVIDIA CUDA",
    )
    parser.add_argument(
        "--screen",
        action="store_true",
        help="compare the existing Phase-6B latent-4 and Phase-6 latent-8 checkpoints",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="run or resume the complete controlled A-D training ablation",
    )
    parser.add_argument(
        "--profile",
        choices=("ci", "final"),
        default="final",
        help="use final for thesis results and ci only for a smoke test",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/phase4-full-v1"),
        help="generated Phase-4 dataset directory",
    )
    parser.add_argument(
        "--phase6b-checkpoint",
        type=Path,
        default=Path("checkpoints/phase6b/r2dn-v2"),
        help="existing selected Phase-6B latent-4 checkpoint",
    )
    parser.add_argument(
        "--phase6-checkpoint",
        type=Path,
        default=Path("checkpoints/phase6/r2dn-v1"),
        help="existing Phase-6 latent-8 checkpoint",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("checkpoints/phase6d/run-cache-v1"),
        help="resumable per-variant/per-seed run cache",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/phase6d/r2dn-v3"),
        help="selected Phase-6D checkpoint directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase6d"),
        help="Phase-6D JSON and PNG artifact directory",
    )
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="explicitly retrain and replace matching Phase-6D cache entries",
    )
    parser.add_argument(
        "--overwrite-checkpoint",
        action="store_true",
        help="explicitly replace an existing selected Phase-6D checkpoint",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    phase6 = load_phase6_spec()
    phase6b = load_phase6b_spec(phase6=phase6)
    phase6d = load_phase6d_spec(phase6=phase6)
    print(phase6d.summary(args.profile))
    if args.spec_only:
        return 0
    try:
        runtime = inspect_jax_runtime(require_cuda=args.require_cuda)
    except (ImportError, RuntimeError) as error:
        print(f"PHASE 6D JAX RUNTIME: FAIL\n{error}")
        return 2
    print(runtime.summary().replace("PHASE 6", "PHASE 6D"))
    if args.runtime_only:
        return 0
    if not args.screen and not args.train:
        parser.error("choose --screen, --train, or both")

    dataset = Phase4Dataset(args.dataset)
    exit_code = 0
    if args.screen:
        phase6b_checkpoint = load_phase6b_checkpoint(
            args.phase6b_checkpoint,
            dataset=dataset,
            phase6b=phase6b,
            phase6=phase6,
        )
        phase6_checkpoint = load_phase6_checkpoint(
            args.phase6_checkpoint,
            dataset=dataset,
            spec=phase6,
        )
        screen = screen_existing_checkpoints(
            dataset,
            (
                ("phase6b_latent4", "6B", phase6b_checkpoint),
                ("phase6_latent8", "6", phase6_checkpoint),
            ),
            phase6d=phase6d,
        )
        report_path, figure_path = generate_screen_artifacts(screen, args.output_dir)
        print(screen.summary())
        print(f"screen report: {report_path}")
        print(f"screen figure: {figure_path}")
        if not screen.passed:
            exit_code = 1

    if args.train:
        if args.checkpoint_dir.exists() and not args.overwrite_checkpoint:
            print(
                "PHASE 6D CHECKPOINT: FAIL\n"
                f"checkpoint already exists: {args.checkpoint_dir}; "
                "pass --overwrite-checkpoint only if replacement is intentional"
            )
            return 2
        study = train_phase6d_study(
            dataset,
            phase6d=phase6d,
            phase6=phase6,
            profile_name=args.profile,
            cache_directory=args.cache_dir,
            overwrite_cache=args.overwrite_cache,
            progress=lambda message: print(message, flush=True),
        )
        manifest = save_phase6d_checkpoint(
            args.checkpoint_dir,
            dataset=dataset,
            study=study,
            phase6d=phase6d,
            phase6=phase6,
            overwrite=args.overwrite_checkpoint,
        )
        report = validate_phase6d_study(study, phase6d=phase6d)
        report_path, figure_path = generate_phase6d_artifacts(report, args.output_dir)
        print(report.summary())
        print(
            f"checkpoint: {args.checkpoint_dir} "
            f"({manifest.selected_variant}, latent={manifest.latent_size}, "
            f"seed={manifest.seed})"
        )
        print(f"report: {report_path}")
        print(f"figure: {figure_path}")
        if not report.passed:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
