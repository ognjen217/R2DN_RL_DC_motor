"""CLI for repeated latent search and long autoregressive stability tests."""

from __future__ import annotations

import argparse
from pathlib import Path

from r2dn_dc_motor.data import Phase4Dataset
from r2dn_dc_motor.models.jax_runtime import inspect_jax_runtime
from r2dn_dc_motor.models.r2dn_phase6b import (
    load_phase6b_checkpoint,
    save_phase6b_checkpoint,
    train_phase6b_study,
)
from r2dn_dc_motor.phase6_spec import load_phase6_spec
from r2dn_dc_motor.phase6b_spec import load_phase6b_spec
from r2dn_dc_motor.validation.phase6b import (
    generate_phase6b_artifacts,
    run_phase6b_validation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Repeat the R2DN latent search over multiple seeds, optionally reuse "
            "the selected Phase-6 checkpoint, and stress the outer autoregressive loop."
        )
    )
    parser.add_argument(
        "--spec-only",
        action="store_true",
        help="validate and print the locked Phase-6B protocol, then exit",
    )
    parser.add_argument(
        "--runtime-only",
        action="store_true",
        help="validate the specification and JAX runtime, then exit",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="fail before reading data unless JAX executes on NVIDIA CUDA",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="run or resume repeated latent search and save the selected checkpoint",
    )
    parser.add_argument(
        "--stress",
        action="store_true",
        help="run held-out replay and long stress tests on the selected checkpoint",
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
        "--phase6-checkpoint",
        type=Path,
        default=Path("checkpoints/phase6/r2dn-v1"),
        help="validated Phase-6 checkpoint eligible for reuse if its latent wins",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/phase6b/r2dn-v2"),
        help="selected Phase-6B checkpoint directory",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("checkpoints/phase6b/run-cache-v1"),
        help="resumable per-latent/per-seed run cache",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase6b"),
        help="Phase-6B JSON and PNG artifact directory",
    )
    parser.add_argument(
        "--overwrite-checkpoint",
        action="store_true",
        help="explicitly replace an existing selected Phase-6B checkpoint",
    )
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="explicitly retrain and replace matching per-run cache entries",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    phase6 = load_phase6_spec()
    phase6b = load_phase6b_spec(phase6=phase6)
    print(phase6b.summary(args.profile))
    if args.spec_only:
        return 0
    try:
        runtime = inspect_jax_runtime(require_cuda=args.require_cuda)
    except RuntimeError as error:
        print(f"PHASE 6B JAX RUNTIME: FAIL\n{error}")
        return 2
    print(runtime.summary().replace("PHASE 6", "PHASE 6B"))
    if args.runtime_only:
        return 0
    if not args.search and not args.stress:
        parser.error("choose --search, --stress, or both")
    if args.search and args.checkpoint_dir.exists() and not args.overwrite_checkpoint:
        print(
            "PHASE 6B CHECKPOINT: FAIL\n"
            f"checkpoint already exists: {args.checkpoint_dir}; "
            "pass --overwrite-checkpoint only if replacement is intentional"
        )
        return 2

    dataset = Phase4Dataset(args.dataset)
    if args.search:
        study = train_phase6b_study(
            dataset,
            phase6b=phase6b,
            phase6=phase6,
            profile_name=args.profile,
            cache_directory=args.cache_dir,
            reusable_phase6_checkpoint=args.phase6_checkpoint,
            overwrite_cache=args.overwrite_cache,
            progress=lambda message: print(message, flush=True),
        )
        manifest = save_phase6b_checkpoint(
            args.checkpoint_dir,
            dataset=dataset,
            study=study,
            phase6b=phase6b,
            phase6=phase6,
            overwrite=args.overwrite_checkpoint,
        )
        print(
            "PHASE 6B SEARCH: PASS\n"
            f"selected latent={manifest.latent_size}, seed={manifest.seed}, "
            f"source={manifest.selected_run_source}, "
            f"validation NRMSE={manifest.validation_free_rollout_nrmse:.6g}"
        )
    if not args.stress:
        return 0

    checkpoint = load_phase6b_checkpoint(
        args.checkpoint_dir,
        dataset=dataset,
        phase6b=phase6b,
        phase6=phase6,
    )
    if checkpoint.manifest.training_profile != args.profile:
        print(
            "PHASE 6B CHECKPOINT: FAIL\n"
            f"checkpoint profile={checkpoint.manifest.training_profile}, "
            f"requested profile={args.profile}"
        )
        return 2
    report = run_phase6b_validation(
        dataset,
        checkpoint,
        phase6b=phase6b,
        phase6=phase6,
        progress=lambda message: print(message, flush=True),
    )
    report_path, figure_path = generate_phase6b_artifacts(
        report,
        args.output_dir,
    )
    print(report.summary())
    print(f"report: {report_path}")
    print(f"figure: {figure_path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
