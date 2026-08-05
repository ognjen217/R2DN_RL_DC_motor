"""CLI for the Phase-6E full-curriculum larger-latent experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from r2dn_dc_motor.data import Phase4Dataset
from r2dn_dc_motor.models.jax_runtime import inspect_jax_runtime
from r2dn_dc_motor.models.r2dn_phase6e import (
    save_phase6e_checkpoint,
    train_phase6e_study,
)
from r2dn_dc_motor.phase2_spec import load_phase2_spec
from r2dn_dc_motor.phase6_spec import load_phase6_spec
from r2dn_dc_motor.phase6b_spec import load_phase6b_spec
from r2dn_dc_motor.phase6e_spec import load_phase6e_spec
from r2dn_dc_motor.validation.phase6e import (
    generate_phase6e_artifacts,
    validate_phase6e_study,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train latent 8/12/16 with the identical full Phase-6 curriculum, "
            "then select on held-out 100 s multisine FULL/RK4 references."
        )
    )
    parser.add_argument(
        "--spec-only",
        action="store_true",
        help="validate and print the locked Phase-6E protocol, then exit",
    )
    parser.add_argument(
        "--runtime-only",
        action="store_true",
        help="validate specification and JAX runtime, then exit",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="run or resume every declared latent/seed training and selection rollout",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="fail unless JAX executes on NVIDIA CUDA",
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
        "--phase6b-report",
        type=Path,
        default=Path("results/phase6b/phase6b_latent_and_stability.json"),
        help="passing Phase-6B report containing the fixed validation anchors",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("checkpoints/phase6e/run-cache-v1"),
        help="resumable per-latent/per-seed training cache",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/phase6e/r2dn-v1"),
        help="selected Phase-6E checkpoint directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase6e"),
        help="Phase-6E JSON and PNG artifact directory",
    )
    parser.add_argument(
        "--overwrite-cache",
        action="store_true",
        help="explicitly retrain and replace matching Phase-6E cache entries",
    )
    parser.add_argument(
        "--overwrite-checkpoint",
        action="store_true",
        help="explicitly replace an existing selected Phase-6E checkpoint",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    phase2 = load_phase2_spec()
    phase6 = load_phase6_spec()
    phase6b = load_phase6b_spec(phase2=phase2, phase6=phase6)
    phase6e = load_phase6e_spec(
        phase2=phase2,
        phase6=phase6,
        phase6b=phase6b,
    )
    print(phase6e.summary(args.profile))
    if args.spec_only:
        return 0
    try:
        runtime = inspect_jax_runtime(require_cuda=args.require_cuda)
    except (ImportError, RuntimeError) as error:
        print(f"PHASE 6E JAX RUNTIME: FAIL\n{error}")
        return 2
    print(runtime.summary().replace("PHASE 6", "PHASE 6E"))
    if args.runtime_only:
        return 0
    if not args.train:
        parser.error("choose --train, --spec-only, or --runtime-only")
    if args.checkpoint_dir.exists() and not args.overwrite_checkpoint:
        print(
            "PHASE 6E CHECKPOINT: FAIL\n"
            f"checkpoint already exists: {args.checkpoint_dir}; "
            "pass --overwrite-checkpoint only if replacement is intentional"
        )
        return 2
    dataset = Phase4Dataset(args.dataset)
    study = train_phase6e_study(
        dataset,
        args.phase6b_report,
        phase6e=phase6e,
        phase6=phase6,
        phase2=phase2,
        profile_name=args.profile,
        cache_directory=args.cache_dir,
        overwrite_cache=args.overwrite_cache,
        progress=lambda message: print(message, flush=True),
    )
    manifest = save_phase6e_checkpoint(
        args.checkpoint_dir,
        dataset=dataset,
        study=study,
        phase6e=phase6e,
        phase6=phase6,
        overwrite=args.overwrite_checkpoint,
    )
    report = validate_phase6e_study(
        study,
        args.phase6b_report,
        phase6e=phase6e,
    )
    report_path, figure_path = generate_phase6e_artifacts(report, args.output_dir)
    print(report.summary())
    print(
        f"checkpoint: {args.checkpoint_dir} "
        f"(latent={manifest.latent_size}, seed={manifest.seed})"
    )
    print(f"report: {report_path}")
    print(f"figure: {figure_path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
