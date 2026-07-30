"""CLI for Phase-6 R2DN curriculum training and checkpoint validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from r2dn_dc_motor.data import Phase4Dataset
from r2dn_dc_motor.models import (
    inspect_jax_runtime,
    load_phase6_checkpoint,
    save_phase6_checkpoint,
    train_phase6_study,
)
from r2dn_dc_motor.phase6_spec import load_phase6_spec
from r2dn_dc_motor.validation.phase6 import (
    generate_phase6_artifacts,
    run_phase6_validation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec-only",
        action="store_true",
        help="validate only the locked Phase-6 contract; no JAX backend is needed",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="run the locked pilot search and repeated final training locally",
    )
    parser.add_argument(
        "--runtime-only",
        action="store_true",
        help="initialize JAX, report the actual accelerator, and exit",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="fail before reading data unless JAX executes on NVIDIA CUDA",
    )
    parser.add_argument(
        "--profile",
        choices=("ci", "final"),
        default="final",
        help="training budget; use final for thesis checkpoints",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/phase4-full-v1"),
        help="generated Phase-4 dataset directory",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/phase6/r2dn-v1"),
        help="versioned Phase-6 checkpoint directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase6"),
        help="JSON and PNG artifact directory",
    )
    parser.add_argument(
        "--overwrite-checkpoint",
        action="store_true",
        help="explicitly replace an existing Phase-6 checkpoint",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = load_phase6_spec()
    print(spec.summary(args.profile))
    if args.spec_only:
        return 0

    try:
        runtime = inspect_jax_runtime(require_cuda=args.require_cuda)
    except RuntimeError as error:
        print(f"PHASE 6 JAX RUNTIME: FAIL\n{error}")
        return 2
    print(runtime.summary())
    if args.runtime_only:
        return 0

    if (
        args.train
        and args.checkpoint_dir.exists()
        and not args.overwrite_checkpoint
    ):
        print(
            "PHASE 6 CHECKPOINT: FAIL\n"
            f"checkpoint already exists: {args.checkpoint_dir}; "
            "pass --overwrite-checkpoint before training if replacement is intentional"
        )
        return 2

    dataset = Phase4Dataset(args.dataset)
    if args.train:
        study = train_phase6_study(
            dataset,
            spec=spec,
            profile_name=args.profile,
            progress=lambda message: print(message, flush=True),
        )
        save_phase6_checkpoint(
            args.checkpoint_dir,
            dataset=dataset,
            study=study,
            spec=spec,
            overwrite=args.overwrite_checkpoint,
        )
    checkpoint = load_phase6_checkpoint(
        args.checkpoint_dir,
        dataset=dataset,
        spec=spec,
    )
    report = run_phase6_validation(dataset, checkpoint, spec=spec)
    report_path, figure_path = generate_phase6_artifacts(
        report,
        checkpoint.training_history,
        args.output_dir,
    )
    print(report.summary())
    print(f"report: {report_path}")
    print(f"figure: {figure_path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
