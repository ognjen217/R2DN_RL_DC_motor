"""CLI for the Phase-6F optimizer-floor ablation."""

from __future__ import annotations

import argparse
from pathlib import Path

from r2dn_dc_motor.data import Phase4Dataset
from r2dn_dc_motor.models.jax_runtime import inspect_jax_runtime
from r2dn_dc_motor.models.r2dn_phase6f import (
    save_phase6f_checkpoint,
    train_phase6f_study,
)
from r2dn_dc_motor.phase2_spec import load_phase2_spec
from r2dn_dc_motor.phase6_spec import load_phase6_spec
from r2dn_dc_motor.phase6b_spec import load_phase6b_spec
from r2dn_dc_motor.phase6e_spec import load_phase6e_spec
from r2dn_dc_motor.phase6f_spec import load_phase6f_spec
from r2dn_dc_motor.validation.phase6f import (
    generate_phase6f_artifacts,
    validate_phase6f_study,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the Phase-6E latent-16 baseline with 3000- and 6000-update "
            "cosine-learning-rate runs while holding architecture, seed, data, "
            "loss, burn-in, and rollout curriculum fixed."
        )
    )
    parser.add_argument("--spec-only", action="store_true")
    parser.add_argument("--runtime-only", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--profile",
        choices=("ci", "final"),
        default="final",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/phase4-full-v1"),
    )
    parser.add_argument(
        "--phase6b-report",
        type=Path,
        default=Path("results/phase6b/phase6b_latent_and_stability.json"),
    )
    parser.add_argument(
        "--phase6e-checkpoint",
        type=Path,
        default=Path("checkpoints/phase6e/r2dn-v1"),
        help="existing latent-16/seed-43 baseline checkpoint",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("checkpoints/phase6f/run-cache-v1"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/phase6f/r2dn-v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase6f"),
    )
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--overwrite-checkpoint", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    phase2 = load_phase2_spec()
    phase6 = load_phase6_spec()
    phase6b = load_phase6b_spec(phase2=phase2, phase6=phase6)
    phase6e = load_phase6e_spec(phase2=phase2, phase6=phase6, phase6b=phase6b)
    phase6f = load_phase6f_spec(
        phase2=phase2,
        phase6=phase6,
        phase6b=phase6b,
        phase6e=phase6e,
    )
    print(phase6f.summary(args.profile))
    if args.spec_only:
        return 0
    try:
        runtime = inspect_jax_runtime(require_cuda=args.require_cuda)
    except (ImportError, RuntimeError) as error:
        print(f"PHASE 6F JAX RUNTIME: FAIL\n{error}")
        return 2
    print(runtime.summary().replace("PHASE 6", "PHASE 6F"))
    if args.runtime_only:
        return 0
    if not args.train:
        parser.error("choose --train, --spec-only, or --runtime-only")
    if args.checkpoint_dir.exists() and not args.overwrite_checkpoint:
        print(
            "PHASE 6F CHECKPOINT: FAIL\n"
            f"checkpoint already exists: {args.checkpoint_dir}; "
            "pass --overwrite-checkpoint only if replacement is intentional"
        )
        return 2
    dataset = Phase4Dataset(args.dataset)
    study = train_phase6f_study(
        dataset,
        args.phase6b_report,
        args.phase6e_checkpoint,
        phase6f=phase6f,
        phase6e=phase6e,
        phase6=phase6,
        phase2=phase2,
        profile_name=args.profile,
        cache_directory=args.cache_dir,
        overwrite_cache=args.overwrite_cache,
        progress=lambda message: print(message, flush=True),
    )
    manifest = save_phase6f_checkpoint(
        args.checkpoint_dir,
        dataset=dataset,
        study=study,
        phase6f=phase6f,
        phase6e=phase6e,
        phase6=phase6,
        overwrite=args.overwrite_checkpoint,
    )
    report = validate_phase6f_study(
        study,
        args.phase6b_report,
        args.phase6e_checkpoint,
        phase6f=phase6f,
    )
    report_path, figure_path = generate_phase6f_artifacts(report, args.output_dir)
    print(report.summary())
    print(
        f"checkpoint: {args.checkpoint_dir} "
        f"(variant={manifest.selected_variant}, updates={manifest.update_count})"
    )
    print(f"report: {report_path}")
    print(f"figure: {figure_path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
