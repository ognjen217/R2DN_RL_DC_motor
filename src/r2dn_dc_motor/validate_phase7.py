"""Train and validate the Phase-7 pure-R2DN accuracy study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from r2dn_dc_motor.data import Phase4Dataset
from r2dn_dc_motor.models.jax_runtime import inspect_jax_runtime
from r2dn_dc_motor.models.r2dn_phase7 import (
    save_phase7_checkpoint,
    train_phase7_study,
)
from r2dn_dc_motor.phase6_spec import load_phase6_spec
from r2dn_dc_motor.phase7_spec import load_phase7_spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="optional Phase-7 TOML path")
    parser.add_argument("--profile", choices=("ci", "final"), default="final")
    parser.add_argument("--spec-only", action="store_true")
    parser.add_argument("--runtime-only", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/phase4-broadband-v2"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("checkpoints/phase7/run-cache-v1"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/phase7/r2dn-v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase7"),
    )
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--overwrite-checkpoint", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    phase7 = load_phase7_spec(args.config)
    print(phase7.summary(args.profile))
    if args.spec_only:
        return 0
    try:
        runtime = inspect_jax_runtime(require_cuda=args.require_cuda)
    except (ImportError, RuntimeError) as error:
        print(f"PHASE 7 RUNTIME: FAIL\n{error}")
        return 2
    print(runtime.summary().replace("PHASE 6", "PHASE 7"))
    if args.runtime_only:
        return 0
    if not args.train:
        print("PHASE 7: no action; pass --train, --spec-only, or --runtime-only")
        return 1
    phase6 = load_phase6_spec()
    dataset = Phase4Dataset(args.dataset)
    study = train_phase7_study(
        dataset,
        phase7=phase7,
        phase6=phase6,
        profile_name=args.profile,
        cache_directory=args.cache_dir,
        overwrite_cache=args.overwrite_cache,
        progress=lambda message: print(message, flush=True),
    )
    manifest = save_phase7_checkpoint(
        args.checkpoint_dir,
        dataset=dataset,
        study=study,
        phase7=phase7,
        phase6=phase6,
        overwrite=args.overwrite_checkpoint,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "phase7_training_study.json"
    report_path.write_text(
        json.dumps(study.history_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "PHASE 7 TRAINING: PASS\n"
        f"selected: {manifest.selected_variant}, seed={manifest.seed}\n"
        f"validation NRMSE: {manifest.validation_free_rollout_nrmse:.6g}\n"
        f"contractivity margin: {manifest.contractivity_margin:.6g}\n"
        f"checkpoint: {args.checkpoint_dir}\n"
        f"report: {report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

