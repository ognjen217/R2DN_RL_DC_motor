"""CLI for Phase-4 dataset generation and integrity validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from r2dn_dc_motor.data import DatasetGenerationError, generate_phase4_dataset
from r2dn_dc_motor.phase2_spec import load_phase2_spec
from r2dn_dc_motor.phase4_spec import load_phase4_spec
from r2dn_dc_motor.spec import SpecValidationError
from r2dn_dc_motor.validation.phase4 import (
    generate_phase4_validation_artifacts,
    run_phase4_validation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="optional Phase-4 TOML path",
    )
    parser.add_argument(
        "--profile",
        choices=("ci", "final"),
        default="final",
        help="locked dataset profile (default: final)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="existing dataset directory to validate",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="generate the selected profile before validation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="dataset output directory when --generate is used",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        help="write JSON and PNG validation artifacts",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly replace an existing --output-dir",
    )
    parser.add_argument(
        "--spec-only",
        action="store_true",
        help="validate only the Phase-4 contract",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        phase2 = load_phase2_spec()
        spec = load_phase4_spec(args.config, phase2=phase2)
        if args.spec_only:
            print(spec.summary(args.profile))
            return 0

        dataset_path = args.dataset
        if args.generate:
            if args.output_dir is None:
                raise ValueError("--generate requires --output-dir")
            if args.dataset is not None and args.dataset != args.output_dir:
                raise ValueError("--dataset and --output-dir must match when both are given")
            generated = generate_phase4_dataset(
                args.output_dir,
                profile_name=args.profile,
                spec=spec,
                phase2=phase2,
                overwrite=args.overwrite,
            )
            dataset_path = generated.root
            print(
                f"generated: {generated.trajectory_count} trajectories, "
                f"{generated.transition_count} transitions"
            )
        if dataset_path is None:
            raise ValueError(
                "provide --dataset, or use --generate with --output-dir; "
                "--spec-only checks only the contract"
            )

        if args.artifacts_dir is None:
            report = run_phase4_validation(dataset_path, spec=spec, phase2=phase2)
            paths: tuple[Path, Path] | None = None
        else:
            report, report_path, figure_path = generate_phase4_validation_artifacts(
                dataset_path,
                args.artifacts_dir,
                spec=spec,
                phase2=phase2,
            )
            paths = (report_path, figure_path)
    except (
        DatasetGenerationError,
        FileExistsError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        SpecValidationError,
    ) as error:
        print("PHASE 4: FAIL")
        print(error)
        return 1

    print(spec.summary(args.profile))
    print(report.summary())
    if paths is not None:
        print(f"report: {paths[0]}")
        print(f"figure: {paths[1]}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
