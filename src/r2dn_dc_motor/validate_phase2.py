"""CLI for Phase-2 configuration and Gate-0 validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from r2dn_dc_motor.phase2_spec import load_phase2_spec
from r2dn_dc_motor.spec import SpecValidationError
from r2dn_dc_motor.validation import generate_phase2_artifacts, run_phase2_validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        help="optional Phase-2 TOML path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="also write JSON and PNG validation artifacts",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        spec = load_phase2_spec(args.config)
        report = run_phase2_validation(spec)
    except (OSError, KeyError, TypeError, ValueError, SpecValidationError) as error:
        print("PHASE 2: FAIL")
        print(error)
        return 1

    print(spec.summary())
    print(report.summary())
    if args.output_dir is not None:
        report_path, figure_path = generate_phase2_artifacts(
            args.output_dir,
            spec=spec,
        )
        print(f"report: {report_path}")
        print(f"figure: {figure_path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
