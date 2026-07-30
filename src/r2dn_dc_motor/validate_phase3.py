"""CLI for Phase-3 specification and Gate-1 observability validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from r2dn_dc_motor.phase2_spec import load_phase2_spec
from r2dn_dc_motor.phase3_spec import load_phase3_spec
from r2dn_dc_motor.spec import SpecValidationError
from r2dn_dc_motor.validation import (
    generate_phase3_artifacts,
    run_phase3_validation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        help="optional Phase-3 TOML path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="also write JSON and PNG validation artifacts",
    )
    parser.add_argument(
        "--spec-only",
        action="store_true",
        help="validate the contract without running the numerical pilot",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        phase2 = load_phase2_spec()
        spec = load_phase3_spec(args.config, phase2=phase2)
        if args.spec_only:
            print(spec.summary())
            return 0
        if args.output_dir is None:
            report = run_phase3_validation(spec, phase2=phase2)
            paths: tuple[Path, Path] | None = None
        else:
            report, report_path, figure_path = generate_phase3_artifacts(
                args.output_dir,
                spec=spec,
                phase2=phase2,
            )
            paths = (report_path, figure_path)
    except (OSError, KeyError, TypeError, ValueError, SpecValidationError) as error:
        print("PHASE 3: FAIL")
        print(error)
        return 1

    print(spec.summary())
    print(report.summary())
    if paths is not None:
        print(f"report: {paths[0]}")
        print(f"figure: {paths[1]}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
