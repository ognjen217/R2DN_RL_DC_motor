"""Command-line validation for the frozen Phase-1 R2DN interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from r2dn_dc_motor.phase1_spec import load_phase1_spec
from r2dn_dc_motor.spec import SpecValidationError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        help="Phase-1 TOML file; defaults to configs/phase1.toml",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        spec = load_phase1_spec(args.config)
    except (OSError, KeyError, TypeError, ValueError, SpecValidationError) as error:
        print("PHASE 1: FAIL")
        print(error)
        return 1

    print(spec.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
