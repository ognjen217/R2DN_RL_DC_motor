"""Command-line validation for the frozen Phase-0 experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from .spec import SpecValidationError, load_phase0_spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        help="Phase-0 TOML file; defaults to configs/phase0.toml",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        spec = load_phase0_spec(args.config)
    except (OSError, KeyError, TypeError, ValueError, SpecValidationError) as error:
        print("PHASE 0: FAIL")
        print(error)
        return 1

    print(spec.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

