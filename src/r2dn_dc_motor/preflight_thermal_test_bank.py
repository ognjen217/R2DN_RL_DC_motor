"""Verify the frozen Phase-7 test bank with FULL/RK4 only."""

from __future__ import annotations

import argparse
from pathlib import Path

from r2dn_dc_motor.data import Phase4Dataset
from r2dn_dc_motor.phase2_spec import load_phase2_spec
from r2dn_dc_motor.phase7_spec import load_phase7_spec
from r2dn_dc_motor.validation.thermal_test_bank import (
    generate_thermal_test_bank_preflight_artifact,
    run_thermal_test_bank_preflight,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-dataset",
        type=Path,
        default=Path("data/phase4-full-v1"),
        help="frozen dataset whose held-out whole trajectories define the bank",
    )
    parser.add_argument("--profile", choices=("ci", "final"), default="final")
    parser.add_argument(
        "--duration-s",
        type=float,
        choices=(1.0, 10.0, 100.0, 1000.0),
        default=1000.0,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase7/thermal_test_bank_preflight"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.profile == "final" and args.duration_s != 1000.0:
        parser.error("the official final preflight is locked to the complete 1000 s horizon")
    phase2 = load_phase2_spec()
    phase7 = load_phase7_spec()
    dataset = Phase4Dataset(args.evaluation_dataset)
    report = run_thermal_test_bank_preflight(
        evaluation_dataset=dataset,
        phase2=phase2,
        phase7=phase7,
        profile_name=args.profile,
        duration_s=args.duration_s,
        progress=lambda message: print(message, flush=True),
    )
    report_path = generate_thermal_test_bank_preflight_artifact(
        report,
        args.output_dir,
    )
    print(report.summary())
    print(f"report: {report_path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
