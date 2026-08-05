"""Evaluate current/improved R2DN and ISO baselines on a thermal test bank."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from r2dn_dc_motor.compare_r2dn_rk4 import load_benchmark_checkpoint
from r2dn_dc_motor.data import Phase4Dataset
from r2dn_dc_motor.models.isothermal_calibration import IsothermalCalibrationCheckpoint
from r2dn_dc_motor.models.jax_runtime import inspect_jax_runtime
from r2dn_dc_motor.phase2_spec import load_phase2_spec
from r2dn_dc_motor.phase6_spec import load_phase6_spec
from r2dn_dc_motor.phase6b_spec import load_phase6b_spec
from r2dn_dc_motor.phase6d_spec import load_phase6d_spec
from r2dn_dc_motor.phase7_spec import load_phase7_spec
from r2dn_dc_motor.validation.thermal_test_bank import (
    R2DNCandidate,
    generate_thermal_test_bank_artifacts,
    run_thermal_test_bank,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument(
        "--evaluation-dataset",
        type=Path,
        default=Path("data/phase4-broadband-v2"),
        help="frozen dataset whose ID/OOD whole trajectories define the test bank",
    )
    parser.add_argument(
        "--iso-cal-checkpoint",
        type=Path,
        default=Path("checkpoints/phase7/iso_cal.json"),
        help="ISO-CAL fitted once on the evaluation dataset train split",
    )
    parser.add_argument(
        "--r2dn-model",
        action="append",
        nargs=3,
        metavar=("LABEL", "CHECKPOINT_DIR", "TRAINING_DATASET"),
        required=True,
        help=(
            "repeat for each learned candidate; the training dataset is used only "
            "to verify checkpoint provenance"
        ),
    )
    parser.add_argument("--profile", choices=("ci", "final"), default="final")
    parser.add_argument(
        "--duration-s",
        type=float,
        default=1000.0,
        help="Evaluation horizon in seconds.",
    )
    parser.add_argument(
        "--allow-partial-final",
        action="store_true",
        help=(
            "Allow diagnostic runs of the final test bank on horizons "
            "shorter than 1000 s. Official final evaluation remains locked."
        ),
    )
    parser.add_argument(
        "--preflight-report",
        type=Path,
        default=None,
        help=(
            "passing FULL/RK4-only preflight report; required for the official "
            "final 1000 s comparison"
        ),
    )
    parser.add_argument("--chunk-steps", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase7/thermal_test_bank"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.duration_s <= 0.0:
        parser.error("--duration-s must be greater than zero")
    if (
        args.profile == "final"
        and args.duration_s != 1000.0
        and not args.allow_partial_final
    ):
        parser.error(
            "the final test bank is locked to the complete 1000 s horizon; "
            "use --allow-partial-final only for diagnostic runs"
        )
    official_final = args.profile == "final" and args.duration_s == 1000.0
    if official_final and args.preflight_report is None:
        parser.error(
            "the final 1000 s comparison requires --preflight-report from "
            "r2dn_dc_motor.preflight_thermal_test_bank"
        )
    if args.allow_partial_final:
        os.environ["R2DN_ALLOW_PARTIAL_HORIZON"] = "1"
    phase2 = load_phase2_spec()
    phase6 = load_phase6_spec()
    phase6b = load_phase6b_spec(phase2=phase2, phase6=phase6)
    phase6d = load_phase6d_spec(phase6=phase6)
    phase7 = load_phase7_spec()
    try:
        runtime = inspect_jax_runtime(require_cuda=args.require_cuda)
    except (ImportError, RuntimeError) as error:
        print(f"THERMAL TEST BANK RUNTIME: FAIL\n{error}")
        return 2
    evaluation_dataset = Phase4Dataset(args.evaluation_dataset)
    calibration = IsothermalCalibrationCheckpoint.load(
        args.iso_cal_checkpoint,
        dataset=evaluation_dataset,
    )
    candidates: list[R2DNCandidate] = []
    for label, checkpoint_path, training_dataset_path in args.r2dn_model:
        training_dataset = Phase4Dataset(Path(training_dataset_path))
        checkpoint, _ = load_benchmark_checkpoint(
            Path(checkpoint_path),
            dataset=training_dataset,
            phase6=phase6,
            phase6b=phase6b,
            phase6d=phase6d,
        )
        candidates.append(R2DNCandidate(name=label, checkpoint=checkpoint))
    preflight_report = None
    if args.preflight_report is not None:
        preflight_report = json.loads(
            args.preflight_report.read_text(encoding="utf-8")
        )
    control_steps = int(round(args.duration_s / phase2.integration_settings.control_period_s))
    chunk_steps = args.chunk_steps or min(10_000, control_steps)
    report = run_thermal_test_bank(
        evaluation_dataset=evaluation_dataset,
        candidates=tuple(candidates),
        calibration=calibration,
        phase2=phase2,
        phase7=phase7,
        runtime=runtime,
        profile_name=args.profile,
        duration_s=args.duration_s,
        chunk_steps=chunk_steps,
        preflight_report=preflight_report,
        progress=lambda message: print(message, flush=True),
    )
    report_path, figure_path = generate_thermal_test_bank_artifacts(
        report,
        args.output_dir,
    )
    print(report.summary())
    print(f"report: {report_path}")
    print(f"figure: {figure_path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
