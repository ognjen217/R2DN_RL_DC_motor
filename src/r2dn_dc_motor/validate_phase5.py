"""CLI for fitting and evaluating the Phase-5 physical baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

from r2dn_dc_motor.data import Phase4Dataset
from r2dn_dc_motor.models import (
    IsothermalCalibrationCheckpoint,
    fit_global_isothermal_parameters,
)
from r2dn_dc_motor.phase2_spec import load_phase2_spec
from r2dn_dc_motor.phase4_spec import load_phase4_spec
from r2dn_dc_motor.phase5_spec import load_phase5_spec
from r2dn_dc_motor.validation.phase5 import (
    generate_phase5_artifacts,
    run_phase5_validation,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit and evaluate ISO-NOM/ISO-CAL physical baselines."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/phase4-full-v1"),
        help="generated Phase-4 dataset directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/phase5/iso_cal.json"),
        help="global ISO-CAL checkpoint path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase5"),
        help="JSON/PNG evaluation artifact directory",
    )
    parser.add_argument(
        "--fit",
        action="store_true",
        help="fit one global ISO-CAL checkpoint on the train split before evaluation",
    )
    parser.add_argument(
        "--overwrite-checkpoint",
        action="store_true",
        help="allow --fit to replace an existing checkpoint",
    )
    parser.add_argument(
        "--spec-only",
        action="store_true",
        help="validate the frozen Phase-5 contract without reading a dataset",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    phase2 = load_phase2_spec()
    phase4 = load_phase4_spec(phase2=phase2)
    phase5 = load_phase5_spec(phase2=phase2, phase4=phase4)
    print(phase5.summary())
    if args.spec_only:
        return

    dataset = Phase4Dataset(args.dataset)
    if args.fit:
        if args.checkpoint.exists() and not args.overwrite_checkpoint:
            raise FileExistsError(
                f"{args.checkpoint} already exists; pass --overwrite-checkpoint "
                "to replace it intentionally"
            )
        checkpoint = fit_global_isothermal_parameters(
            dataset,
            phase2,
            resistance_bounds=phase5.resistance_bounds,
            friction_bounds=phase5.friction_bounds,
            minimum_regressor_energy=float(
                phase5.calibration["minimum_regressor_energy"]
            ),
            forbidden_fit_features=tuple(
                phase5.interface["forbidden_fit_features"]
            ),
            method=str(phase5.calibration["method"]),
            selection_policy=str(phase5.calibration["selection_policy"]),
        )
        checkpoint.save(args.checkpoint)
        print(f"saved ISO-CAL checkpoint: {args.checkpoint}")
    else:
        checkpoint = IsothermalCalibrationCheckpoint.load(
            args.checkpoint,
            dataset=dataset,
        )

    report = run_phase5_validation(
        dataset,
        checkpoint,
        spec=phase5,
        phase2=phase2,
    )
    report_path, figure_path = generate_phase5_artifacts(report, args.output_dir)
    status = "PASS" if report.passed else "FAIL"
    print(f"PHASE 5 BASELINES: {status}")
    print(
        "ISO-CAL parameters: "
        f"R_eff={report.calibrated_parameters['effective_resistance_ohm']:.9g} ohm, "
        "b="
        f"{report.calibrated_parameters['viscous_friction_n_m_s_per_rad']:.9g} "
        "N m s/rad"
    )
    for model_name, metrics in report.metrics.items():
        print(
            f"{model_name}: validation one-step={metrics.one_step_nrmse['validation']:.6g}, "
            f"long={metrics.rollout_nrmse['validation']['long']:.6g}, "
            f"hot-long={metrics.long_rollout_regime_nrmse['hot']:.6g}, "
            f"runtime={metrics.runtime_transitions_per_s:.0f} transitions/s"
        )
    print(f"report: {report_path}")
    print(f"figure: {figure_path}")
    if not report.passed:
        failed = ", ".join(check.name for check in report.checks if not check.passed)
        raise SystemExit(f"Phase-5 validation failed: {failed}")


if __name__ == "__main__":
    main()
