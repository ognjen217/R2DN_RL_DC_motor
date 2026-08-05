"""CLI for R2DN/ISO-NOM/ISO-CAL versus FULL/RK4."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from r2dn_dc_motor.compare_r2dn_rk4 import load_benchmark_checkpoint
from r2dn_dc_motor.data import Phase4Dataset
from r2dn_dc_motor.models.isothermal_calibration import (
    IsothermalCalibrationCheckpoint,
)
from r2dn_dc_motor.models.jax_runtime import inspect_jax_runtime
from r2dn_dc_motor.phase2_spec import load_phase2_spec
from r2dn_dc_motor.phase6_spec import load_phase6_spec
from r2dn_dc_motor.phase6b_spec import STRESS_SCENARIOS, load_phase6b_spec
from r2dn_dc_motor.phase6d_spec import load_phase6d_spec
from r2dn_dc_motor.validation.hidden_thermal_benchmark import (
    build_hidden_thermal_report,
    generate_hidden_thermal_artifacts,
    run_isothermal_trace,
)
from r2dn_dc_motor.validation.phase5 import build_isothermal_models
from r2dn_dc_motor.validation.r2dn_rk4_benchmark import (
    build_voltage_trace,
    load_benchmark_anchor,
    resolve_scenario,
    run_r2dn_trace,
    run_rk4_trace,
)

LOCKED_HORIZONS_S = (1.0, 10.0, 100.0, 1000.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the best temperature-blind R2DN with ISO-NOM and ISO-CAL "
            "against the FULL electrothermal RK4 ground truth."
        )
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="fail unless JAX executes R2DN on the locked NVIDIA CUDA backend",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/phase4-full-v1"),
        help="generated Phase-4 FULL dataset directory",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/phase6e/r2dn-v1"),
        help="selected best R2DN checkpoint (Phase-6E latent-16/seed-43 by default)",
    )
    parser.add_argument(
        "--iso-cal-checkpoint",
        type=Path,
        default=Path("checkpoints/phase5/iso_cal.json"),
        help="single locked train-only Phase-5 ISO-CAL checkpoint",
    )
    parser.add_argument(
        "--phase6b-report",
        type=Path,
        default=Path("results/phase6b/phase6b_latent_and_stability.json"),
        help="completed Phase-6B report containing the exact stress anchors",
    )
    parser.add_argument(
        "--scenario",
        choices=STRESS_SCENARIOS,
        default="multisine",
        help="one voltage trace shared by all four systems",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        choices=(1000.0,),
        default=1000.0,
        help="locked complete horizon; cumulative metrics are at 1/10/100/1000 s",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "id_test", "ood_test"),
        default="validation",
        help="burn-in anchor split",
    )
    parser.add_argument(
        "--anchor-index",
        type=int,
        default=0,
        help="zero-based anchor index in the requested split",
    )
    parser.add_argument(
        "--chunk-steps",
        type=int,
        default=10_000,
        help="R2DN JAX scan chunk; must divide one million steps",
    )
    parser.add_argument(
        "--maximum-plot-points",
        type=int,
        default=20_000,
        help="maximum decimated points drawn per trajectory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/hidden_thermal_comparison"),
        help="JSON and PNG output directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    phase2 = load_phase2_spec()
    phase6 = load_phase6_spec()
    phase6b = load_phase6b_spec(phase2=phase2, phase6=phase6)
    phase6d = load_phase6d_spec(phase6=phase6)
    control_period = phase2.integration_settings.control_period_s
    scenario = resolve_scenario(phase6b, args.scenario)
    voltages = build_voltage_trace(
        scenario,
        duration_s=args.duration_s,
        control_period_s=control_period,
    )
    if args.chunk_steps < 1 or voltages.size % args.chunk_steps:
        parser.error("--chunk-steps must be positive and divide the complete horizon")

    print(
        "R2DN / ISO-NOM / ISO-CAL VS FULL/RK4\n"
        f"horizons: {', '.join(f'{value:g}' for value in LOCKED_HORIZONS_S)} s\n"
        f"scenario: {args.scenario}\n"
        "candidate inputs: current, angular speed, applied voltage (no temperature)\n"
        "ground truth: FULL electrothermal plant with true temperature dependence",
        flush=True,
    )
    try:
        runtime = inspect_jax_runtime(require_cuda=args.require_cuda)
    except (ImportError, RuntimeError) as error:
        print(f"R2DN RUNTIME: FAIL\n{error}")
        return 2
    print(runtime.summary().replace("PHASE 6", "HIDDEN-THERMAL BENCHMARK"), flush=True)

    dataset = Phase4Dataset(args.dataset)
    checkpoint, require_phase6b_model_match = load_benchmark_checkpoint(
        args.checkpoint_dir,
        dataset=dataset,
        phase6=phase6,
        phase6b=phase6b,
        phase6d=phase6d,
    )
    calibration = IsothermalCalibrationCheckpoint.load(
        args.iso_cal_checkpoint,
        dataset=dataset,
    )
    anchor, _ = load_benchmark_anchor(
        dataset,
        args.phase6b_report,
        split=args.split,
        anchor_index=args.anchor_index,
        checkpoint=checkpoint,
        require_checkpoint_match=require_phase6b_model_match,
    )
    print(
        f"R2DN checkpoint: phase={checkpoint.manifest.phase}, "
        f"latent={checkpoint.manifest.latent_size}, seed={checkpoint.manifest.seed}\n"
        f"anchor: {anchor.provenance.trajectory_id}, "
        f"start={anchor.provenance.start_step}, "
        f"hidden T0={anchor.provenance.initial_temperature_c:.6g} °C",
        flush=True,
    )

    print("running temperature-blind R2DN...", flush=True)
    r2dn = run_r2dn_trace(
        checkpoint,
        anchor,
        voltages,
        duration_s=args.duration_s,
        chunk_steps=args.chunk_steps,
    )
    print("running FULL/RK4 ground truth with internal temperature state...", flush=True)
    full_rk4 = run_rk4_trace(
        phase2,
        anchor.initial_full_state,
        voltages,
        duration_s=args.duration_s,
    )

    models = build_isothermal_models(calibration, phase2)
    initial_observation = np.asarray(
        (
            anchor.initial_full_state.current_a,
            anchor.initial_full_state.speed_rad_s,
        ),
        dtype=np.float64,
    )
    isothermal = {}
    for name, model in models.items():
        print(f"running temperature-blind {name}...", flush=True)
        isothermal[name] = run_isothermal_trace(
            model,
            initial_observation,
            voltages,
            duration_s=args.duration_s,
        )

    report = build_hidden_thermal_report(
        dataset=dataset,
        checkpoint=checkpoint,
        calibration=calibration,
        phase2=phase2,
        scenario=scenario,
        duration_s=args.duration_s,
        horizons_s=LOCKED_HORIZONS_S,
        anchor=anchor,
        r2dn=r2dn,
        full_rk4=full_rk4,
        isothermal=isothermal,
        runtime=runtime,
    )
    report_path, figure_path = generate_hidden_thermal_artifacts(
        report,
        r2dn_observations=r2dn.observations,
        isothermal_observations={
            name: trace.observations for name, trace in isothermal.items()
        },
        full_rk4=full_rk4,
        physical_voltages_v=voltages,
        output_directory=args.output_dir,
        maximum_plot_points=args.maximum_plot_points,
    )
    print(report.summary())
    print(f"report: {report_path}")
    print(f"figure: {figure_path}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
