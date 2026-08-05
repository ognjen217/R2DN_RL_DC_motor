"""CLI for the matched 1000-second R2DN versus FULL/RK4 benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from r2dn_dc_motor.data import Phase4Dataset
from r2dn_dc_motor.models.jax_runtime import inspect_jax_runtime
from r2dn_dc_motor.models.r2dn_phase6b import load_phase6b_checkpoint
from r2dn_dc_motor.models.r2dn_phase6d import load_phase6d_checkpoint
from r2dn_dc_motor.models.r2dn_phase6e import load_phase6e_checkpoint
from r2dn_dc_motor.models.r2dn_phase6f import load_phase6f_checkpoint
from r2dn_dc_motor.models.r2dn_phase7 import load_phase7_checkpoint
from r2dn_dc_motor.models.r2dn_training import load_phase6_checkpoint
from r2dn_dc_motor.phase2_spec import load_phase2_spec
from r2dn_dc_motor.phase6_spec import load_phase6_spec
from r2dn_dc_motor.phase6b_spec import STRESS_SCENARIOS, load_phase6b_spec
from r2dn_dc_motor.phase6d_spec import load_phase6d_spec
from r2dn_dc_motor.phase6e_spec import load_phase6e_spec
from r2dn_dc_motor.phase6f_spec import load_phase6f_spec
from r2dn_dc_motor.phase7_spec import load_phase7_spec
from r2dn_dc_motor.validation.r2dn_rk4_benchmark import (
    build_benchmark_report,
    build_voltage_trace,
    generate_benchmark_artifacts,
    load_benchmark_anchor,
    resolve_scenario,
    run_r2dn_trace,
    run_rk4_trace,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a selected Phase-6, Phase-6B, Phase-6D, Phase-6E, Phase-6F, "
            "or Phase-7 "
            "R2DN checkpoint "
            "against the canonical FULL electrothermal RK4 solver on the same "
            "long voltage trace."
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
        help="generated Phase-4 dataset directory",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/phase6b/r2dn-v2"),
        help=(
            "selected and validated Phase-6 through Phase-7 "
            "checkpoint directory"
        ),
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
        help="locked Phase-6B voltage scenario used by both models",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=1000.0,
        help="matched simulated horizon in seconds",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "id_test", "ood_test"),
        default="validation",
        help="Phase-6B burn-in anchor split",
    )
    parser.add_argument(
        "--anchor-index",
        type=int,
        default=0,
        help="zero-based anchor index within the requested Phase-6B split",
    )
    parser.add_argument(
        "--chunk-steps",
        type=int,
        default=10_000,
        help="JAX scan chunk; must divide the total number of control steps",
    )
    parser.add_argument(
        "--maximum-plot-points",
        type=int,
        default=20_000,
        help="maximum decimated samples drawn per trajectory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase6c"),
        help="benchmark JSON and PNG output directory",
    )
    return parser


def load_benchmark_checkpoint(
    checkpoint_directory: Path,
    *,
    dataset: Phase4Dataset,
    phase6: Any,
    phase6b: Any,
    phase6d: Any,
) -> tuple[Any, bool]:
    """Load a validated Phase-6 through Phase-7 checkpoint."""

    manifest_payload = json.loads(
        (checkpoint_directory / "manifest.json").read_text(encoding="utf-8")
    )
    checkpoint_phase = str(manifest_payload.get("phase"))
    if checkpoint_phase == "7":
        phase7 = load_phase7_spec()
        return (
            load_phase7_checkpoint(
                checkpoint_directory,
                dataset=dataset,
                phase7=phase7,
                phase6=phase6,
            ),
            False,
        )
    if checkpoint_phase == "6F":
        phase6e = load_phase6e_spec(phase6=phase6, phase6b=phase6b)
        phase6f = load_phase6f_spec(
            phase6=phase6,
            phase6b=phase6b,
            phase6e=phase6e,
        )
        return (
            load_phase6f_checkpoint(
                checkpoint_directory,
                dataset=dataset,
                phase6f=phase6f,
                phase6e=phase6e,
                phase6=phase6,
            ),
            False,
        )
    if checkpoint_phase == "6E":
        phase6e = load_phase6e_spec(phase6=phase6, phase6b=phase6b)
        return (
            load_phase6e_checkpoint(
                checkpoint_directory,
                dataset=dataset,
                phase6e=phase6e,
                phase6=phase6,
            ),
            False,
        )
    if checkpoint_phase == "6D":
        return (
            load_phase6d_checkpoint(
                checkpoint_directory,
                dataset=dataset,
                phase6d=phase6d,
                phase6=phase6,
            ),
            False,
        )
    if checkpoint_phase == "6B":
        return (
            load_phase6b_checkpoint(
                checkpoint_directory,
                dataset=dataset,
                phase6b=phase6b,
                phase6=phase6,
            ),
            True,
        )
    if checkpoint_phase == "6":
        return (
            load_phase6_checkpoint(
                checkpoint_directory,
                dataset=dataset,
                spec=phase6,
            ),
            False,
        )
    raise ValueError(
        "unsupported R2DN checkpoint phase: "
        f"{manifest_payload.get('phase')!r}; expected 6, 6B, 6D, 6E, or 6F "
        "(Phase 7 is also supported)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    phase2 = load_phase2_spec()
    phase6 = load_phase6_spec()
    phase6b = load_phase6b_spec(phase2=phase2, phase6=phase6)
    phase6d = load_phase6d_spec(phase6=phase6)
    control_period = phase2.integration_settings.control_period_s
    voltages = build_voltage_trace(
        resolve_scenario(phase6b, args.scenario),
        duration_s=args.duration_s,
        control_period_s=control_period,
    )
    if args.chunk_steps < 1 or voltages.size % args.chunk_steps:
        parser.error("--chunk-steps must be positive and divide the complete horizon")

    print(
        "R2DN VS FULL/RK4 BENCHMARK\n"
        f"horizon: {args.duration_s:g} s ({voltages.size} control steps)\n"
        f"scenario: {args.scenario}\n"
        f"RK4: {phase2.integration_settings.substeps_per_control} substeps/control step",
        flush=True,
    )
    try:
        runtime = inspect_jax_runtime(require_cuda=args.require_cuda)
    except (ImportError, RuntimeError) as error:
        print(f"R2DN RUNTIME: FAIL\n{error}")
        return 2
    print(runtime.summary().replace("PHASE 6", "R2DN BENCHMARK"), flush=True)

    dataset = Phase4Dataset(args.dataset)
    checkpoint, require_phase6b_model_match = load_benchmark_checkpoint(
        args.checkpoint_dir,
        dataset=dataset,
        phase6=phase6,
        phase6b=phase6b,
        phase6d=phase6d,
    )
    anchor, phase6b_report = load_benchmark_anchor(
        dataset,
        args.phase6b_report,
        split=args.split,
        anchor_index=args.anchor_index,
        checkpoint=checkpoint,
        require_checkpoint_match=require_phase6b_model_match,
    )
    scenario = resolve_scenario(phase6b, args.scenario)
    print(
        f"anchor: {anchor.provenance.trajectory_id}, "
        f"start={anchor.provenance.start_step}, "
        f"burn-in={anchor.provenance.burn_in_steps}",
        flush=True,
    )
    print("running R2DN cold and warm traces...", flush=True)
    r2dn = run_r2dn_trace(
        checkpoint,
        anchor,
        voltages,
        duration_s=args.duration_s,
        chunk_steps=args.chunk_steps,
    )
    print(
        f"R2DN complete: cold={r2dn.timing.cold_wall_time_s:.6g} s, "
        f"warm={r2dn.timing.warm_wall_time_s:.6g} s",
        flush=True,
    )
    print("running canonical FULL/RK4 reference...", flush=True)
    rk4 = run_rk4_trace(
        phase2,
        anchor.initial_full_state,
        voltages,
        duration_s=args.duration_s,
    )
    print(
        f"FULL/RK4 complete: steps={rk4.timing.control_steps_completed}/"
        f"{voltages.size}, wall={rk4.timing.wall_time_s:.6g} s",
        flush=True,
    )
    report = build_benchmark_report(
        dataset=dataset,
        checkpoint=checkpoint,
        phase6b_report=phase6b_report,
        phase2=phase2,
        scenario=scenario,
        duration_s=args.duration_s,
        chunk_steps=args.chunk_steps,
        anchor=anchor,
        r2dn=r2dn,
        rk4=rk4,
        runtime=runtime,
    )
    report_path, figure_path = generate_benchmark_artifacts(
        report,
        r2dn_observations=r2dn.observations,
        rk4_observations=rk4.observations,
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
