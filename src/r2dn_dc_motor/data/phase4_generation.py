"""Deterministic Phase-4 trajectory planning and FULL simulation."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from r2dn_dc_motor.data.phase4_dataset import (
    NormalizationStatistics,
    RawPhase4Trajectory,
    canonical_sha256,
    trajectory_content_sha256,
)
from r2dn_dc_motor.phase2_spec import Phase2Spec, load_phase2_spec
from r2dn_dc_motor.phase4_spec import (
    EXCITATION_FAMILIES,
    OOD_AXES,
    SPLIT_NAMES,
    DatasetProfile,
    Phase4Spec,
    load_phase4_spec,
)
from r2dn_dc_motor.plants import MotorParameters

FloatArray = NDArray[np.floating]


class DatasetGenerationError(RuntimeError):
    """Raised when a planned FULL trajectory leaves the frozen safe domain."""


@dataclass(frozen=True)
class TrajectoryPlan:
    """All deterministic choices required to reproduce one trajectory."""

    trajectory_id: str
    split: str
    excitation_family: str
    seed: int
    duration_s: float
    initial_current_a: float
    initial_speed_rad_s: float
    initial_temperature_c: float
    load_torque_n_m: float
    ood_axis: str | None
    reference_resistance_scale: float
    inertia_scale: float
    thermal_capacitance_scale: float
    novel_profile: bool

    @property
    def parameter_scales(self) -> dict[str, float]:
        return {
            "reference_resistance": self.reference_resistance_scale,
            "inertia": self.inertia_scale,
            "thermal_capacitance": self.thermal_capacitance_scale,
        }


@dataclass(frozen=True)
class GeneratedDataset:
    """Paths and identity of a completed dataset build."""

    root: Path
    manifest_path: Path
    normalization_path: Path
    fingerprint: str
    trajectory_count: int
    transition_count: int


class _StreamingMoments:
    def __init__(self, features: int) -> None:
        self.count = 0
        self.mean = np.zeros(features, dtype=np.float64)
        self.m2 = np.zeros(features, dtype=np.float64)

    def update(self, values: FloatArray) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1, self.mean.size)
        if array.size == 0:
            return
        batch_count = array.shape[0]
        batch_mean = np.mean(array, axis=0)
        centered = array - batch_mean
        batch_m2 = np.sum(centered * centered, axis=0)
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        self.m2 += batch_m2 + delta * delta * self.count * batch_count / total
        self.count = total

    def standard_deviation(self, floor: float) -> FloatArray:
        variance = self.m2 / max(self.count, 1)
        return np.maximum(np.sqrt(variance), floor)


def build_trajectory_plans(
    spec: Phase4Spec,
    profile: DatasetProfile,
) -> tuple[TrajectoryPlan, ...]:
    """Create disjoint, deterministic whole-trajectory plans."""

    plans: list[TrajectoryPlan] = []
    seed = profile.seed_base
    for split in SPLIT_NAMES:
        count = profile.split_counts[split]
        for split_index in range(count):
            family_index = split_index % len(EXCITATION_FAMILIES)
            family = EXCITATION_FAMILIES[family_index]
            rng = np.random.default_rng(seed)
            temperature = _sample_training_temperature(
                spec,
                rng,
                band_index=(split_index // len(EXCITATION_FAMILIES) + family_index) % 3,
            )
            load = float(spec.domain["nominal_load_torque_n_m"])
            ood_axis: str | None = None
            resistance_scale = 1.0
            inertia_scale = 1.0
            thermal_scale = 1.0
            novel_profile = False

            if split == "ood_test":
                ood_axis = OOD_AXES[split_index % len(OOD_AXES)]
                if ood_axis == "higher_initial_temperature":
                    temperature = float(
                        rng.uniform(*_pair(spec.ood["higher_initial_temperature_c"]))
                    )
                elif ood_axis == "stronger_load":
                    load = float(
                        rng.uniform(*_pair(spec.ood["stronger_load_torque_n_m"]))
                    )
                elif ood_axis == "novel_profile":
                    novel_profile = True
                elif ood_axis == "physical_parameter_shift":
                    resistance_scale = _sample_away_from_one(
                        rng,
                        _pair(spec.ood["reference_resistance_scale"]),
                    )
                    inertia_scale = _sample_away_from_one(
                        rng,
                        _pair(spec.ood["inertia_scale"]),
                    )
                    thermal_scale = _sample_away_from_one(
                        rng,
                        _pair(spec.ood["thermal_capacitance_scale"]),
                    )

            plans.append(
                TrajectoryPlan(
                    trajectory_id=f"{split}-{family}-{split_index:04d}",
                    split=split,
                    excitation_family=family,
                    seed=seed,
                    duration_s=profile.duration_for(family),
                    initial_current_a=float(
                        rng.uniform(*_pair(spec.domain["initial_current_a"]))
                    ),
                    initial_speed_rad_s=float(
                        rng.uniform(*_pair(spec.domain["initial_speed_rad_s"]))
                    ),
                    initial_temperature_c=temperature,
                    load_torque_n_m=load,
                    ood_axis=ood_axis,
                    reference_resistance_scale=resistance_scale,
                    inertia_scale=inertia_scale,
                    thermal_capacitance_scale=thermal_scale,
                    novel_profile=novel_profile,
                )
            )
            seed += 1
    return tuple(plans)


def generate_phase4_dataset(
    output_directory: Path | str,
    *,
    profile_name: str = "ci",
    spec: Phase4Spec | None = None,
    phase2: Phase2Spec | None = None,
    overwrite: bool = False,
) -> GeneratedDataset:
    """Generate, normalize, hash, and atomically publish a Phase-4 dataset."""

    phase2 = phase2 or load_phase2_spec()
    spec = spec or load_phase4_spec(phase2=phase2)
    profile = spec.profile(profile_name)
    output = Path(output_directory)
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"output directory already exists: {output}; pass overwrite=True explicitly"
        )

    staging = output.parent / f".{output.name}.building"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    trajectories_root = staging / "trajectories"
    for split in SPLIT_NAMES:
        (trajectories_root / split).mkdir(parents=True)

    plans = build_trajectory_plans(spec, profile)
    observation_moments = _StreamingMoments(2)
    control_moments = _StreamingMoments(1)
    records: list[dict[str, Any]] = []
    transition_count = 0

    grouped: dict[tuple[str, str], list[TrajectoryPlan]] = {}
    for plan in plans:
        grouped.setdefault((plan.split, plan.excitation_family), []).append(plan)

    try:
        for key in (
            (split, family)
            for split in SPLIT_NAMES
            for family in EXCITATION_FAMILIES
        ):
            group_plans = grouped[key]
            trajectories = simulate_trajectory_group(group_plans, spec, phase2)
            for plan, trajectory in zip(group_plans, trajectories, strict=True):
                relative_path = (
                    Path("trajectories") / plan.split / f"{plan.trajectory_id}.npz"
                )
                path = staging / relative_path
                save = np.savez_compressed if bool(spec.dataset["compression"]) else np.savez
                save(path, **trajectory.arrays())
                record = _trajectory_record(
                    plan,
                    trajectory,
                    path=relative_path,
                    spec=spec,
                    phase2=phase2,
                )
                records.append(record)
                transition_count += trajectory.transitions
                if plan.split == "train":
                    observation_moments.update(trajectory.states[:, :2])
                    control_moments.update(trajectory.applied_voltages)

        floor = float(spec.normalization["standard_deviation_floor"])
        normalization = NormalizationStatistics(
            observation_mean=observation_moments.mean,
            observation_std=observation_moments.standard_deviation(floor),
            control_mean=control_moments.mean,
            control_std=control_moments.standard_deviation(floor),
            observation_count=observation_moments.count,
            control_count=control_moments.count,
        )
        normalization_path = staging / "normalization.npz"
        normalization.save(normalization_path)

        records.sort(key=lambda record: record["trajectory_id"])
        manifest = _build_manifest(
            spec=spec,
            phase2=phase2,
            profile=profile,
            records=records,
            transition_count=transition_count,
            normalization=normalization,
        )
        manifest["dataset_fingerprint"] = canonical_sha256(manifest)
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if transition_count != profile.minimum_total_transitions:
            raise DatasetGenerationError(
                "generated transition count does not match the locked profile"
            )
        if output.exists():
            shutil.rmtree(output)
        staging.rename(output)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    return GeneratedDataset(
        root=output,
        manifest_path=output / "manifest.json",
        normalization_path=output / "normalization.npz",
        fingerprint=str(manifest["dataset_fingerprint"]),
        trajectory_count=len(records),
        transition_count=transition_count,
    )


def simulate_trajectory_group(
    plans: list[TrajectoryPlan] | tuple[TrajectoryPlan, ...],
    spec: Phase4Spec,
    phase2: Phase2Spec,
) -> tuple[RawPhase4Trajectory, ...]:
    """Vectorized classical-RK4 FULL simulation for equal-family trajectories."""

    if not plans:
        return ()
    family = plans[0].excitation_family
    duration = plans[0].duration_s
    if any(
        plan.excitation_family != family or not math.isclose(plan.duration_s, duration)
        for plan in plans
    ):
        raise ValueError("a simulation group must share excitation family and duration")

    count = len(plans)
    steps = spec.steps(duration, phase2)
    dt = phase2.integration_settings.control_period_s
    integration_step = phase2.integration_settings.integrator_step_s
    substeps = phase2.integration_settings.substeps_per_control
    parameters = _parameter_arrays(plans, phase2.motor_parameters)
    states = np.asarray(
        [
            (
                plan.initial_current_a,
                plan.initial_speed_rad_s,
                plan.initial_temperature_c,
            )
            for plan in plans
        ],
        dtype=np.float64,
    )
    state_history = np.empty((steps + 1, count, 3), dtype=np.float64)
    commanded = np.empty((steps, count), dtype=np.float64)
    applied = np.empty((steps, count), dtype=np.float64)
    loads = np.asarray([plan.load_torque_n_m for plan in plans], dtype=np.float64)
    load_history = np.broadcast_to(loads, (steps, count)).copy()

    open_loop, references, identification = _exogenous_signals(plans, spec, steps, dt)
    integral_error = np.zeros(count, dtype=np.float64)
    state_history[0] = states
    safe_voltage = float(spec.domain["safe_voltage_limit_v"])
    current_guard = float(spec.domain["current_guard_a"])
    nominal = phase2.motor_parameters
    minimum_resistance_scale = float(spec.ood["reference_resistance_scale"][0])
    minimum_temperature = phase2.limits["winding_temperature_c"].minimum
    guard_resistance = (
        nominal.reference_resistance_ohm
        * minimum_resistance_scale
        * (
            1.0
            + nominal.resistance_temperature_coefficient_per_c
            * (minimum_temperature - nominal.reference_temperature_c)
        )
    )
    kp = float(spec.excitation["pi_proportional_gain_v_per_rad_s"])
    ki = float(spec.excitation["pi_integral_gain_v_per_rad"])
    closed_loop = family in {
        "pi_closed_loop",
        "pi_identification",
        "random_speed_reference",
    }

    for step in range(steps):
        if closed_loop:
            error = references[step] - states[:, 1]
            command = kp * error + ki * integral_error + identification[step]
            constrained = np.clip(command, -safe_voltage, safe_voltage)
        else:
            command = open_loop[step]
            constrained = np.clip(command, -safe_voltage, safe_voltage)

        back_emf = parameters["back_emf_constant_v_s_per_rad"] * states[:, 1]
        guard_low = np.maximum(
            -safe_voltage,
            back_emf - guard_resistance * current_guard,
        )
        guard_high = np.minimum(
            safe_voltage,
            back_emf + guard_resistance * current_guard,
        )
        if np.any(guard_low > guard_high):
            failed = [
                plans[index].trajectory_id
                for index in np.flatnonzero(guard_low > guard_high)
            ]
            raise DatasetGenerationError(
                "no feasible voltage inside the current safety envelope: "
                + ", ".join(failed)
            )
        constrained = np.clip(constrained, guard_low, guard_high)
        if closed_loop:
            residual = command - constrained
            integrate = (np.abs(residual) <= 1e-15) | (
                np.sign(error) != np.sign(residual)
            )
            integral_error[integrate] += error[integrate] * dt
        # Persisted controls are float32 by contract. Quantize before integration
        # so a trajectory can be replayed exactly from its on-disk control array.
        constrained = constrained.astype(np.float32).astype(np.float64)
        commanded[step] = command
        applied[step] = constrained
        for _ in range(substeps):
            states = _rk4_batch_step(
                states,
                constrained,
                loads,
                parameters,
                integration_step,
            )
            violation = _domain_violation_mask(states, phase2)
            if np.any(violation):
                failed = []
                for index in np.flatnonzero(violation):
                    state = states[index]
                    failed.append(
                        f"{plans[index].trajectory_id}"
                        f"(i={state[0]:.6g}, omega={state[1]:.6g}, T={state[2]:.6g})"
                    )
                raise DatasetGenerationError(
                    "FULL trajectory left the Phase-2 domain: " + ", ".join(failed)
                )
        state_history[step + 1] = states

    return tuple(
        RawPhase4Trajectory(
            states=state_history[:, index, :].astype(np.float32),
            commanded_voltages=commanded[:, index, None].astype(np.float32),
            applied_voltages=applied[:, index, None].astype(np.float32),
            load_torques=load_history[:, index, None].astype(np.float32),
            speed_references=references[:, index, None].astype(np.float32),
        )
        for index in range(count)
    )


def _exogenous_signals(
    plans: list[TrajectoryPlan] | tuple[TrajectoryPlan, ...],
    spec: Phase4Spec,
    steps: int,
    dt: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    count = len(plans)
    open_loop = np.zeros((steps, count), dtype=np.float64)
    references = np.zeros((steps, count), dtype=np.float64)
    identification = np.zeros((steps, count), dtype=np.float64)
    family = plans[0].excitation_family
    for index, plan in enumerate(plans):
        rng = np.random.default_rng(plan.seed + 1_000_003)
        if family == "prbs_voltage":
            open_loop[:, index] = _prbs(rng, spec, steps, dt, plan.novel_profile)
        elif family == "piecewise_constant_voltage":
            open_loop[:, index] = _piecewise_voltage(
                rng, spec, steps, dt, plan.novel_profile
            )
        elif family == "step_ramp_voltage":
            open_loop[:, index] = _step_ramp(
                rng, spec, steps, dt, plan.novel_profile
            )
        elif family == "multisine_voltage":
            open_loop[:, index] = _multisine(
                rng, spec, steps, dt, plan.novel_profile
            )
        elif family == "heating_cooling_cycle":
            open_loop[:, index] = _heating_cooling(
                rng, spec, steps, dt, plan.novel_profile
            )
        elif family in {
            "pi_closed_loop",
            "pi_identification",
            "random_speed_reference",
        }:
            references[:, index] = _speed_reference(
                rng,
                spec,
                steps,
                dt,
                family=family,
                novel=plan.novel_profile,
            )
            if family == "pi_identification":
                identification[:, index] = _identification_signal(
                    rng,
                    spec,
                    steps,
                    dt,
                    plan.novel_profile,
                )
        else:
            raise ValueError(f"unsupported excitation family: {family}")
    return open_loop, references, identification


def _prbs(
    rng: np.random.Generator,
    spec: Phase4Spec,
    steps: int,
    dt: float,
    novel: bool,
) -> FloatArray:
    low, high = _pair(spec.excitation["open_loop_amplitude_v"])
    amplitude = float(rng.uniform(low, high))
    if novel:
        return _chirp(amplitude, steps, dt, phase=float(rng.uniform(0.0, 2.0 * np.pi)))
    hold = float(rng.uniform(*_pair(spec.excitation["prbs_hold_s"])))
    hold_steps = max(1, round(hold / dt))
    chunks = math.ceil(steps / hold_steps)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=chunks)
    return np.repeat(signs * amplitude, hold_steps)[:steps]


def _piecewise_voltage(
    rng: np.random.Generator,
    spec: Phase4Spec,
    steps: int,
    dt: float,
    novel: bool,
) -> FloatArray:
    if novel:
        amplitude = float(rng.uniform(*_pair(spec.excitation["open_loop_amplitude_v"])))
        return _chirp(amplitude, steps, dt, phase=float(rng.uniform(0.0, 2.0 * np.pi)))
    low_hold, high_hold = _pair(spec.excitation["piecewise_hold_s"])
    amplitude = float(max(_pair(spec.excitation["open_loop_amplitude_v"])))
    values: list[float] = []
    while len(values) < steps:
        hold_steps = max(1, round(float(rng.uniform(low_hold, high_hold)) / dt))
        level = float(rng.uniform(-amplitude, amplitude))
        values.extend([level] * hold_steps)
    return np.asarray(values[:steps], dtype=np.float64)


def _step_ramp(
    rng: np.random.Generator,
    spec: Phase4Spec,
    steps: int,
    dt: float,
    novel: bool,
) -> FloatArray:
    amplitude = float(max(_pair(spec.excitation["open_loop_amplitude_v"])))
    if novel:
        return _chirp(amplitude, steps, dt, phase=float(rng.uniform(0.0, 2.0 * np.pi)))
    segment_steps = max(2, round(float(rng.uniform(0.4, 1.0)) / dt))
    output = np.empty(steps, dtype=np.float64)
    previous = 0.0
    cursor = 0
    segment = 0
    while cursor < steps:
        length = min(segment_steps, steps - cursor)
        target = float(rng.uniform(-amplitude, amplitude))
        if segment % 2 == 0:
            output[cursor : cursor + length] = target
        else:
            output[cursor : cursor + length] = np.linspace(
                previous,
                target,
                length,
                endpoint=False,
            )
        previous = target
        cursor += length
        segment += 1
    return output


def _multisine(
    rng: np.random.Generator,
    spec: Phase4Spec,
    steps: int,
    dt: float,
    novel: bool,
) -> FloatArray:
    amplitude = float(rng.uniform(*_pair(spec.excitation["open_loop_amplitude_v"])))
    if novel:
        return _chirp(amplitude, steps, dt, phase=float(rng.uniform(0.0, 2.0 * np.pi)))
    time = np.arange(steps, dtype=np.float64) * dt
    frequencies = np.asarray([0.07, 0.17, 0.43, 0.91], dtype=np.float64)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=frequencies.size)
    signal = np.sum(
        np.sin(2.0 * np.pi * time[:, None] * frequencies + phases),
        axis=1,
    )
    maximum = max(float(np.max(np.abs(signal))), 1e-12)
    return amplitude * signal / maximum


def _heating_cooling(
    rng: np.random.Generator,
    spec: Phase4Spec,
    steps: int,
    dt: float,
    novel: bool,
) -> FloatArray:
    amplitude = float(max(_pair(spec.excitation["open_loop_amplitude_v"])))
    heating_steps = round(0.6 * steps)
    if novel:
        heating = _chirp(
            amplitude,
            heating_steps,
            dt,
            phase=float(rng.uniform(0.0, 2.0 * np.pi)),
        )
    else:
        hold_steps = max(1, round(float(rng.uniform(0.03, 0.08)) / dt))
        chunks = math.ceil(heating_steps / hold_steps)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=chunks)
        heating = np.repeat(signs * amplitude, hold_steps)[:heating_steps]
    return np.concatenate(
        (heating, np.zeros(steps - heating_steps, dtype=np.float64))
    )


def _speed_reference(
    rng: np.random.Generator,
    spec: Phase4Spec,
    steps: int,
    dt: float,
    *,
    family: str,
    novel: bool,
) -> FloatArray:
    low, high = _pair(
        spec.ood["novel_speed_reference_rad_s"]
        if novel
        else spec.domain["speed_reference_rad_s"]
    )
    if novel:
        time = np.arange(steps, dtype=np.float64) * dt
        duration = max(steps * dt, dt)
        phase = 2.0 * np.pi * (0.03 * time + 0.5 * 0.3 / duration * time**2)
        return max(abs(low), abs(high)) * np.sin(phase)
    if family == "pi_closed_loop":
        reference = float(rng.uniform(low, high))
        signal = np.full(steps, reference, dtype=np.float64)
        signal[: max(1, round(0.05 / dt))] = 0.0
        return signal
    hold_low, hold_high = _pair(spec.excitation["reference_hold_s"])
    values: list[float] = []
    while len(values) < steps:
        hold_steps = max(1, round(float(rng.uniform(hold_low, hold_high)) / dt))
        reference = float(rng.uniform(low, high))
        values.extend([reference] * hold_steps)
    return np.asarray(values[:steps], dtype=np.float64)


def _identification_signal(
    rng: np.random.Generator,
    spec: Phase4Spec,
    steps: int,
    dt: float,
    novel: bool,
) -> FloatArray:
    amplitude = float(rng.uniform(*_pair(spec.excitation["identification_amplitude_v"])))
    if novel:
        return _chirp(amplitude, steps, dt, phase=float(rng.uniform(0.0, 2.0 * np.pi)))
    time = np.arange(steps, dtype=np.float64) * dt
    return amplitude * (
        0.6 * np.sin(2.0 * np.pi * 0.7 * time + rng.uniform(0.0, 2.0 * np.pi))
        + 0.4 * np.sin(2.0 * np.pi * 1.9 * time + rng.uniform(0.0, 2.0 * np.pi))
    )


def _chirp(amplitude: float, steps: int, dt: float, *, phase: float) -> FloatArray:
    time = np.arange(steps, dtype=np.float64) * dt
    duration = max(steps * dt, dt)
    chirp_phase = 2.0 * np.pi * (0.05 * time + 0.5 * 1.2 / duration * time**2)
    return amplitude * np.sin(chirp_phase + phase)


def _parameter_arrays(
    plans: list[TrajectoryPlan] | tuple[TrajectoryPlan, ...],
    nominal: MotorParameters,
) -> dict[str, FloatArray]:
    count = len(plans)
    arrays = {
        "armature_inductance_h": np.full(count, nominal.armature_inductance_h),
        "reference_resistance_ohm": np.asarray(
            [
                nominal.reference_resistance_ohm * plan.reference_resistance_scale
                for plan in plans
            ]
        ),
        "reference_temperature_c": np.full(
            count, nominal.reference_temperature_c
        ),
        "resistance_temperature_coefficient_per_c": np.full(
            count, nominal.resistance_temperature_coefficient_per_c
        ),
        "back_emf_constant_v_s_per_rad": np.full(
            count, nominal.back_emf_constant_v_s_per_rad
        ),
        "torque_constant_n_m_per_a": np.full(
            count, nominal.torque_constant_n_m_per_a
        ),
        "inertia_kg_m2": np.asarray(
            [nominal.inertia_kg_m2 * plan.inertia_scale for plan in plans]
        ),
        "viscous_friction_n_m_s_per_rad": np.full(
            count, nominal.viscous_friction_n_m_s_per_rad
        ),
        "thermal_capacitance_j_per_c": np.asarray(
            [
                nominal.thermal_capacitance_j_per_c
                * plan.thermal_capacitance_scale
                for plan in plans
            ]
        ),
        "thermal_resistance_c_per_w": np.full(
            count, nominal.thermal_resistance_c_per_w
        ),
        "ambient_temperature_c": np.full(count, nominal.ambient_temperature_c),
    }
    return {name: np.asarray(value, dtype=np.float64) for name, value in arrays.items()}


def _derivative_batch(
    states: FloatArray,
    voltages: FloatArray,
    loads: FloatArray,
    p: dict[str, FloatArray],
) -> FloatArray:
    current = states[:, 0]
    speed = states[:, 1]
    temperature = states[:, 2]
    resistance = p["reference_resistance_ohm"] * (
        1.0
        + p["resistance_temperature_coefficient_per_c"]
        * (temperature - p["reference_temperature_c"])
    )
    current_rate = (
        voltages - resistance * current - p["back_emf_constant_v_s_per_rad"] * speed
    ) / p["armature_inductance_h"]
    speed_rate = (
        p["torque_constant_n_m_per_a"] * current
        - p["viscous_friction_n_m_s_per_rad"] * speed
        - loads
    ) / p["inertia_kg_m2"]
    copper_loss = resistance * current**2
    cooling = (temperature - p["ambient_temperature_c"]) / p[
        "thermal_resistance_c_per_w"
    ]
    temperature_rate = (
        copper_loss - cooling
    ) / p["thermal_capacitance_j_per_c"]
    return np.column_stack((current_rate, speed_rate, temperature_rate))


def _rk4_batch_step(
    states: FloatArray,
    voltages: FloatArray,
    loads: FloatArray,
    parameters: dict[str, FloatArray],
    step_s: float,
) -> FloatArray:
    k1 = _derivative_batch(states, voltages, loads, parameters)
    k2 = _derivative_batch(states + step_s * k1 / 2.0, voltages, loads, parameters)
    k3 = _derivative_batch(states + step_s * k2 / 2.0, voltages, loads, parameters)
    k4 = _derivative_batch(states + step_s * k3, voltages, loads, parameters)
    return states + step_s * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def _domain_violation_mask(states: FloatArray, phase2: Phase2Spec) -> NDArray[np.bool_]:
    finite = np.isfinite(states).all(axis=1)
    current = phase2.limits["armature_current_a"]
    speed = phase2.limits["angular_speed_rad_s"]
    temperature = phase2.limits["winding_temperature_c"]
    return ~(
        finite
        & (states[:, 0] >= current.minimum)
        & (states[:, 0] <= current.maximum)
        & (states[:, 1] >= speed.minimum)
        & (states[:, 1] <= speed.maximum)
        & (states[:, 2] >= temperature.minimum)
        & (states[:, 2] <= temperature.maximum)
    )


def _trajectory_record(
    plan: TrajectoryPlan,
    trajectory: RawPhase4Trajectory,
    *,
    path: Path,
    spec: Phase4Spec,
    phase2: Phase2Spec,
) -> dict[str, Any]:
    states = trajectory.states
    safe_voltage = float(spec.domain["safe_voltage_limit_v"])
    saturation = np.abs(
        trajectory.commanded_voltages - trajectory.applied_voltages
    ) > 1e-7
    return {
        "trajectory_id": plan.trajectory_id,
        "path": path.as_posix(),
        "split": plan.split,
        "excitation_family": plan.excitation_family,
        "seed": plan.seed,
        "duration_s": plan.duration_s,
        "transitions": trajectory.transitions,
        "initial_state": {
            "armature_current_a": plan.initial_current_a,
            "angular_speed_rad_s": plan.initial_speed_rad_s,
            "winding_temperature_c": plan.initial_temperature_c,
        },
        "load_torque_n_m": plan.load_torque_n_m,
        "ood_axis": plan.ood_axis,
        "parameter_scales": plan.parameter_scales,
        "novel_profile": plan.novel_profile,
        "simulator": "FULL",
        "terminated": False,
        "saturation_fraction": float(np.mean(saturation)),
        "ranges": {
            "armature_current_a": [float(np.min(states[:, 0])), float(np.max(states[:, 0]))],
            "angular_speed_rad_s": [float(np.min(states[:, 1])), float(np.max(states[:, 1]))],
            "winding_temperature_c": [
                float(np.min(states[:, 2])),
                float(np.max(states[:, 2])),
            ],
            "armature_voltage_v": [
                float(np.min(trajectory.applied_voltages)),
                float(np.max(trajectory.applied_voltages)),
            ],
        },
        "safe_voltage_limit_v": safe_voltage,
        "current_guard_a": float(spec.domain["current_guard_a"]),
        "control_period_s": phase2.integration_settings.control_period_s,
        "content_sha256": trajectory_content_sha256(trajectory),
    }


def _build_manifest(
    *,
    spec: Phase4Spec,
    phase2: Phase2Spec,
    profile: DatasetProfile,
    records: list[dict[str, Any]],
    transition_count: int,
    normalization: NormalizationStatistics,
) -> dict[str, Any]:
    package_root = Path(__file__).resolve().parents[3]
    phase2_config = package_root / "configs" / "phase2.toml"
    phase4_config = package_root / "configs" / "phase4.toml"
    split_counts = {
        split: sum(record["split"] == split for record in records)
        for split in SPLIT_NAMES
    }
    return {
        "schema_version": 1,
        "dataset_id": spec.dataset["dataset_id"],
        "dataset_version": spec.dataset["version"],
        "profile": profile.name,
        "generator": spec.dataset["generator"],
        "simulator": {
            "name": "FULL",
            "phase2_config_sha256": _file_sha256(phase2_config),
            "phase4_config_sha256": _file_sha256(phase4_config),
            "control_period_s": phase2.integration_settings.control_period_s,
            "integrator": "classical_rk4",
            "integrator_step_s": phase2.integration_settings.integrator_step_s,
            "parameters": asdict(phase2.motor_parameters),
        },
        "storage": {
            "format": spec.dataset["storage_format"],
            "dtype": spec.dataset["dtype"],
            "compression": bool(spec.dataset["compression"]),
            "time_axis": "implicit_uniform_control_period",
        },
        "features": {
            name: list(values) for name, values in spec.features.items()
        },
        "split_policy": {
            "name": spec.splits["policy"],
            "id_policy": spec.splits["id_policy"],
            "names": list(SPLIT_NAMES),
        },
        "excitation_families": list(EXCITATION_FAMILIES),
        "split_counts": split_counts,
        "trajectory_count": len(records),
        "transition_count": transition_count,
        "seed_base": profile.seed_base,
        "normalization": {
            "path": "normalization.npz",
            "fit_split": normalization.fit_split,
            "observation_features": list(
                spec.normalization["observation_features"]
            ),
            "control_features": list(spec.normalization["control_features"]),
            "temperature_used": False,
            "observation_count": normalization.observation_count,
            "control_count": normalization.control_count,
        },
        "trajectories": records,
    }


def _sample_training_temperature(
    spec: Phase4Spec,
    rng: np.random.Generator,
    *,
    band_index: int,
) -> float:
    band = _pair(spec.domain["temperature_bands_c"][band_index])
    return float(rng.uniform(*band))


def _sample_away_from_one(
    rng: np.random.Generator,
    bounds: tuple[float, float],
) -> float:
    low, high = bounds
    if rng.random() < 0.5:
        return float(rng.uniform(low, 0.95))
    return float(rng.uniform(1.05, high))


def _pair(value: Any) -> tuple[float, float]:
    return float(value[0]), float(value[1])


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
