from dataclasses import replace

import numpy as np
import pytest

from r2dn_dc_motor.data import (
    Phase4Dataset,
    RawPhase4Trajectory,
    build_trajectory_plans,
    generate_phase4_dataset,
    simulate_trajectory_group,
)
from r2dn_dc_motor.phase2_spec import load_phase2_spec
from r2dn_dc_motor.phase4_spec import OOD_AXES, SPLIT_NAMES, load_phase4_spec
from r2dn_dc_motor.plants import MotorState


@pytest.fixture
def phase2():
    return load_phase2_spec()


@pytest.fixture
def phase4(phase2):
    return load_phase4_spec(phase2=phase2)


def test_plans_have_disjoint_ids_seeds_and_declared_ood_axes(phase4):
    plans = build_trajectory_plans(phase4, phase4.profile("ci"))
    ids = [plan.trajectory_id for plan in plans]
    seeds = [plan.seed for plan in plans]

    assert len(ids) == len(set(ids)) == 32
    assert len(seeds) == len(set(seeds)) == 32
    assert {plan.split for plan in plans} == set(SPLIT_NAMES)
    assert {
        plan.ood_axis for plan in plans if plan.split == "ood_test"
    } == set(OOD_AXES)
    assert all(plan.ood_axis is None for plan in plans if plan.split != "ood_test")


def test_vectorized_generator_matches_scalar_full_plant(phase2, phase4):
    plans = build_trajectory_plans(phase4, phase4.profile("ci"))
    plan = next(
        plan
        for plan in plans
        if plan.split == "train" and plan.excitation_family == "prbs_voltage"
    )
    generated = simulate_trajectory_group([plan], phase4, phase2)[0]
    parameters = replace(
        phase2.motor_parameters,
        reference_resistance_ohm=(
            phase2.motor_parameters.reference_resistance_ohm
            * plan.reference_resistance_scale
        ),
        inertia_kg_m2=(
            phase2.motor_parameters.inertia_kg_m2 * plan.inertia_scale
        ),
        thermal_capacitance_j_per_c=(
            phase2.motor_parameters.thermal_capacitance_j_per_c
            * plan.thermal_capacitance_scale
        ),
    )
    scalar = phase2.build_plant(parameters=parameters).rollout(
        generated.applied_voltages[:, 0],
        initial_state=MotorState(
            plan.initial_current_a,
            plan.initial_speed_rad_s,
            plan.initial_temperature_c,
        ),
        load_torques_n_m=plan.load_torque_n_m,
    )

    assert scalar.terminated is False
    np.testing.assert_allclose(
        generated.states,
        scalar.states.astype(np.float32),
        rtol=2e-7,
        atol=2e-7,
    )


def test_raw_trajectory_model_view_drops_all_evaluation_only_features():
    trajectory = RawPhase4Trajectory(
        states=np.asarray([[1.0, 2.0, 30.0], [1.1, 2.2, 30.1]]),
        commanded_voltages=np.asarray([[4.5]]),
        applied_voltages=np.asarray([[4.0]]),
        load_torques=np.asarray([[0.15]]),
        speed_references=np.asarray([[100.0]]),
    )

    model = trajectory.model_view()

    assert model.observations.shape == (2, 1, 2)
    assert model.controls.shape == (1, 1, 1)
    np.testing.assert_array_equal(model.observations[:, 0], trajectory.states[:, :2])
    np.testing.assert_array_equal(model.controls[:, 0], trajectory.applied_voltages)


@pytest.mark.phase4_gate
def test_ci_generation_is_bitwise_reproducible_by_logical_fingerprint(
    tmp_path,
    phase2,
    phase4,
):
    first = generate_phase4_dataset(
        tmp_path / "first",
        profile_name="ci",
        spec=phase4,
        phase2=phase2,
    )
    second = generate_phase4_dataset(
        tmp_path / "second",
        profile_name="ci",
        spec=phase4,
        phase2=phase2,
    )

    assert first.fingerprint == second.fingerprint
    assert first.trajectory_count == second.trajectory_count == 32
    assert first.transition_count == second.transition_count == 7_600
    first_dataset = Phase4Dataset(first.root)
    second_dataset = Phase4Dataset(second.root)
    assert first_dataset.manifest == second_dataset.manifest


def test_existing_output_requires_explicit_overwrite(tmp_path, phase2, phase4):
    output = tmp_path / "dataset"
    output.mkdir()

    with pytest.raises(FileExistsError, match="overwrite"):
        generate_phase4_dataset(
            output,
            profile_name="ci",
            spec=phase4,
            phase2=phase2,
        )
