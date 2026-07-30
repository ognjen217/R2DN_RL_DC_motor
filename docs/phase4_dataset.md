# Phase 4 — FULL simulator dataset

## Purpose

Phase 4 converts the validated FULL electrothermal simulator into a
reproducible system-identification dataset. It does not train ISO models or
R2DN.

## Stored trajectory contract

Every trajectory is stored as one compressed NPZ file:

| Array | Shape | Meaning |
|---|---|---|
| `states` | `(T + 1, 3)` | current, speed, winding temperature |
| `commanded_voltages` | `(T, 1)` | controller/excitation request |
| `applied_voltages` | `(T, 1)` | safe voltage actually seen by FULL |
| `load_torques` | `(T, 1)` | known plant disturbance |
| `speed_references` | `(T, 1)` | closed-loop reference, zero for open loop |

The time axis is implicit and uniform with `Ts = 1 ms`. FULL uses ten
classical-RK4 substeps of `0.1 ms` per stored transition.

`RawPhase4Trajectory.model_view()` is the enforced model boundary. It returns:

- observations `[current, speed]`;
- controls `[applied voltage]`.

Temperature, load, reference, and commanded voltage remain available only in
the raw/evaluation object.

## Excitation families

Each split contains all eight families:

1. PRBS voltage;
2. piecewise-constant voltage;
3. alternating step/ramp voltage;
4. multisine voltage;
5. safe PI closed loop;
6. PI plus small identification excitation;
7. random piecewise speed references;
8. heating followed by cooling.

The input generator is limited to `±20 V`, inside the plant actuator domain of
`±48 V`. A conservative 6.5 A envelope further limits applied voltage using
measured speed and the lowest declared cold resistance. It does not use hidden
temperature. Requested voltage remains stored separately, so safety
intervention is observable rather than silently discarded.

## Splits

- `train`: model fitting and the only source of normalization statistics;
- `validation`: model selection with new complete trajectories;
- `id_test`: untouched new seeds inside the training domain;
- `ood_test`: higher initial temperature, stronger load, novel profiles, or
  declared physical-parameter shifts.

No individual time sample is randomly assigned to a split.

## Profiles

| Profile | Train | Validation | ID test | OOD test | Total transitions |
|---|---:|---:|---:|---:|---:|
| `ci` | 8 | 8 | 8 | 8 | 7,600 |
| `final` | 224 | 32 | 32 | 32 | 4,800,000 |

The final profile uses 12 s for seven families and 36 s for each
heating/cooling trajectory.

## Reproducibility and integrity

`manifest.json` records the simulator parameters, configuration hashes, seeds,
splits, excitation metadata, observed ranges, and one logical SHA-256 for every
trajectory. A canonical fingerprint identifies the complete dataset.

The validator recomputes file hashes and train-only normalization, checks every
array and physical limit, verifies the deterministic seed plan, and confirms
ID/OOD separation.

## Commands

Fast local/CI build:

```bash
python -m pip install -e ".[dev,phase4]"
python -m r2dn_dc_motor.validate_phase4 \
  --generate \
  --profile ci \
  --output-dir data/phase4-ci \
  --artifacts-dir results/phase4-ci
```

Locked final build:

```bash
python -m r2dn_dc_motor.validate_phase4 \
  --generate \
  --profile final \
  --output-dir data/phase4-full-v1 \
  --artifacts-dir results/phase4-final
```

Existing datasets can be checked without regenerating:

```bash
python -m r2dn_dc_motor.validate_phase4 \
  --profile final \
  --dataset data/phase4-full-v1
```
