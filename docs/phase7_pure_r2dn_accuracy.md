# Phase 7 — improved pure R2DN and hidden-thermal test bank

## Research question

Can a pure contractive R2DN, using only current, angular speed, and applied
voltage, learn the effects of hidden thermal dynamics more accurately than
incomplete isothermal physical models while remaining stable in long
autoregressive simulation?

`FULL/RK4` is ground truth. It evolves temperature internally and uses the true
resistance-temperature law. `ISO-NOM`, `ISO-CAL`, and every R2DN candidate are
temperature-blind.

## What Phase 7 changes

The v2 dataset keeps the Phase-4 storage/interface contract but expands the
identification domain:

- 320 whole trajectories and 21.6 million transitions;
- 60 s standard trajectories and 120 s heating/cooling/reheating trajectories;
- independently sampled multisine frequencies over 0.05--4 Hz;
- random amplitudes and phases, steps, PRBS, chirps, and closed-loop signals;
- cold, medium, and hot initial temperature bands;
- disjoint train, validation, ID-test, and OOD-test trajectories.

Temperature is persisted only for FULL-reference generation and analysis. The
model view remains exactly `[current, speed, applied voltage]`.

Three pure-R2DN variants isolate the experimental factors:

| Variant | Data | Objective | Width |
|---|---|---|---:|
| `broadband_standard` | broadband v2 | historical Phase-6 loss | 32 |
| `broadband_delta_multiscale` | broadband v2 | increment auxiliary + 1/10/100/1000/5000-step losses | 32 |
| `broadband_delta_multiscale_wide` | broadband v2 | same enhanced objective | 64 |

All use latent size 16, burn-in 250, seeds 17/29/43, the official
`ContractingR2DN`, and a positive contractivity margin. The increment loss is
fit with train-only increment statistics. It does not expose temperature and
does not change the absolute-state output used during deployment.

## Reproducible execution

### 1. Preflight the frozen long-horizon bank with FULL/RK4 only

Phase 7 no longer reuses the Phase-6B stress inputs. Its dedicated scenarios
are locked in `configs/phase7.toml`:

- PRBS: `5 V`, 250-step hold, seed `78011`;
- sine: `7 V`, `0.25 Hz`, phase `0.35 rad`;
- multisine: amplitudes `[3, 2.5, 2] V`, frequencies
  `[0.07, 0.37, 1.43] Hz`, phases `[0.2, 1.1, 2.0] rad`.

Before any R2DN evaluation, run all nine 1000-second FULL/RK4 references:

```bash
mkdir -p results/phase7/current_test_bank_preflight

python -m r2dn_dc_motor.preflight_thermal_test_bank \
  --evaluation-dataset data/phase4-full-v1 \
  --profile final \
  --duration-s 1000 \
  --output-dir results/phase7/current_test_bank_preflight \
  2>&1 | tee results/phase7/current_test_bank_preflight/preflight.log
```

This command does not import JAX and does not require CUDA or an R2DN
checkpoint. It passes only if every reference completes one million control
steps without any plant-domain termination and every trajectory satisfies
`Tmax <= 110 C`. The plant limit remains `120 C`; the lower preflight ceiling
provides a 10 C safety margin. The JSON report freezes the dataset fingerprint,
whole-trajectory anchors, scenario parameters, and a test-bank fingerprint.

### 2. Multi-trajectory baseline for the current Phase-6E model

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m r2dn_dc_motor.evaluate_thermal_test_bank \
  --require-cuda \
  --evaluation-dataset data/phase4-full-v1 \
  --iso-cal-checkpoint checkpoints/phase5/iso_cal.json \
  --r2dn-model current checkpoints/phase6e/r2dn-v1 data/phase4-full-v1 \
  --profile final \
  --duration-s 1000 \
  --preflight-report \
    results/phase7/current_test_bank_preflight/thermal_test_bank_preflight.json \
  --output-dir results/phase7/current_test_bank
```

This replaces the single-trajectory conclusion with mean, median, worst case,
win count, and divergence count across ID, excitation-OOD, and thermal-OOD
cases.

### 3. Generate and validate the broadband FULL dataset

```bash
python -m r2dn_dc_motor.validate_phase4 \
  --config configs/phase4_broadband.toml \
  --generate \
  --profile final \
  --output-dir data/phase4-broadband-v2 \
  --artifacts-dir results/phase7/dataset
```

### 4. Refit ISO-CAL once on the new train split

```bash
python -m r2dn_dc_motor.validate_phase5 \
  --fit \
  --dataset data/phase4-broadband-v2 \
  --checkpoint checkpoints/phase7/iso_cal.json \
  --output-dir results/phase7/iso_cal
```

The fit still uses only current, speed, and applied voltage. It is bound to the
new dataset fingerprint and cannot read validation/test trajectories.

### 5. Train/resume the pure-R2DN ablation

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m r2dn_dc_motor.validate_phase7 \
  --train \
  --profile final \
  --require-cuda \
  --dataset data/phase4-broadband-v2 \
  --cache-dir checkpoints/phase7/run-cache-v1 \
  --checkpoint-dir checkpoints/phase7/r2dn-v1 \
  --output-dir results/phase7/training
```

Each variant/seed is written to an atomic run cache, so interruption does not
discard completed runs. Selection uses only fixed validation windows over 5000
steps. ID/OOD test-bank results cannot select the model.

### 6. Preflight and run the final current-versus-improved comparison

The broadband-v2 evaluation dataset selects a different set of held-out
anchors, so it requires its own preflight report:

```bash
python -m r2dn_dc_motor.preflight_thermal_test_bank \
  --evaluation-dataset data/phase4-broadband-v2 \
  --profile final \
  --duration-s 1000 \
  --output-dir results/phase7/final_test_bank_preflight
```

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m r2dn_dc_motor.evaluate_thermal_test_bank \
  --require-cuda \
  --evaluation-dataset data/phase4-broadband-v2 \
  --iso-cal-checkpoint checkpoints/phase7/iso_cal.json \
  --r2dn-model current checkpoints/phase6e/r2dn-v1 data/phase4-full-v1 \
  --r2dn-model improved checkpoints/phase7/r2dn-v1 data/phase4-broadband-v2 \
  --profile final \
  --duration-s 1000 \
  --preflight-report \
    results/phase7/final_test_bank_preflight/thermal_test_bank_preflight.json \
  --output-dir results/phase7/final_test_bank
```

The checkpoint's own training dataset is supplied only for provenance and
normalization validation. Both learned models are then evaluated from the same
v2 anchors, with the same future voltage and FULL/RK4 reference.

## Decision rule

- If improved R2DN beats ISO-NOM and ISO-CAL, the central accuracy hypothesis
  is supported.
- If it beats ISO-NOM but not ISO-CAL, data-driven identification compensates
  for nominal mismatch, but calibrated physics remains more accurate.
- If it beats neither, the result demonstrates that contraction ensures stable
  rollouts but not superior hidden-dynamics identification.

The optional hybrid resistance model is intentionally excluded until this
comparison is complete.
