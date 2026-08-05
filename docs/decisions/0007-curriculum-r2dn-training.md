# ADR 0007: R2DN curriculum training and validation-only selection

## Numerical finite-update guard

Every optimizer update is applied only when the complete loss tuple, raw
gradient norm, every gradient leaf, the current state, and the candidate
parameter/Adam state are finite.
An invalid minibatch leaves both the parameters and Adam state unchanged. The
deterministic sampler then supplies a replacement window, so the locked update
budget counts only finite optimizer updates. Rejected minibatches are recorded
in the training history, and a stage fails if more than 5% of its requested
updates (with a minimum budget of ten retries) are rejected.

This guard does not alter the loss, optimizer, learning rate, gradient clipping,
curriculum horizons, or selection windows. It prevents asynchronous detection
from observing an overflow only after a non-finite update has already poisoned
the run.

## Status

Accepted for Phase 6.

## Decision

The official pinned `ContractingR2DN` is trained only through the Phase-4
temperature-free model view:

```text
input  = normalized [current_k, speed_k, applied_voltage_k]
target = normalized [current_k+1, speed_k+1]
```

Every window begins with measured-observation burn-in. The predictive part of
the loss combines teacher-forced one-step error and an autoregressive rollout
that receives no measured observations after burn-in. Burn-in reconstruction
error is retained with a small weight.

The final profile first compares latent dimensions 4, 6, and 8 with a small
pilot budget. It then trains the selected architecture from scratch with seeds
17, 29, and 43. The saved checkpoint is the seed with minimum held-out
validation free-rollout NRMSE over 5 seconds.
All latent candidates share one fixed set of pilot-validation windows, and all
three final seeds share another fixed set, so model-selection noise comes from
training rather than from changing evaluation samples.

ID and OOD test trajectories never select the architecture, seed, or
checkpoint. Their clean comparison with ISO-CAL belongs to Phase 7.

## Consequences

- the hidden temperature and evaluation-only signals cannot leak into fitting;
- the current/speed scale difference is removed by the train-only Phase-4
  normalizer;
- architecture and checkpoint selection reward long autoregressive behavior,
  not only one-step accuracy;
- the selected checkpoint is bound to the dataset fingerprint, upstream R2DN
  commit, normalization file, full training history, and random seed;
- Phase 6 proves a reproducible training pipeline and stable checkpoint, but it
  deliberately does not claim the Phase-7 world-model quality gate.
