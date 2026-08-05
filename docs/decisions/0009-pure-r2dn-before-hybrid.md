# ADR-0009: Evaluate an improved pure R2DN before a hybrid model

## Status

Accepted for Phase 7.

## Context

The canonical 1000 s trajectory showed that the Phase-6E R2DN is stable but
systematically less accurate than ISO-CAL. Combined NRMSE was approximately
0.193 for R2DN and 0.0085 for ISO-CAL. The R2DN error was already present at
1 s and did not grow materially by 1000 s, which indicates representation or
identification bias rather than long-horizon divergence.

A hybrid ISO-CAL plus learned resistance correction could likely reduce the
error, but it would answer a different research question and would require a
new stability analysis for the complete neural-physical transition.

## Decision

Phase 7 keeps the official pure `ContractingR2DN` as the primary learned model.
Temperature remains unavailable to the model. Improvements are introduced as
auditable ablations:

1. a broadband, long-duration FULL dataset;
2. a train-only increment-normalized auxiliary objective and cumulative
   multi-horizon rollout loss;
3. a width increase from 32 to 64 only after the data/loss variant;
4. three fixed seeds and validation-only model selection;
5. a frozen ID, excitation-OOD, and thermal-OOD test bank evaluated at
   1/10/100/1000 s.

The R2DN output remains the absolute normalized next state. The increment term
is an auxiliary loss, not an external residual connection, so the upstream
contracting parameterization and deployed autoregressive interface are not
replaced. The positive Equation-20 residual is checked for every trained run.

## Consequences

- A gain from the broadband-only variant can be attributed to data coverage.
- A further gain from the delta/multiscale variant can be attributed to the
  objective rather than architecture width.
- ISO-CAL may still win. That is a valid result and must be reported.
- The optional hybrid resistance model and MPC remain later extensions, after
  the pure-model comparison is complete.

