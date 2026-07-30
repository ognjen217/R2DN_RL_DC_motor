# ADR 0005 — Versioned Phase-4 dataset

## Status

Accepted.

## Context

The world model must infer the hidden thermal regime from measured history
without receiving winding temperature. Random sample-level splitting would put
near-identical adjacent samples from one simulated trajectory into multiple
splits and produce optimistic validation results.

The Phase-2 scalar plant is deliberately diagnostic and object-oriented. A
multi-million-sample dataset would be unnecessarily slow if every control
period constructed Python objects and closures.

## Decision

- Only the Phase-2 FULL electrothermal equations generate Phase-4 data.
- Classical RK4 remains fixed at a 1 ms control period and 0.1 ms substep.
- The generator evaluates the same equations in vectorized trajectory groups.
  A regression test compares it against the scalar FULL implementation.
- Each complete trajectory is one compressed NPZ file.
- `train`, `validation`, `id_test`, and `ood_test` are disjoint sets of files
  and trajectory IDs.
- Raw files retain state `[i, omega, T]`, applied and commanded voltage, load,
  and speed reference.
- Applied voltage is constrained by `±20 V` and a deterministic,
  temperature-free 6.5 A current-safety envelope based on measured speed and
  the lowest declared cold resistance; the unconstrained command is retained.
- The model loader exposes only `[i, omega]` and applied voltage.
- Normalization is fit only on `train`; temperature is never used.
- Logical array hashes and a canonical manifest fingerprint identify content
  independently of ZIP timestamps or file-system metadata.
- The `ci` profile exercises the full contract quickly. The locked `final`
  profile contains 320 trajectories and 4.8 million transitions.

## Consequences

The dataset can be reproduced from configuration and seeds, audited without
training a model, and consumed without accidental temperature leakage. Large
trajectory files remain outside Git and can be regenerated or distributed as a
separate data artifact.
