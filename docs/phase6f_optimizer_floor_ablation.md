# Phase 6F — latent-16 optimizer-floor ablation

Phase 6F tests whether the remaining long-rollout error is primarily caused by
optimization rather than network capacity. It freezes the Phase-6E winner
(`latent=16`, `seed=43`, `feature_size=32`, hidden layers `[32, 32]`), the
dataset, loss, burn-in, and rollout curriculum.

The ordered candidates are:

1. the existing Phase-6E checkpoint: constant learning rate `1e-3`, 3000
   optimizer updates;
2. a fresh run with cosine decay from `1e-3` to `1e-5`, 3000 updates;
3. a fresh run with the same cosine endpoints and 6000 updates.

The first comparison isolates the learning-rate schedule. The second isolates
the additional update budget under the same schedule. All fresh runs reuse the
same initialization seed and deterministic training-window sampler as the
baseline.

Selection uses the median combined NRMSE on three new 100-second multisine
FULL/RK4 references. The Phase-6E selection excitations and the canonical
Phase-6C 1000-second multisine are excluded. A lower-budget candidate is kept
when its score is within 2% of the best result.

Run the final study with:

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m r2dn_dc_motor.validate_phase6f \
  --train \
  --profile final \
  --require-cuda \
  --dataset data/phase4-full-v1 \
  --phase6b-report results/phase6b/phase6b_latent_and_stability.json \
  --phase6e-checkpoint checkpoints/phase6e/r2dn-v1 \
  --cache-dir checkpoints/phase6f/run-cache-v1 \
  --checkpoint-dir checkpoints/phase6f/r2dn-v1 \
  --output-dir results/phase6f
```

The command is resumable through its per-variant cache. The selected checkpoint
can be passed directly to `compare_r2dn_rk4` for the final 1000-second
post-selection benchmark.
