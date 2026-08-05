# ADR 0008: Repeated latent search and outer-loop stress testing

## Status

Accepted for Phase 6B.

## Context

The Phase-6 pilot selected latent dimension 8 at the upper boundary of the
tested set \(\{4,6,8\}\). One seed per pilot architecture was not sufficient to
distinguish a real capacity trend from initialization variance. The official
R2DN contraction condition also does not directly certify the closed
autoregressive map produced by feeding \(\hat y_k\) into the next regressor.

## Decision

Phase 6B evaluates latent dimensions 4, 6, 8, 10, 12, and 16 with three fixed
pilot seeds per dimension and identical validation windows. Architecture
selection minimizes the median validation free-rollout NRMSE. A smaller latent
within 3% of the best median is preferred. Latent 24 is evaluated only if
latent 16 improves on latent 12 by strictly more than 5%.

Final training retains the Phase-6 curriculum, seeds 17/29/43, and the
5000-step validation selection criterion. A matching validated Phase-6
checkpoint may be reused if latent 8 wins again.

After selection, the model is evaluated without affecting selection:

- held-out validation/ID/OOD replay measures error growth through 10 seconds;
- eight synthetic voltage families stress validation/ID/OOD burn-in anchors
  through \(10^4\), \(10^5\), and \(10^6\) autoregressive steps;
- paired \(10^{-3}\) perturbations of latent and predicted output measure
  sensitivity of the outer feedback loop;
- physical bounds, normalized bounds, latent-norm magnitude and tail growth,
  electromagnetic/kinetic energy, input work, and output tail growth are
  recorded.

## Consequences

- architecture selection is robust to one lucky initialization;
- the search boundary can expand once under a predeclared rule;
- ID, OOD, replay, and stress results remain post-selection evidence;
- every run is resumable and bound to the dataset and complete protocol hash;
- passing Phase 6B supports boundedness for the tested million-step scenarios,
  but does not claim formal infinite-horizon stability or passivity;
- fair R2DN versus ISO-CAL comparison remains Phase 7.
