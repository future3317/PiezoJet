# Predictive-validity replay protocol

Date: 2026-07-16

This protocol begins only after the mathematical/implementation audit passed.
It does not change the registered losses, strict gates, split, seeds, or test
panel.

## Two independent questions

The two-tower design makes the exposure replay two parallel experiments:

1. **Physical experiment:** branch580/strict149 training exposure tests
   `U_{eta,delta}`, true-BEC `Z*^T U_{eta,delta}`, predicted-BEC ionic response,
   BEC, Phi, and Lambda generalization.
2. **Macro experiment:** macro4961 exposure tests only the independent total
   tensor predictor against the matched direct-total control.

Total-only gradients cannot enter physical factors. Macro improvement is
therefore a negative control and software-isolation result, not evidence that
4,961 total labels repair ionic amplitude collapse or that factorization
improves total prediction.

## Frozen protocol

- passes: `1, 5, 10, 20`;
- seeds: `42, 7, 1729`;
- formula-disjoint validation/test: immutable `10/20`;
- checkpoint selection: validation loss only;
- every registered test point is reported; test20 never selects a pass,
  checkpoint, loss weight, or later architecture;
- physical and direct control share macro exposure, structural checkpoint,
  graph/tensor conventions, seed, and validation-selection rule.

## Additional read-only diagnostics

The normal-equation weight remains `0.1`. Each strict joint pass now records:

- raw `||grad_thetaU L_normal|| / (||grad_thetaU L_U|| + eps)`;
- the same ratio after registered loss weights;
- normal-equation optical residual energy and mode counts in
  `<delta`, `[delta,3delta)`, `[3delta,10delta)`, and `>=10delta` using true
  optical eigenvalues.

The evaluator additionally records the response-weighted log stiffness bias

`log(|v*^T Phi_pred v*|+eps) - log(|lambda*|+eps)`,

weighted by true mode-effective charge, true strain coupling, and true
regularized response magnitude. Its Pearson correlation with per-material
direct-ionic amplitude ratio is diagnostic only.

## Reporting contract

For every pass and seed, report:

- total TRS;
- direct-`U` ionic material-macro cosine, amplitude ratio, and MAE skill;
- active-panel cosine and active count;
- true-BEC `Z*^T U_pred` cosine and amplitude ratio;
- BEC/Phi/Lambda/U errors;
- predicted-spectrum regions and response-weighted stiffness bias;
- normal/direct-`U` head-gradient ratio and four residual-energy fractions;
- complete exposure ledger.

Across seeds, report individual results, mean and sample SD, and a hierarchical
bootstrap that first resamples seeds and then complete materials within each
sampled seed. The physical-macro minus matched-direct total TRS interval is
paired by seed and material. An ionic MAE-skill interval crossing zero is
reported as `inconclusive`, never as a small improvement.

## Predeclared interpretation

| Observation | Adjudication |
| --- | --- |
| Train U/BEC/Phi fails to improve by 20 passes | Optimization, scale, head parameterization, or normal-loss weighting remains unresolved; do not claim data scarcity |
| Train improves while validation/test U and ionic stay near zero | Formula-disjoint OOD or strict-label coverage becomes the leading explanation |
| True-BEC `Z*^T U_pred` improves but predicted-BEC ionic collapses | Joint BEC/U direction and scale is the leading composite bottleneck |
| U Frobenius error improves but true-BEC response does not | Ordinary U supervision misses response-active information; test response-weighted oracle/loss using validation only |
| Spectrum hardens and stiffness bias correlates negatively with amplitude | Hard-end resolvent shortcut gains support |
| Macro total improves while ionic does not | Expected under tower isolation; only macro labels/predictor succeeded |
| Ionic peaks at 5--10 passes and declines at 20 | Overfitting; use validation selection and do not select from the test curve |

## Deferred proposals

- `r=6` factorization is not a compression: `V in R^(3N x 6)` plus
  `C in R^(6 x 6)` predicts about `18N+36` values versus direct U's `18N`, while
  adding latent gauge. It is not implemented.
- `r<6` remains deferred because rank-4 has 19.81% true-BEC response error.
- mass/scale, latent decoder, mode slots, and factor pseudo-labeling remain
  outside this replay.

## End-to-end instrumentation check: pass 1, seed 42

This is one of twelve preregistered points and remains inconclusive until the
other seeds/passes complete. It verifies the new reporting path:

- direct ionic macro cosine `0.0494`, amplitude ratio `0.0110`, and MAE skill
  `-0.00072`;
- hierarchical material interval for ionic MAE skill
  `[-0.00409, 0.00011]`, hence `inconclusive`;
- true-BEC `Z*^T U_pred` cosine `0.0152` and amplitude ratio `0.0778`;
- response-weighted log stiffness bias `+0.1745`; its per-material correlation
  with direct ionic amplitude is `-0.1299`, too weak and single-seed to support
  a hard-spectrum shortcut claim;
- raw normal/direct-`U` head-gradient ratio `11.44`; after registered weights
  it is `1.14`;
- all 4,296 true train optical modes and 100% of recorded normal residual energy
  fall in `|lambda| >= 10 delta`. The requested four bins therefore expose that
  the absolute `delta` is far below this cohort's entire train spectrum, but do
  not yet distinguish relative hardness within the last bin. The loss remains
  unchanged until the registered replay is complete.

The physical macro TRS is `-0.00405`; the matched direct-total control is
`-0.00274`. Their paired interval crosses zero. This is only the declared
negative control and says nothing about ionic factorization.
