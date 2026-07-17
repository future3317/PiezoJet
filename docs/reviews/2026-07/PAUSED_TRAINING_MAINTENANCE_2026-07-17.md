# Paused-training maintenance audit (2026-07-17)

## Boundary

The user paused all PiezoJet training before this maintenance pass. No model
fit, pretraining, evaluation on frozen labels, or GPU experiment was launched.
The only executions were static checks, CPU unit tests, registry refresh, and
paper compilation.

## Correctness decisions

- The response-stratification metric remains the full Cartesian Frobenius norm.
  For the symmetric last two piezo indices it is numerically equal to the norm
  of the orthonormal 18-dimensional irrep coordinates. Counting the two
  Cartesian shear entries is the required metric weight, not a duplicate
  sample; an unweighted engineering-Voigt norm is not $O(3)$ invariant.
- All maintained acoustic projection consumers now call
  `projectors.translation_projector`. The convention audit and force-constant
  loss no longer construct their own translation basis/projector.
- The point-group Reynolds helper is named `symmetry_projection`; the old
  ambiguous `projector` module was removed. The distinct acoustic module stays
  `projectors`.

## Removed executable historical surface

The main trainer no longer exposes the M2.1 / implicit-first-32 shortcuts or
the `--operator-learning-capacity` flag that injected a fixed historical v5
loss bundle. The single-material action and operator-capacity PowerShell
launchers were removed as well. Their registered outputs, run-local resolved
configurations, and read-only summarizer remain preserved as negative evidence.

## Training resumption boundary

The next permissible GPU work remains the registered non-executing Stage-A
A0--A3 plan. It uses fresh output directories and fold-train-only structural
pretraining; it needs a subsequent explicit user request before execution.
