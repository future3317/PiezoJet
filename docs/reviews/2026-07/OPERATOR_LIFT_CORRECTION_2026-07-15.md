> **Superseded historical snapshot.** The pInv/ridge observable-lift chart
> described below was subsequently falsified and removed from the maintained
> model and trainer. It remains only as an audit trail. The current ionic path
> is the independent translation-free `Z*^T U_{eta,delta}` response documented
> in `AGENTS.md` and `DIRECT_U_IDENTIFIABILITY_CORRECTION_2026-07-15.md`.

# Operator and observable-lift correction — 2026-07-15

## Policy now implemented

- Training and default inference use only the continuous signed regularized
  optical operator `D_delta` with `delta = 1e-3 eV/Angstrom^2`.
- The exact stationary operator `O_0` is a true-DFPT-stable-stratum diagnostic
  only. No default path branches on a predicted minimum optical eigenvalue.
- The observable ionic lift uses `torch.linalg.pinv` with relative cutoff
  `1e-6`: `Lambda_active = A^+ y` and
  `Lambda_null = (I - A^+ A) R_null`. The legacy ridge lift remains an explicit
  `ridge_legacy` compatibility option only.
- The physical response map `A = Z*^T D_delta(Phi)` remains attached, but the
  pInv lift chart uses `sg(A)`: `Lambda_active=sg(A)^+ y` and
  `Lambda_null=(I-sg(A)^+sg(A))R_null`. A macro ionic target thus cannot send
  an algebraically redundant SVD gradient into the Born-charge or
  force-constant heads. The attached physical factorized response is instead
  calibrated to the direct ionic branch by a separate masked consistency loss.
  The direct-factor response-active loss continues to use true DFPT `Z*` and
  `Phi` to isolate Lambda supervision.

## New audits

For each material the model/evaluator records:

`||A Lambda_active - y||_F / (||y||_F + epsilon_lift)` and
`||A Lambda_null||_F / (||Lambda_null||_F + epsilon_null)`.

The null denominator has a scale-aware roundoff floor. This prevents an
otherwise annihilated floating-point null residual from receiving an arbitrary
order-one leakage ratio.

`outputs/observable_lift_geometry_v1/` records the true-factor geometry under
the same regularized operator. 586/610 maps (96.1%) have full row rank; their
active-lift relative residual has median `3.51e-16`, p95 `1.10e-15`, and
maximum `3.32e-14`. Thus the former attached pInv macro route was redundant
for nearly all materials, not an independent factor constraint. Four train149
rank-one materials have a source ionic target outside their true factor-map
image (relative residual 0.049--0.123); the consistency term masks only these
unrealizable constraints. Their direct macro and direct-factor labels remain
supervised.

## Result-status boundary

The historical three-seed 149/10/20 replay used the old predicted-spectrum
`auto` policy and ridge lift. Its reported TRS/ionic values are retained only
as a legacy failure diagnostic. They are not results for the current
regularized/pseudoinverse implementation. `config.yaml` now writes prospective
runs under `outputs/observable_subspace_v4_detached_lift/`, so historical
outputs cannot be overwritten. No post-correction training has been run.

## Verification

The full test suite was rerun after this correction:

```text
106 passed, 403 warnings in 20.32s
```

The warnings are existing TorchScript deprecation/type-annotation and symfc
deprecation notices; there were no test failures.
