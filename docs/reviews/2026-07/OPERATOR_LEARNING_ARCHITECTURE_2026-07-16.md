# Direct response-operator learning update

## Scope and status

This update changes supervision, not the declared continuous response
operator or the production fallback policy. The historical inverse/resolvent
action loss remains disabled. New objectives are first evaluated on explicit
same-ID strict-train cohorts of 1, 8, and 32 materials; no frozen test ID is
read and no validation claim is made.

The retained failed numerical audit is
`outputs/capacity_decomposition_v1/operator_learning_batch_invariance_v1.json`.
It showed that an elementwise spectral normalization made the isolated Phi
normal-equation loss `7.77e12` and the Phi-head gradient `1.92e14`. This was
fixed at the objective level rather than hidden by an extremely small weight.
The first implementation is audited in
`operator_learning_batch_invariance_v5.json`. Its matched ladder is retained
at `outputs/operator_learning_capacity_v2/summary.json`: it improves the 1-
and 8-material probes but fails the 32-material gate. At 32 materials, Phi
relative error changes from 0.68049 to 2.17519 and cosine from 0.76921 to
-0.73438. No frozen validation was opened.

That failure exposed two issues now isolated in a fresh replay: a
machine-scale denominator on genuinely soft actions and the unnecessary
constraint `Lambda=B^T K S`. The corrected batch/gradient audit is
`operator_learning_batch_invariance_v6_independent_lambda_spectral_floor.json`.

## One scalar energy without K/S overconstraint

The maintained atom-coordinate energy is assembled explicitly as

```text
E(u, eta) = 0.5 u^T Phi u - u^T Lambda eta + 0.5 eta^T C_aff eta.
```

`Phi` remains a signed periodic edge-stiffness operator. `Lambda` is predicted
by an independent O(3)-equivariant atom head and projected to zero net force.
The scalar assembly gives the required mixed-derivative symmetry without
requiring `Phi` and `Lambda` to share the same edge `K` and strain map `S`.

## Gauge-safe Phi objectives

Let `Phi*` be the cleaned true optical Hessian. Its response-active modes are
the `q=6` true optical eigenvectors with smallest absolute eigenvalues. They
form `Vq`; no predicted eigenvector is selected or matched.

The low-mode direct action objective is

```text
rho( ||(Phi_hat - Phi*) Vq||_F^2 / max(||Phi* Vq||_F^2, eps) ).
```

The leakage objective is

```text
rho( ||(I - Vq Vq^T) Phi_hat Vq||_F^2
     / max(||Phi* Vq||_F^2, eps) ).
```

`rho(s)=sqrt(1+s)-1` is the same robust normalized philosophy used by the
maintained tensor objectives. Degenerate rotations within `span(Vq)` do not
change either loss.

Here `eps` is not machine epsilon. For `q` unit probe columns it is

```text
eps_material = q [0.01 median_nonzero(|lambda_optical(Phi*)|)]^2.
```

The mixed Phi-probe families use the same material-relative floor. A genuine
soft or response-null direction remains informative without allowing division
by roundoff to dominate the batch.

The direct mixed probe objective evaluates `Phi_hat X` and `Phi* X` on three
separately normalized families:

- 40% deterministic random optical displacements;
- 30% true modes with smallest `|lambda|`;
- 30% dominant columns of true `U*=D_delta(Phi*) Lambda*` when strict Lambda
  is available.

Records without strict Lambda still use the first two families; no `U*` is
fabricated. Probe seeds are stable hashes of material ID, atom count, and
probe family, so batch order and batch size cannot change a direction.

## Isolated factor objectives and gradient routes

The BEC oracle uses true `Phi`, true `Lambda`, and therefore true `U*`:

```text
e_Z_oracle = conversion / volume * Z_hat^T U*.
```

Only the shared physical encoder and BEC head receive this gradient. The
existing U oracle already contracts predicted direct `U_hat` with true BEC;
it is reused rather than duplicated.

The Phi oracle uses true `Lambda` and true `U*`:

```text
r = (Phi_hat^2 + delta^2 I) U* - Phi_hat Lambda*.
```

`r` is resolved in true optical modes. Four fixed regions
`|lambda| < delta`, `[delta,3delta)`, `[3delta,10delta)`, and
`[10delta,infinity)` are normalized and averaged equally. A one-percent
within-material spectral floor excludes response-null roundoff directions;
the normalized residual uses `rho`. This term routes only to the Phi
parameterization and shared physical representation.

Random field and strain probes supervise `Z_hat E` and `Lambda_hat eta`.
True probe operands are detached. Lambda probes require strict full Lambda;
observed-only blocks continue to use their direct printed-block loss.

## Multi-response supervision

The same raw DFPT archive exposes `epsilon_ion`. It is now attached as
`y_dfpt_ionic_dielectric` with an availability mask and supervises the
physical self-bilinear branch

```text
epsilon_ion ~ Z^T D_delta(Phi) Z.
```

Strict true factors also generate an explicitly named regularized ionic
elastic softening target

```text
C_soft,delta = conversion / volume * Lambda^T D_delta(Phi) Lambda.
```

Its loss is evaluated in the full Cartesian fourth-rank tensor convention,
so engineering shear is converted exactly once. It is not called a measured
total stiffness. The external JARVIS elastic-total table remains a separate,
availability-masked macro candidate.

The former GMTNet total dielectric weight on the physical tower is now zero.
Total dielectric is learned by an independent macro head parameterized as
`I + S S^T`, where the symmetric equivariant `S` is assembled from invariant
scalars and graph quadrupoles. Total elastic has another independent macro
head: positive bulk/shear isotropic projectors plus structure-conditioned PSD
outer products of equivariant symmetric strain modes. It is converted to the
engineering-Voigt convention only at the output boundary. Unit tests verify
O(3) covariance, dielectric/elastic positivity, and zero gradients into the
physical encoder and `Z/Phi/Lambda/U` heads.

## Preregistered capacity weighting

Unweighted samples8 gradient routes are stored in the v5 audit. New losses
share, rather than multiply, each direct-factor gradient budget. The four Phi
terms share one force-constant-head budget; the two BEC terms share one BEC
budget; Lambda probe and ionic elastic share one Lambda budget. Exact weights
are persisted in every retained resolved run config.

The independent-Lambda/spectral-floor replay deliberately holds those v5
weights fixed. The v6 route audit is retained for interpretation but is not
used to retune this replay, keeping the parameterization/normalization delta
separate from another action-weight sweep.

The matched ladder failed its 32-material gate. Its fixed-weight injection flag
and launcher were subsequently removed from the maintained trainer rather than
kept as an executable fallback. The immutable outputs and summarizer remain for
audit and never authorize a formula-disjoint validation run.
