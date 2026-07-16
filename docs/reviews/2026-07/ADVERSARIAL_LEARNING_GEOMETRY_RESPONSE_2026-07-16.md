# Response to the learning-geometry audit

## Scope and evidence boundary

The audit correctly separates physical validity from statistical learnability.
The source-convention closure is already strong: true DFPT factors propagated
through the declared regularized optical operator reconstruct source ionic
response to component MAE `0.00452 C/m2` and component-micro cosine `0.99997`.
That supports the BEC-axis transform, printed-Lambda sign, Voigt map, units,
and optical solve; it is not evidence that the learned factors generalize.

The published direct-U smoke is one pass and one seed. Its near-zero response
therefore remains an implementation diagnostic, never an architecture verdict.

## Capacity decomposition v1: completed stages 1--2

This registered, same-ID diagnostic is deliberately confined to the declared
strict-train material lists.  It does not load, inspect, select on, or report
the frozen validation/test panel.

### Batch invariance and aggregation correction

`batch_invariance_samples8.json` checks BEC, `Phi`, `Lambda`, direct `U`,
true-BEC ionic response, and optical `Phi` action under single-graph versus
batched evaluation, graph permutation, graph duplication, and batch sizes
2/4/8.  The maximum forward relative discrepancy is `9.01e-6`; batch loss
minus the average singleton loss is `5.96e-8`; and every requested
parameter-group gradient agrees with its singleton mean to `1.20e-5` or
better. Capacity mode is `eval`, with augmentation and dropout disabled.

The initial version of this audit exposed a real aggregation error:
`born_loss` reduced selected atoms across a batch, giving larger crystals more
loss and gradient weight.  The maintained loss now computes an atom-resolved
invariant loss for each graph and averages graphs.  The regression test
`test_born_batch_invariance.py` prevents a reintroduction.  Historical runs
remain preserved and must be described as having atom-weighted BEC
supervision; they are not overwritten or trajectory-compared to runs using
the corrected aggregation.

### Per-material metric and symmetry-floor audit

`per_material_samples8.json` and `per_material_samples32.json` contain every
material's target/prediction norm, Frobenius error, relative error, cosine,
amplitude ratio, active-target flag, atom/edge count, crystal system, space
group, DFPT optical stability, and `Phi/Lambda/U` scales.  The corresponding
summary explicitly keeps all rows and reports a separate active-target panel;
it never treats cosine as sufficient when a response is nearly zero.

For each of `Z*`, `Phi`, `Lambda`, and direct `U`, the audit constructs the
full Cartesian space-group plus atom-permutation Reynolds average.  It records
the source target residual, learned prediction residual, and the maximum
possible cosine for an invariant prediction, `||P_G y||/||y||`.  The target
ceilings are essentially one (the mean target residual for `Phi` and `Lambda`
is about `1e-7` and `1e-8`, respectively), so source symmetry noise is not a
credible explanation of the same-ID capacity failure.  Conversely, the
learned direct-U Reynolds residual is substantial: mean `0.401` on samples8
and `0.556` on samples32.  This is a diagnostic of the current finite graph
and equivariant representation under exact crystal automorphisms; it is not a
license to post-project predictions or add a production fallback.  It must be
separated from the upcoming K/S oracle before attributing failure to K/S
coupling.

## Hypotheses and decisions

| Audit hypothesis | Assessment | Controlled action | Decision boundary |
|---|---|---|---|
| Bilinear `Z*^T U` and homogeneous normal equation admit a zero basin | Accepted as a concrete optimization risk | Teacher-forced U curriculum; normal-equation warmup/ramp | First run one/two-material noninductive memorization probes, then compare validation-selected formula-disjoint runs. |
| 149 strict complete labels cannot alone establish broad formula-disjoint generalization | Accepted | Do not claim success from memorization; retain frozen val10/test20 | Expand train-only strict data only after a preregistered validation improvement. |
| Low-mode subspace/action rather than Hessian component MAE controls ionic response | Plausible and already diagnosed, not yet causal | Next factor experiment uses a response-operator action objective after the capacity probe | Do not add an eigenspace loss until its gauge/degeneracy behavior is unit-tested. |
| Bond-additive energy class is too narrow | Plausible second-order hypothesis | Require an oracle representability projection before adding a wider energy class | Historical cross-bond ablations alone do not justify promotion. |
| Explicit long-range electrostatics is missing | Plausible research direction | Literature and implementation investigation only after the zero-basin/capacity result | No unvalidated Ewald correction is inserted into production propagation. |
| Fully isolating macro and physical towers wastes 4,961 labels | Real tradeoff | Keep isolation while testing teacher forcing; later compare protected shared representation only with a matched control | Total-only labels must never update physical factor heads or source-branch targets. |

## Implemented change: teacher-forced displacement curriculum

`PiezoJet.predict_displacement_response` now exposes only the shared physical
encoder and independent `U_eta` head. `train.py` adds:

1. direct-factor pretraining (`Z*`, `Phi`, printed/full `Lambda`), already
   maintained before this audit;
2. a new teacher-forced `U_eta` stage: strict rows use the nonzero target
   `D_delta(Phi_true) Lambda_true`, and branch rows use true-BEC ionic
   supervision `Z_true^T U_eta`;
3. an explicit normal-equation schedule. Its weight is zero through a declared
   warmup and then ramps linearly. Zero warmup/ramp reproduces the historical
   objective exactly.

The teacher-forced stage cannot be minimized by jointly shrinking predicted
`Z*`, `Phi`, `Lambda`, and `U_eta`, because those predicted factors are absent
from both curriculum targets. It introduces no new inference-time solve.

## Registered tests

- `outputs/teacher_forced_zero_basin_cpu_smoke_v3/` completed one CPU pass for
  one and two materials. It verifies the factor, U, joint, same-ID evaluator,
  and ledger path, but has **no predictive interpretation**.
- `scripts/run_teacher_forced_zero_basin_probes.ps1` is the full 1/8/32-
  material capacity ladder (100 factor epochs, 100 U epochs, 300 joint epochs,
  normal equation held at zero). It labels its same-ID mode explicitly.
- The historical `run_teacher_forced_after_exposure` executor was removed
  after its retained cohort completed; the archived run waited for the active
  registered exposure replay to exit, then starts that capacity probe without
  GPU contention.

The capacity probe is successful only if all relevant train-sample factors
(`Z*`, cleaned `Phi`, completed `Lambda`, regularized `U_eta`) and the
true-BEC ionic contraction are near exact; it is not enough to reduce a scalar
training loss. If it fails, the next action is an optimization/model-class
diagnosis, not data expansion. If it succeeds, the next action is a matched
three-seed validation comparison of historical immediate-normal training versus
teacher forcing/warmup; test outputs remain unread for selection.

## Completed capacity and model-class diagnosis

The registered 1/8/32 same-ID ladder failed its deliberately stringent
capacity criterion. Even on one material, BEC cosine was `0.99665` but force
constant and completed-Lambda cosines were only `0.57018` and `0.49847`;
eight- and 32-material probes were worse. This is not a holdout result, but it
rules out expanding strict data as a response to the current failure.

`outputs/hessian_bond_laplacian_oracle_v1/train149.json` then projects each
strict-train DFPT Hessian onto a *more expressive* fixed-graph bond-Laplacian
class with an unrestricted symmetric 3x3 stiffness for every undirected graph
edge. It is an offline float64 least-squares oracle, never a prediction path,
loss, or Lambda lift. Across 149 training materials it has mean relative
Frobenius residual `0.03455`, median `0.01505`, p90 `0.09446`, and explained
Frobenius fraction mean `0.99685` (median `0.99977`). Thus graph connectivity
plus a symmetric edge-K Laplacian is broadly capable of representing the
observed Hessians; replacing it wholesale is not currently justified.

The residual is nevertheless correlated with the target's non-symmetric
Cartesian off-diagonal force blocks (Pearson `0.79394`): ten materials exceed
10% residual and one reaches 25.10%. This isolates a real tail limitation of
the present symmetric-K parameterization, but not the global explanation for
the learned near-zero response. The next maintained diagnostic is therefore
true-factor response-operator action supervision for the predicted Hessian,
with a gauge-safe test before any low-eigenspace projector loss.

## Rejected shortcuts

- No Materials Project labels are mixed into the JARVIS-only benchmark.
- No test formula is added to training.
- No hard predicted-spectrum switch, pseudoinverse lift, or hidden shared
  macro-to-factor gradient route is restored.
- No long-range correction is claimed before an audited analytic convention
  and matched ablation exist.
