# Electromechanical-jet and differential-polarization adjudication

Date: 2026-07-17

This report records post-freeze diagnostics. Unless explicitly stated as
`val10`, all capacity IDs are drawn from strict train1603, are optimized and
evaluated on the same IDs, and cannot support a generalization claim. Frozen
test20 is not read anywhere in this adjudication.

## Question and controls

The observed joint model had a reproducibly useful direct-`U`/ionic branch but
an almost unchanged electronic loss. The investigation separates four possible
causes:

1. trainer/control mismatch;
2. current electronic output-basis restriction;
3. insufficient explicit high-order/global representation;
4. whether a shared first-order response jet or a nonlinear polarization state
   generalizes better under the available labels.

The CPU exact-clone control duplicates the macro/direct model, optimizer, data
order, and ten updates. Prediction, loss, gradient, parameter, and optimizer
state differences are all exactly zero. CUDA shows only scatter/roundoff-scale
differences (maximum prediction difference `1.79e-7`, gradient difference
`1.86e-8`); its deliberately tighter `1e-7` whole-trajectory parameter gate is
not interpreted as a semantic mismatch.

On formula-disjoint val10, the selected physical checkpoints have electronic
macro amplitude only `0.0142--0.0212`, electronic cosine `-0.1644--0.0486`,
and negative electronic MAE skill for all three seeds. The current electronic
geometric-basis oracle has mean minimum stabilized residual `0.17797`; its
mean `l=3` residual is `0.19716`, while both `l=1` copies and `l=2` are nearly
spanned. This identifies a concrete model-class limitation rather than a loss
weight diagnosis alone.

## Same-ID capacity matrix

All rows use seed 42, 200 fixed epochs, no checkpoint selection, RTX 4060 Ti,
and `num_workers=0`. `Active` uses the independent 18-component norm threshold
`0.05*sqrt(18) C/m2`. Near-zero materials remain in stabilized norm metrics but
not in active cosine.

| Candidate | Cohort | Active | Electronic relative error | Electronic cosine | `l=3` relative error | BEC relative error | BEC cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current Cartesian electronic head | 32 | 28 | 0.60814 | 0.39285 | 0.53280 | n/a | n/a |
| Explicit global-irrep `l<=3` | 32 | 28 | 0.04492 | 0.99973 | 0.04014 | n/a | n/a |
| First-order electromechanical jet | 32 | 28 | 0.03942 | 0.99977 | 0.03593 | 0.01933 | 0.99918 |
| First-order jet + redundant probes | 32 | 28 | 0.04167 | 0.99973 | 0.03835 | 0.02104 | 0.99907 |
| Literal autodiff differential polarization | 8 | 6 | 0.14986 | 0.99961 | 0.11294 | 0.02105 | 0.99937 |
| Literal autodiff differential polarization | 32 | 28 | 0.09839 | 0.99786 | 0.07766 | 0.04673 | 0.99710 |
| Literal autodiff + response-jet probes | 8 | 6 | 0.13981 | 0.99981 | 0.10529 | 0.02455 | 0.99931 |

The global-irrep result falsifies the claim that the electronic labels are
intrinsically unlearnable on the panel. The current head is specifically short
of high-order/global span. The displacement--strain first-order
electromechanical jet also shows
that BEC and electronic labels can be fit simultaneously. It is the exact
identifiable map
`Delta P^(1) = c_e/Omega sum_k Z*_k^T u_k + e_el:eta`, not a fidelity fallback
and not a claim about finite-amplitude polarization. Adding response
probes does not improve its fixed endpoint; these probes are an unbiased
stochastic rewriting of already-complete coefficient supervision, not new
label information.

The fresh nonlinear response-jet control uses weight `0.25`, three displacement
and strain probes, the same samples8 IDs/seed/200 fixed epochs, and a random
initialization rather than resuming the no-jet checkpoint. It slightly improves
electronic active error by `0.01005` and `l=3` error by `0.00765`, while BEC
error worsens by `0.00351` and optimizer time rises from 964 to 1,018 seconds.
Both runs pass their same-ID gate, but the probe term does not improve both
Jacobians and contains no new label information. It is retained as a mixed
negative control rather than added to the maintained objective.

## First-order and literal nonlinear models

`ElectromechanicalJetHead` is A1 and directly parameterizes the complete
first-order jet. `IndependentElectrostaticHeads` is A0 and uses two statistically
independent generators, so BEC and electronic losses have no shared parameter
tensors. The implemented nonlinear A2/A3 candidates instead define

```text
Delta P_theta(x; u, eta) = P_theta(T_eta(x + u_o)) - P_theta(x)
u_o = u - graph_mean(u)
```

`T_eta` deforms Cartesian positions, the row-vector cell, and periodic edge
shifts by the same `F=I+eta`; internal fractional coordinates are evaluated in
the undeformed reference cell. The fixed complete-shell edge topology is not
rebuilt inside a derivative. The state network is an O(3)-equivariant polar
vector with explicit `l<=3` periodic message passing, invariant global
attention, and reciprocal scalar context. No absolute polarization or
Berry-phase branch label exists.

A2 differentiates Cartesian polarization. A3 differentiates reduced
polarization `P0=det(F)F^-1 P`, with `F=I+eta`; the variable is explicit and
there is no automatic switch. The radial envelope is `(1-r/r_c)^3_+`, so the
function and its first two derivatives vanish at the cutoff. Under zero-point
BEC/electronic-Jacobian labels alone, A2/A3 contain no additional identifiable
higher-order information beyond A1. Finite displacement/strain/field labels
would be required to establish a nonlinear advantage.

Three reverse-mode calls differentiate the three output components:

```text
Z*[k,a,i] = Omega/c_e * d DeltaP[i] / d u[k,a]
e_el[i,mu] = d DeltaP[i] / d eta_V[mu]
```

Training backpropagates once more through these Jacobians. The reference term
is constant in `(u,eta)`, so coefficient evaluation omits its second network
execution using the exact derivative identity
`d(P(T(x))-P(x))/d(u,eta)=dP(T(x))/d(u,eta)`; the public nonlinear increment
still evaluates the literal difference.

Differentiable reciprocal geometry is ephemeral. Caching it on the module
would retain the previous second-order graph and double memory at the next
step; only fixed, non-differentiable batches use the geometry cache.

## Mathematical and implementation tests

The candidate passes:

- bitwise `Delta P(0,0)=0`;
- exact invariance to uniform displacement and resulting BEC acoustic sum;
- equality between returned BEC/electronic tensors and Jacobians of the public
  nonlinear increment;
- engineering-shear central finite differences;
- finite second-order gradients into trainable parameters;
- O(3) covariance of the nonlinear increment, BEC, and electronic tensor;
- atom-permutation and batch invariance.

The full repository suite passes `173` tests. The samples8 full-batch and 4+4
material-weighted microbatch epoch-1 losses are `1.29530430` and `1.29530424`,
respectively. Large-cohort accumulation therefore retains the cohort-mean loss
and one AdamW step per epoch up to expected floating-point reduction order.

## Development split and decision boundary

`data/processed/electrostatic_development_folds.json` partitions the 4,939
formula-safe full-public electrostatic records into five indivisible
reduced-formula folds. Development sizes are `988/989/987/988/987`. Every
record has BEC, same-OUTCAR electronic piezo, electronic dielectric, and force
constants; strict Lambda is not required. Frozen val10 and test20 labels are
not read. The older strict1603 folds remain only for tasks requiring complete
Lambda.

At the samples8 gate the nonlinear model has genuine same-ID capacity and exact
response semantics, but it is not promoted to production. The fixed samples32
run also passes both electronic and BEC strong gates. Its epoch
25/50/75/100/200 total normalized losses are
`0.76857/0.31812/0.15237/0.08249/0.01446`. The first four electronic losses are
`0.27768/0.17403/0.10190/0.06359`, and BEC losses are
`0.49089/0.14409/0.05047/0.01889`; final component quality is reported by the
physical metrics rather than inferred from the summed loss. The final
electronic active relative error/cosine are `0.09839/0.99786`; BEC relative
error/cosine are `0.04673/0.99710`. Two 16-material, material-weighted
microbatches preserve one cohort-mean AdamW update per epoch. The run takes
4,443 optimizer seconds on an RTX 4060 Ti and reports 14.09 GiB peak allocated
memory.

This is a model-class gate, not a development result. Each of the five
development folds contains some samples32 IDs. Therefore the fitted
samples32 checkpoint cannot initialize any fold without label leakage. A
fold run must use random response heads and the same fold-train-only
structure-pretrained encoder state for A0--A3.

The first formula-disjoint plumbing pilot uses fold0, seed42, fixed
train100/development100 subsets, batch size 4, and 100 stochastic updates. A1
selects update 25 but has electronic active relative error `0.99826`, cosine
`0.06239`, amplitude `0.00406`, and BEC relative error about `0.99616`.
Later checkpoints worsen, so random initialization collapses across formulas
despite same-ID capacity. The gradient audit initially used the first four
training records and was confounded by weak targets; the maintained audit now
selects a fixed response-active, norm-stratified batch and reports all-task as
well as shared-parameter gradient norms. The replacement batch uses
JVASP-55695, JVASP-42957, JVASP-52196, and JVASP-11504, spanning electronic
target norms `0.2432--8.8161 C/m2`. At the selected checkpoint,
electronic/BEC all-parameter gradient norms are `0.04598/0.03224`, shared norms
are `0.04547/0.03217`, and shared cosine is `-0.01883`. Thus the tasks have
comparable active-panel scale and are nearly orthogonal/slightly conflicting;
the old roughly 5,700-fold ratio is not global loss-scale evidence. The matched A0 run was interrupted at
the user's request before producing a checkpoint; its `failure.json` is kept
and it has no performance result.

Training was then explicitly paused. Before the next authorized run, A0 was
changed to evaluate and backpropagate its three
parameter-disjoint towers sequentially inside the same optimizer update. A
parameter-by-parameter CPU regression test confirms equality with backward on
the summed objective, including matching unused-parameter masks. This reduces
peak activation residency from two tower graphs to one without changing the
loss, update count, batching, or comparison contract.

The historical matched Stage-A commands were recorded, but not executed, in
`outputs/electromechanical_jet_fold_adjudication/stage_a_n100_fold0_seed42_plan.json`.
The plan fixes fold0, seed42, train/development limits 100/100, one shared
fold-train-only structure checkpoint, random response heads, fresh output
directories, `num_workers=0`, and the same response-active norm-stratified
gradient audit. It explicitly forbids automatic promotion or dataset expansion
and certifies frozen validation/test labels unread. Preparation also exposed a
schema-drift bug in `pretrain_e3nn`: schema-2 fold manifests intentionally omit
duplicated train lists. The pretrainer now derives the formula-safe train panel
from the global population minus the development subset and has a regression
test for that path.

## 2026-07-21 corrected N=200 and resumed protocol

The stabilized BEC metric was applied consistently to training, reporting, and
selection in the corrected fold-0/seed-42 N=200 A1 run. Four complete
development evaluations are retained:

| update | selection | electronic cosine | electronic amplitude | electronic relative error | BEC cosine | BEC relative error | dielectric relative error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 2.190065 | 0.120911 | 0.121086 | 1.005560 | 0.179049 | 0.984477 | 0.654994 |
| 200 | **2.155039** | 0.210184 | 0.325705 | **0.979902** | 0.438994 | 0.931054 | **0.654315** |
| 300 | 2.204108 | 0.241909 | 0.532308 | 1.027085 | 0.515650 | **0.921804** | 0.669061 |
| 400 | 2.281876 | 0.262935 | 0.597799 | 1.062868 | 0.536327 | 0.976481 | 0.671151 |

Update 200 is selected. Training loss continued to fall while the summed
development score worsened at updates 300 and 400. The user therefore stopped
the run after the complete update-400 evaluation. This is evidence for
development overfitting and a changing three-task tradeoff, not a completed
500-update result, an architecture rejection, or a production promotion.

The trainer now evaluates every 50 updates for the N=800 comparison and uses
guardrail-aware early stopping with four eligible evaluations of patience.
Patience starts only after an eligible checkpoint exists; failed guardrails do
not select a fallback. Every evaluation persists the complete model, AdamW and
schedule state and emits a compact curve containing train/development scores,
generalization gap, physical guardrails, timing, and early-stop state.

Execution was explicitly resumed on 2026-07-21 using code commit
`cc13d5119b62dbbac5c5a27c361cd39d86fb633a`. A physical-batch-4 structural
pretraining attempt was stopped before epoch one when observed CUDA use reached
21,993/24,564 MiB. It wrote no checkpoint and is retained as
`interrupted_resource_guardrail`. A physical-batch-2 replacement kept logical
batch 32 and the same material-mean objective in
`stage_a_full_fold0_seed42_pretrain_cc13d51_attempt2`, but was stopped after four
complete epochs once an audit established that it duplicated the earlier
complete 20-epoch fold-only checkpoint. The reused checkpoint has the same
3,951 IDs, data hashes, graph/encoder configuration, objective, seed, and
logical batch; its source commit is
`27d5617473d6f94858faee93afd503b07e62cad3`. The loader records that source
commit separately from the downstream response-run commit and still rejects
semantic configuration drift, leakage, or a non-strict state-dict load. Frozen
validation10/test20 remain unread.

The literature-linked decision boundary is deliberately conditional. The
near-zero shared electronic/BEC gradient cosine and comparable norms do not
justify PCGrad or GradNorm before A0/A1/A1.5 is adjudicated. If all N=800
models fit training but fail development, the next minimal candidate is
BEC-first response-aware pretraining with replay of the other valid tasks.
Higher-body-order or stronger reciprocal message passing is considered only if
the learning curve or stratified residuals locate a representation/long-range
floor. Scale--shape output is considered only after direction improves while
amplitude remains collapsed.

## Artifacts

- `outputs/electronic_generator_adjudication_v1/e0_exact_clone_cpu/summary.json`
- `outputs/electronic_generator_adjudication_v1/e0_exact_clone/summary.json`
- `outputs/electronic_generator_adjudication_v1/e1_electronic_diagnostics/`
- `outputs/electronic_generator_adjudication_v1/e2_current_head_capacity_optimized/`
- `outputs/electronic_generator_adjudication_v1/e3_global_irrep_capacity/`
- `outputs/electronic_generator_adjudication_v1/p1_differential_capacity/`
- `outputs/electronic_generator_adjudication_v1/j1_differential_jet_capacity/`
- `outputs/electronic_generator_adjudication_v1/p1_literal_autodiff_capacity/`
- `outputs/electronic_generator_adjudication_v1/j1_literal_autodiff_jet_capacity/`
- `outputs/electromechanical_jet_fold_adjudication/pilot_n100_fold0_a1_seed42_retry1/`
- `outputs/electromechanical_jet_fold_adjudication/pilot_n100_fold0_a1_seed42_retry1/active_gradient_audit.json`
- `outputs/electromechanical_jet_fold_adjudication/pilot_n100_fold0_a0_seed42/failure.json`
- `outputs/electromechanical_jet_fold_adjudication/stage_a_n100_fold0_seed42_plan.json`
- `outputs/vnext_stage3_guardrailed_adjudication_v3/stage_a_n200_fold0_a1_electromechanical_jet_seed42/`
- `outputs/vnext_stage3_guardrailed_adjudication_v3/stage_a_full_fold0_seed42_pretrain_cc13d51_attempt1/`
- `outputs/vnext_stage3_guardrailed_adjudication_v3/stage_a_full_fold0_seed42_pretrain_cc13d51_attempt2/`

The historical directory name `p1_differential_capacity` predates the current
terminology. Its stored bytes remain immutable, but the maintained code calls
this model the first-order electromechanical jet; it must not be cited as the
literal nonlinear model.
