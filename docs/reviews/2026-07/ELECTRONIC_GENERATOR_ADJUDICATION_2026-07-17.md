# Electronic generator and differential-polarization adjudication

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
4. a mismatch between independently emitted response tensors and a genuine
   differential polarization state.

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
| Shared linearized coefficient generator | 32 | 28 | 0.03942 | 0.99977 | 0.03593 | 0.01933 | 0.99918 |
| Linearized generator + response-jet probes | 32 | 28 | 0.04167 | 0.99973 | 0.03835 | 0.02104 | 0.99907 |
| Literal autodiff differential polarization | 8 | 6 | 0.14986 | 0.99961 | 0.11294 | 0.02105 | 0.99937 |
| Literal autodiff differential polarization | 32 | 28 | 0.09839 | 0.99786 | 0.07766 | 0.04673 | 0.99710 |
| Literal autodiff + response-jet probes | 8 | 6 | 0.13981 | 0.99981 | 0.10529 | 0.02455 | 0.99931 |

The global-irrep result falsifies the claim that the electronic labels are
intrinsically unlearnable on the panel. The current head is specifically short
of high-order/global span. The linearized joint coefficient generator also
shows that BEC and electronic labels can be fit simultaneously. Adding response
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

## Literal nonlinear model

The linearized control is not the requested differential-polarization model.
The implemented nonlinear candidate instead defines

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

`data/processed/strict_train1603_development_folds_v1.json` partitions strict
train1603 into five indivisible reduced-formula folds. Development sizes are
`321/321/320/321/320`, fit sizes are `1282/1282/1283/1282/1283`, and balancing
uses crystal system and train-only GMTNet response-norm strata. Frozen val10 and
test20 labels are not read.

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
development folds contains 2--11 of the samples32 IDs. Therefore the fitted
samples32 checkpoint cannot initialize any fold without label leakage. A
future fold run must use random response parameters or a structure-only
pretrained state; its roughly 40-fold larger fit cohort makes a matched
fixed-epoch run a separate, preregistered compute decision.

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

The historical directory name `p1_differential_capacity` predates the fidelity
correction. Its summary protocol is explicitly renamed
`L1_shared_linearized_coefficient_generator_same_id_capacity`; it must not be
cited as the literal nonlinear P1 model.
