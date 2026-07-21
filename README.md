# PiezoJet

PiezoJet is an O(3)-equivariant, atom-coordinate model for crystal linear
response. The maintained implementation predicts an atom-resolved internal
displacement response and contracts it with Born effective charges; it does
not infer a full internal-strain tensor from a macroscopic tensor through a
pseudoinverse.

The project is currently an auditable physical-data implementation and
diagnosis, not a state-of-the-art accuracy claim.

## Maintained model

The physical ionic prediction is

\[
\widehat e^{\mathrm{ion}}_U
=\frac{c_e}{\Omega}\widehat Z^{*\mathsf T}\widehat U_{\eta,\delta}.
\]

`U_eta` denotes the production regularized internal-displacement response
coordinate `U_{eta,delta}`. It is produced by an independent, translation-free
global equivariant tower.  In addition to local scalar/vector/quadrupole
features and reciprocal context, its readout contains an explicit symmetric
trace-free octupole (`l=3`) channel and all-to-all crystal conditioning. It is
not formed from predicted
`Phi` or `Lambda`, so the
ionic macro loss has no inverse, SVD, detached chart, or straight-through
gradient route. Only on the true stable, well-conditioned stratum is the exact
stationary diagnostic `U_eta_stat = Phi_o^-1 Lambda_o` interpreted as
`du/deta`; an unstable regularized target is not an equilibrium derivative.
For a strict DFPT label, rather than an identity of the predicted heads, the
teacher target is
\(U_{\eta,\delta}^{\star}=\mathcal D_\delta(\Phi^{\star})\Lambda^{\star}\).

The model also predicts `Z*`, `Phi`, and `Lambda` as physical factors. Their
separate diagnostic response is

\[
e^{\mathrm{ion}}_{\Phi\Lambda}
=\frac{c_e}{\Omega}Z^{*\mathsf T}\mathcal D_\delta(\Phi)\Lambda,
\qquad
\mathcal D_\delta(\Phi)=\Phi(\Phi^2+\delta^2I)^{-1}.
\]

Training and default factor diagnostics use one continuous signed regularized
operator with `delta = 1e-3 eV/Angstrom^2`. Exact stationary propagation is an
explicit true-DFPT stable-stratum diagnostic only. There is no predicted-
spectrum `auto` switch.

`Phi` and `Lambda` are independent coefficients of one explicit scalar
atom-coordinate energy,

\[
\mathcal E(u,\eta)=\tfrac12u^\mathsf T\Phi u
-u^\mathsf T\Lambda\eta+\tfrac12\eta^\mathsf TC\eta.
\]

This preserves the mixed-derivative/Maxwell relation while deliberately not
imposing the stronger model-class restriction `Lambda = B^T K S`. `Phi` is
assembled from signed periodic edge stiffnesses; `Lambda` comes from an
independent O(3)-equivariant atom head followed by the acoustic projection.

Strict-complete records provide

\[
U_\eta^\star=\mathcal D_\delta(\Phi)\Lambda
\]

and the first-order real block constraint

\[
\Phi U_\eta-\delta V_\eta=\Lambda,
\qquad
\Phi V_\eta+\delta U_\eta=0.
\]

The auxiliary `V_eta` exists only during training.  This block system is
equivalent to `(Phi + i delta I)(U + i V) = Lambda` and avoids the historical
normal equation's squared condition number and hard-mode weighting.  Teacher-
`U` AdamW moments are preserved when entering joint training; the isolated
displacement tower uses its own `5e-4` joint learning rate.

The total-only GMTNet target and the physical branch decomposition use
independent towers:

- `tensor`: macro total tower, trained on all GMTNet total labels;
- `physical_tensor`: same-OUTCAR electronic plus direct-`U_eta` ionic tower;
- `factorized_ionic_piezo`: `Z*/Phi/Lambda` diagnostic, never substituted for
  the maintained direct-`U_eta` prediction.

This separation is required because total-only data cannot identify the
electronic/ionic allocation. A macro-total gradient cannot enter the physical
encoder, `Z*`, `Phi`, `Lambda`, `U_eta`, or electronic decoder.

There is deliberately no production API that packages direct-`U_eta` piezo
with factor-derived elastic/dielectric blocks as one energy density. Such a
post-hoc quadratic form would make its own mixed derivatives agree by
construction, but would not establish that the independently predicted
`U_eta` is the stationary response of the factor energy. Energy consistency is
instead an explicit, reported first-order residual on strict records.

Same-OUTCAR electronic and true-BEC ionic component labels supervise their
own branches.  Their already-audited identity `total = electronic + ionic` is
logged as `branch_sum` closure but has zero optimization weight: it contains no
new label information and was measured to send a 3.7-times-larger, opposing
gradient into `U_eta` while the electronic head was inaccurate.

Periodic graph truncation retains the complete equal-distance shell at the
neighbor-budget boundary.  It never selects an arbitrary subset of a
degenerate shell, which would break atom-permutation/space-group symmetry in
high-symmetry crystals.

All tensor auxiliary losses form a complete Cartesian Frobenius norm before a
pseudo-Huber reduction. Macro tensors are reduced per material, BECs per atom,
and ragged full `Phi`, `Lambda`, and `U_eta` targets per material.

## Data and conventions

The physical data root is `E:\DATA\PiezoJet`; the repository retains only
lightweight manifests, split definitions, and regenerable local caches. See
[`docs/DATA_CATALOG_E_DATA.md`](docs/DATA_CATALOG_E_DATA.md) for source
coverage, elastic-label policy, resumable raw-DFPT retrieval, and the
strictly source-tagged Materials Project auxiliary table.
[`data/processed/canonical_datasets.json`](data/processed/canonical_datasets.json)
is the single machine-readable role map. Versioned directories are immutable
provenance and are never searched as fallbacks.

On Linux or another host, set the absolute `PIEZOJET_DATA_ROOT` environment
variable (for example `/home/workspace/lrh/DATA/PiezoJet`). The strict loader
then rebases only physical roles beneath the manifest's declared data root;
repository-local split and manifest paths remain in the checkout. This keeps
all dataset payloads and regenerable graph/symmetry caches in the shared DATA
tree without symlinks or copies inside the repository, while the resolved run
configuration still records the exact external paths.

`config.yaml` names only that manifest. All maintained config consumers call
one strict loader, which rejects a simultaneous version-specific dataset path.
The production factor architecture likewise has one accepted identifier,
`independent_quadratic_response`; old architecture names have no alias.

- GMTNet provides 4,998 JARVIS structures and total piezoelectric labels.
- `E:\DATA\PiezoJet\processed\jarvis_dfpt_v9_full_public\` contains 4,995
  parsed schema-4 tensor payloads. Together with three SHA256-indexed raw ZIP
  quarantines, all 4,998 public archives are accounted for.
- `E:\DATA\PiezoJet\processed\jarvis_strain_completion_v10_zero_dimensional_fix\`
  contains 1,638 strict `Lambda` completions from the unchanged gates. The v10
  audit fixes the zero-dimensional invariant-space case; it does not relax a
  threshold.
- `data/processed/full_corpus_multitask_train1595_v2.json` contains 4,944
  reduced-formula-safe macro training records and preserves the frozen
  val10/test20 IDs. Its strict factor train contains 1,595 records; thirteen
  accepted strict records sharing a held-out reduced formula are excluded.
  Val and test themselves share `HNaO`, so this is train-versus-held-out
  separation, not a fully three-way formula-OOD split. The former 4,961/1,603
  split grouped unreduced unit-cell formulas and is historical only.
- The formula-safe full-public electrostatic pool contains 4,939 materials
  with same-archive BEC, `OUTCAR total - ionic` electronic piezo, electronic
  dielectric, and force-constant labels. All 4,995 parsed payloads pass explicit
  finite/shape checks for these four label families. It does not require a completed
  `Lambda`.  Five immutable reduced-formula-disjoint development folds have
  sizes `988/989/987/988/987`; frozen val10/test20 labels are not read.
- `outputs/vnext_identifiability_census_v1/` certifies the symmetry/acoustic
  coefficient-space ranks on all 4,939 development-safe records. Macro ionic
  response alone is full rank for 788, printed blocks for 4,171, and their
  joint map for 4,182, so only 11 records are algebraic joint increments.
  `outputs/strain_completion_v12_joint_identifiable_v1/` then calibrates
  conditioning on strict-train1595 and predicts unused printed blocks. None of
  the 11 passes every independent and source-completeness gate; v12 therefore
  adds zero full-Lambda labels and does not replace the v10 production role.
- The official `jdft_3d-12-12-2022.json` release contains 75,993 unique JIDs
  and is retained as a structure/metadata auxiliary table.  Its overlap with
  the historical GMTNet piezo IDs is 4,937, and some same-JID relaxed cells
  differ.  It therefore neither replaces the 4,998 GMTNet-pinned structures
  nor silently supplies response labels.

The high-quality partial factor pool covers 4,995/4,998 materials, but strict
acceptance is selection-biased: 1,638/4,995 (32.79%) overall and only
40/1,005 (3.98%) for trigonal records. Strict coverage must therefore never be
described as uniform scaling.

The cache preserves VASP source BEC axes and applies one audited transform at
ingestion: source `Z[i,j] = dP_i/du_j` becomes internal coordinate-row
`Z[j,i]`. Printed OUTCAR internal strain is already `dF/deta = Lambda`; it is
not globally sign-flipped. Internal Voigt order is `(xx, yy, zz, yz, xz, xy)`.

OUTCAR ionic and total tensors are independently Reynolds-projected with the
same point group. The electronic target is their difference, so

\[
P_G e^{\mathrm{el}}+P_G e^{\mathrm{ion}}=P_G e^{\mathrm{total}}
\]

to floating-point roundoff. GMTNet total and raw same-ID OUTCAR total agree for
the audited 610-archive convention cohort after one common conversion. Two global-train IDs conflict
only after the GMTNet target projection; their branch macro losses remain
masked. They are not in the frozen strict val/test panel.

Materials Project credentials and labels are not used in this JARVIS-only
benchmark.

## Exposure-matched protocol

One pass is a complete traversal, not a fixed number of optimizer updates.

- factor stage: one DFPT-branch pass plus one strict-only pass;
- teacher-displacement stage: one branch true-BEC ionic pass plus one strict
  direct-`U_eta` pass;
- joint stage: one macro pass, one branch pass, and one strict-only pass;
- matched direct control: the identical macro passes, split, structural
  checkpoint, seed, and validation-loss checkpoint selection.

The registered replay uses 1/5/10/20 passes and seeds 42, 7, and 1729:

```powershell
$env:PYTHONPATH = 'E:\CODE\PiezoJet\src'
& .\scripts\run_exposure_matched_replay.ps1
```

`metrics.csv` records factor/joint macro, branch, and strict effective passes,
examples seen, and unique coverage. `summary.json` additionally records each
stream's optimizer updates and the number of updates containing each label
objective. Test outputs never select a checkpoint or loss weight.

This historical replay contains two deliberately separate questions. The physical curve
tests whether the former 610 branch and 149 strict training labels learn
`U_{eta,delta}`, true-BEC `Z*^T U_{eta,delta}`, and predicted-BEC ionic
response. The legacy macro curve tests whether 4,961 total-only labels train the
independent total predictor. Because the towers are gradient-isolated, the
macro curve is a negative control/software-isolation check and cannot show that
total-only labels improve ionic factors.

## Current evidence

The electronic-branch adjudication first rules out a trainer/control mismatch:
an exact-clone CPU run has bitwise-identical predictions, losses, gradients,
parameters, and AdamW state for ten updates; CUDA differs only at the expected
scatter/roundoff scale.  The maintained Cartesian electronic head then fails
the samples32 same-ID capacity gate (active relative error `0.60814`, cosine
`0.39285`), with the dominant residual in the `l=3` block.  Its read-only
geometric-basis oracle likewise leaves mean `l=3` residual `0.19716` on val10.
An explicit global-irrep `l<=3` control removes this representation floor on
the same train-only samples32 panel (active relative error `0.04492`, cosine
`0.99973`).

Two different notions of a polarization generator are kept distinct.  The
`ElectromechanicalJetHead` directly emits `Z*` and `e_el` and thereby defines
the complete identifiable displacement--strain first-order polarization jet

\[
\Delta P^{(1)}=\frac{c_e}{\Omega}\sum_\kappa Z_\kappa^{*\mathsf T}u_\kappa
+e^{\mathrm{el}}:\eta .
\]

This is a genuine first-order response model, not merely a fidelity control;
it does not claim to be a finite-perturbation polarization state. On samples32 it fits
electronic response to relative error `0.03942` and BEC to `0.01933`.  Adding
stochastic response-jet probes does not improve either endpoint (`0.04167` and
`0.02104`, respectively).  The `NonlinearDifferentialPolarizationTower`
instead evaluates

\[
\Delta P_\theta(x;u,\eta)=P_\theta(T_\eta(x+u_o))-P_\theta(x)
\]

on genuinely perturbed positions, cell, and periodic edge shifts, and obtains
both tensors as Jacobians at zero.  It never assigns an absolute Berry-phase
polarization target.  It has two explicit, non-fallback variants: Cartesian
polarization and reduced polarization
`P0 = det(F) F^-1 P`, with `F = I + eta`. With zero-point Jacobian labels
alone, its higher-order degrees of freedom are not more identifiable than the
first-order jet; finite perturbation/field labels would be required to learn
genuinely additional nonlinear content. Exact-zero, uniform-translation, Jacobian,
engineering-shear finite-difference, O(3), atom-permutation, batch-invariance,
and second-order-training tests pass.  Its completed samples8/200 CUDA gate has
electronic active relative error `0.14986`, cosine `0.99961`, BEC relative error
`0.02105`, BEC cosine `0.99937`, and acoustic leakage `6.1e-7 e`.  The
preregistered samples32/200 gate also passes: electronic active relative error
is `0.09839`, cosine `0.99786`, the `l=3` stabilized relative error is
`0.07766`, and BEC relative error/cosine are `0.04673/0.99710`, with acoustic
leakage `1.13e-6 e`.  The run used two material-weighted 16-material CUDA
microbatches, 4,443 optimizer seconds, and 14.09 GiB peak allocated memory.
These are noninductive model-class results, not validation performance or a
production promotion.  A fresh samples8 response-jet control (weight `0.25`,
three probes) changes electronic active error from `0.14986` to `0.13981` and
`l=3` error from `0.11294` to `0.10529`, but worsens BEC error from `0.02105`
to `0.02455` and adds 5.5% runtime.  The mixed result does not justify adding
the algebraically redundant probe objective to the maintained candidate.

Method selection no longer needs the frozen val10 panel.  The current
electrostatic adjudication uses the 4,939-material five-fold map
`data/processed/electrostatic_development_folds_v2.json`; the corrected
strict1595 folds are used for tasks that require complete `Lambda`. Every development
fold contains some samples32 capacity IDs, so a fitted same-ID checkpoint is
forbidden as a fold initializer. The completed Stage-A comparison covered
A0, A1, and the retained zero-gated A1.5 control. The next controlled
comparison adds A0-PM and A1.6 without changing the tensor loss or backbone.
Every encoder uses a fold-train-only structure checkpoint, random response
heads, identical stochastic updates, and a fixed response-active gradient
audit batch. It supervises the complete available first-order electrostatic
coefficient set: BEC, electronic piezo, and electronic dielectric. A0 has three
independent towers, A1 hard-shares the response trunk, and historical A1.5 has
three exactly-zero-gated task adapters. A0-PM narrows all three independent
e3nn encoders to a measured capacity match; A1.6 shares the chemistry/geometry
encoder but splits into charge--screening and polar--strain response trunks,
followed by nonzero per-irrep task adapters. A random-initialized N=100 A1 pilot selected update 25 but
collapsed across formulas (electronic active relative error `0.99826`, cosine
`0.06239`, amplitude `0.00406`; BEC relative error about `0.99616`). This is a
negative pilot, not the structure-pretrained adjudication. A read-only audit on
four fixed response-active, norm-stratified train materials corrects the
misleading first-batch gradient ratio: electronic/BEC all-parameter norms are
`0.04598/0.03224`, shared-parameter norms are `0.04547/0.03217`, and shared
cosine is `-0.01883`. Thus there is no evidence for a 5,700-fold global task
scale imbalance; the active-panel gradients are comparable and nearly
orthogonal/slightly conflicting. The matched A0
pilot was explicitly interrupted at the user's request and has no result.
The first structure-pretrained N=200 A1 attempt under
`outputs/vnext_stage3_electrostatic_adjudication_v1/` completed 125 updates,
but its checkpoint selection is invalid: one exactly zero BEC target entered a
raw relative-error denominator of `1e-30`. The training objective was already
finite, so this is not a negative architecture result; only update 50 was
retained and update 125 cannot be reconstructed. The corrected metric uses the
same `0.1 e/component` floor for BEC training, reporting, and selection, while
retaining raw relative error only for audit.

The corrected fold-0/seed-42 N=200 A1 run under
`outputs/vnext_stage3_guardrailed_adjudication_v3/` retains complete
development evaluations at updates 100, 200, 300, and 400. The stabilized
selection score is `2.19007/2.15504/2.20411/2.28188`; update 200 is selected.
At that checkpoint, electronic active relative error/cosine/amplitude are
`0.97990/0.21018/0.32571`, nonzero-BEC cosine and stabilized relative error are
`0.43899/0.93105`, and dielectric stabilized relative error is `0.65432`.
Training loss continued to fall while the development score worsened at 300
and 400, so the user-stopped update-400 state is preserved as single-seed
development-overfitting/multitask-tradeoff evidence. It is neither a completed
500-update result nor a production promotion, and frozen validation10/test20
remain unread.

The first replacement protocol under
`outputs/vnext_stage3_corrected_adjudication_v2/` is now immutable interrupted
evidence: its physical-batch-16 structural pretraining was stopped before the
first epoch checkpoint after reaching approximately 16,048/16,380 MiB, and it
has no performance result. Its nested balanced N=200 and N=800 response
manifests remain valid in
`data/processed/electrostatic_balanced_subsets_v1/`; both cover all 85
fold-train elements and contain 200/800 unique reduced formulas. Structural
pretraining uses all 3,951 fold-train structures, independently of the smaller
response-supervised panel, and excludes every development formula.

The fresh frozen protocol is
`outputs/vnext_stage3_guardrailed_adjudication_v3/`. Structural pretraining
uses gradient accumulation to logical batch 32; every
exposure epoch records 3,951 structures, zero response labels, 124 AdamW
updates (123 batches of 32 plus a 15-material tail), and zero development-
formula overlap. A current-commit physical-batch-4 resource attempt was stopped
before its first epoch checkpoint after observed device use reached
21,993/24,564 MiB. It has no performance result and cannot initialize a model.
The physical-batch-2 replacement
`stage_a_full_fold0_seed42_pretrain_cc13d51_attempt2` kept the identical
logical-batch objective and reached four complete epochs before it was stopped
as a redundant recomputation. The earlier complete 20-epoch checkpoint
`stage_a_full_fold0_seed42_pretrain/best_encoder.pt` uses the same 3,951 IDs,
data hashes, graph/encoder configuration, objective, seed, and logical batch;
its source commit is `27d5617473d6f94858faee93afd503b07e62cad3`.
Cross-commit pretraining reuse is accepted only after those semantic checks and
strict state-dict loading; the pretraining-source and downstream-response
commits are recorded separately rather than relabeling checkpoint provenance.
The completed historical response training retained logical batch 32, microbatch 16,
evaluation batch 32, and `num_workers=0`. A0 uses three independent AdamW optimizers and
advances its parameter-disjoint towers in common-update blocks, with only one
tower and optimizer state CUDA-resident. Every tower receives the identical
deterministic material schedule and selection occurs only at common update
numbers; a CPU regression test also compares sequential backward with the
summed objective within floating-point tolerance. Shared candidates persist exact
run-local model/AdamW/schedule/provenance checkpoints at every full-development
evaluation, together with complete train/development metrics. Microbatching preserves the same material-mean objective but is not
claimed to be bitwise AdamW-identical to a larger forward. Every run binds the
code commit, canonical-data hash, fold/subset hashes, graph-cache schema 5, and
`electrostatic_stabilized_v2` metric version. The three stabilized errors rank
only checkpoints with positive active electronic cosine, positive nonzero-BEC
cosine, and active electronic amplitude ratio at least 0.05; a failed
guardrail cannot be hidden by a low summed score. Exact-zero BEC prediction
norm remains a separate absolute leakage field. Reports include parameter
count, counted FLOPs per logical update, optimizer GPU seconds, and peak CUDA
allocation.

Execution was explicitly resumed on 2026-07-21. No directory was overwritten or
resumed after a resource interruption or redundant recomputation. The matched
N=800/fold-0/seed-42 development comparison is now informative. Complete A0
and A1 runs both select update 500: their stabilized three-task scores are
`1.66731` and `1.77987`, respectively. A0 improves electronic/BEC stabilized
errors (`0.48502/0.69716` versus `0.50951/0.80633`) and their directional
metrics, while A1 is slightly better on electronic dielectric (`0.46403`
versus `0.48514`). A1.5 tracks A1 through update 350 (`1.89298` versus A1
`1.88757` at the same update) and was explicitly stopped as sufficient partial
negative evidence; it is not reported as a complete 500-update result. Frozen
validation10/test20 remain unread.

This comparison diagnoses a real sharing/optimization problem but is not a
capacity-matched final ranking: A0 has `19,295,132` parameters and A1 has
`6,454,490`. At A1 initialization, shared electronic--BEC,
electronic--dielectric, and BEC--dielectric gradient cosines are
`-0.0272/-0.5822/+0.2167`; at the selected checkpoint they are near zero.
A1.5 also has a specific optimization confound: its scalar residual gates start
at exactly zero, which initially blocks gradients to the adapter internals. At
update 350, `tanh(scale)` is only `0.00607/-0.00113` for the electronic and
dielectric adapters and `-0.10505` for BEC. The partial result therefore rejects
this zero-gated A1.5 implementation under the fixed budget, not all soft-sharing
architectures.

The implemented fairness repair is explicit rather than heuristic. With the
registered full configuration, width multiplier `0.56` gives A0-PM
`6,358,299` trainable parameters, versus `6,454,490` for A1 and `6,673,790`
for A1.6. A0-PM cannot load the old full-width initializer: the planner creates
or requires a separate checkpoint with the same fold-train IDs, objective,
seed, and provenance but the exact narrow encoder layout. A1.6 uses the full
hidden O(3) representation in both response trunks; it does not discard odd
covariants merely because BEC and dielectric outputs are inversion-even.
Its `TrainableIrrepAdapter` applies per-irrep-block RMS normalization and one
positive invariant amplitude per multiplicity channel, initialized at `0.075`,
with small nonzero equivariant mixing. Regression tests verify O(3)
equivariance and nonzero first-step gradients for the amplitude, mixing, and
context routes. A1.5 remains byte-for-byte behaviorally available as the
zero-gate negative control and is excluded from the default next comparison.

This is a preregistered model-class upgrade, not a performance claim. The next
minimal experiment keeps the present e3nn `l<=3` backbone and three-task loss
fixed while comparing A0-full, A0-PM, A1, and A1.6. Cartesian many-body/MACE,
scale--shape losses, BEC-first curriculum, long-range electrostatics, and
Gaunt kernels are separate later hypotheses; combining them before this
sharing/capacity adjudication would make the result uninterpretable.

The data interface is not the leading explanation: all three models see the
same balanced 800 unique formulas, same-archive labels, fold-train-only
initializer, deterministic schedule, and stabilized metric. Exact-clone,
microbatch-gradient, tensor, and provenance checks pass. Data scale and
formula extrapolation nevertheless remain important: A0's selected train/dev
scores are `1.20774/1.66731`, its development score is still improving at
update 500, and even its train active electronic relative error is `0.88098`.
Thus the current evidence is best described as task-interference plus
undertraining/sample-efficiency and OOD generalization, not corrupted DFPT
labels or a confirmed tensor-convention bug. A parameter-matched A0 control is
the next fairness check before a full-fold promotion; the earlier N=100
plan at `outputs/electromechanical_jet_fold_adjudication_v2/` remains a
non-executed protocol record. The frozen v3 protocol will bind every selected
checkpoint to the complete code/data/graph provenance. The similarly named
plan under `electromechanical_jet_fold_adjudication/` is a superseded A0--A3
historical plan and is not executable by the maintained runner.

The explicit global-`l=3` displacement head resolves the former same-ID
representation bottleneck.  On the preregistered samples32 capacity panel, a
200-epoch no-consistency fit reaches `U` relative error `0.15827`, cosine
`0.95703`, active true-BEC ionic cosine `0.99829`, and amplitude ratio
`1.01418`.  A corrected readout-basis oracle reduces the superseded global
head's mean/worst minimum residual `0.09498/0.91137` to
`0.00366/0.05945` with the explicit STF octupole; its mean maximum cosine is
`0.99989`.  An unrestricted translation-free lookup has worst residual
`8.64e-9`.  These are train-only capacity results, not generalization evidence.

A post-freeze legacy train1603/val10 adjudication then localized the joint
degradation.  Direct-`U` versus true-BEC ionic gradients are aligned
(`+0.549` cosine), whereas the redundant branch-sum gradient has norm `0.3002`
versus direct-`U` `0.0817` and cosine `-0.556`.  Removing that redundant
objective improves the validation-selected epoch from the prior isolated-U
run's loss/direct-`U`/ionic/TRS of `1.54465 / 0.37147 / 0.29212 / 0.05800` to
`0.97701 / 0.24816 / 0.13628 / 0.38696` for seed 42.  The completed seeds
42/7/1729 give validation-selected total TRS `0.29165 +/- 0.08272`, direct-`U`
loss `0.25054 +/- 0.00485`, ionic loss `0.13820 +/- 0.01368`, and electronic
loss `0.29781 +/- 0.00049` (mean +/- sample SD).  Thus the zero-amplitude U
collapse is reproducibly removed, while the electronic branch remains flat.
The old split used unreduced unit-cell formulas and contains reduced-formula
leakage, so these values are retained as mechanism diagnostics rather than
reduced-formula-OOD evidence.

The matched direct-total validation control reaches TRS
`0.37382 +/- 0.08634`; the paired physical-model macro tower minus direct
difference is negative for every seed and averages `-0.08217 +/- 0.03399`.
The present result therefore supports the global-`l=3` physical mechanism but
does **not** support total-tensor superiority over a matched direct regressor.
All of these comparisons use the legacy val10 panel with reduced-formula
leakage from train; frozen test20 remains unread. See
`outputs/global_l3_no_redundant_sum_multiseed_v1/report/validation_report.md`.

The maintained CUDA path batches nonlocal global-`l=3` attention and the
`Z*^T U` contraction, and omits inactive macro/optical diagnostics during
branch or strict training. Forward/gradient oracle tests and the 181-test CPU
suite pass; `num_workers` remains zero. Concurrent microbenchmarks are
recorded in `docs/reviews/2026-07/GPU_VECTORIZATION_AUDIT_2026-07-17.md` and
are not promoted as clean end-to-end throughput claims.

The first matched direct-operator capacity ladder is retained in
`outputs/operator_learning_capacity_v2/summary.json`. It improves most 1- and
8-material same-ID factors, but fails at 32 materials: force-constant relative
error changes from `0.68049` to `2.17519` and cosine from `0.76921` to
`-0.73438`; factorized ionic cosine is `-0.60515` with amplitude ratio
`0.03024`. This is a negative capacity result, so frozen validation was not
opened. A fresh matched replay isolates the independent-`Lambda` scalar-energy
parameterization and material spectral floor while holding operator weights
fixed. That replay also failed to justify promotion. The historical fixed-v5
weight bundle, its auxiliary-loss module and capacity executor, and its
launchers have therefore been removed from the maintained trainer; the
registered summaries and read-only summarizer remain immutable negative
evidence.

The end-to-end one-pass smoke is in
`outputs/direct_u_multistream_smoke_v1/`. It completed on the frozen test20 and
produced:

- total TRS: `-0.00405`;
- direct-`U_eta` ionic material-macro cosine: `-0.01505`;
- direct-`U_eta` ionic amplitude ratio: `0.00537`;
- factorized `Phi/Lambda` ionic macro cosine: `-0.04038`.

These are implementation-smoke numbers, not a performance estimate. They show
that one pass remains close to zero-amplitude prediction.

The schema-6 adversarial diagnostics further show that all 480 true and all
480 predicted optical modes on test20 lie at `|lambda| >= 3 delta`, so the
current checkpoint did not collapse by predicting modes inside the soft
regularization window. A rank-4 true-`U_eta` SVD oracle retains `95.98%` of
displacement singular energy but still has `19.81%` true-BEC response error;
rank at most six is a six-strain-RHS matrix fact, not evidence for six physical
phonon modes. With true BEC, predicted `Z*^T U_eta` has mean cosine `0.1153`
and amplitude ratio `0.0563`, directly exposing response-active alignment and
scale error in this one-pass smoke.

The strict substitution grid is more diagnostic. With true `Z*`, `Phi`, and
`Lambda`, the declared regularized operator reproduces the source ionic target
with component MAE `0.00452 C/m^2` and component-micro cosine `0.99997`.
Replacing true factors with one-pass predictions degrades direction and
amplitude. The remaining bottleneck is learned factor/displacement quality,
not a failure of the source closure or an algebraic lift.

Earlier fixed-update and protocol A--G results are historical optimization
forensics under older code/data conventions. Their persisted reports remain
under `outputs/`, but their executable training branches have been removed
from the maintained package. They must not be pooled with the direct-`U_eta`
replay.

## Validation

Use the required EGNN environment:

```powershell
$env:PYTHONPATH = 'E:\CODE\PiezoJet\src'
& 'D:\Anaconda\envs\EGNN\python.exe' -m pytest -q
```

The optical-operator audit checks:

- equality to the explicit eigendecomposition;
- invariance to an arbitrary optical-basis rotation;
- finite output at repeated and zero modes;
- complex-solve VJP agreement with central finite differences;
- exact branch-label closure after common Reynolds projection;
- rotation invariance of full Cartesian robust losses;
- absence of a total-only gradient route into physical factors.

Run a bounded full-corpus smoke with:

```powershell
& 'D:\Anaconda\envs\EGNN\python.exe' -m piezojet.train `
  --config config.yaml `
  --splits-file data/processed/full_corpus_multitask_train1595_v2.json `
  --epochs 1 --factor-pretrain-epochs 1 `
  --early-stopping-patience 0 `
  --output-dir outputs/direct_u_multistream_smoke_v1
```

Evaluate the frozen physical panel with:

```powershell
& 'D:\Anaconda\envs\EGNN\python.exe' -m piezojet.evaluate_dfpt `
  --checkpoint outputs/direct_u_multistream_smoke_v1/loss_best.pt `
  --splits-file data/processed/full_corpus_multitask_train1595_v2.json `
  --split test `
  --output outputs/direct_u_multistream_smoke_v1/dfpt_test.json
```

## Main files

- `src/piezojet/model.py`: equivariant encoders, factor heads, direct `U_eta`
  head, and optical operator;
- `src/piezojet/train.py`: invariant losses and exposure-matched streams;
- `src/piezojet/data.py`: source conversion, projection, masks, and caching;
- `src/piezojet/evaluate_dfpt.py`: physical units, strict substitution grid,
  stability strata, spectra, delta sensitivity, low-rank oracle, and
  response-active projector/cross-covariance diagnostics;
- `src/piezojet/electronic_capacity.py`: current-head, global-irrep,
  linearized-coefficient, and literal-autodiff same-ID model-class probes;
- `src/piezojet/train_direct_baseline.py`: matched macro-only control;
- `docs/reviews/2026-07/DIRECT_U_IDENTIFIABILITY_CORRECTION_2026-07-15.md`: detailed correction
  report;
- `docs/reviews/2026-07/SECOND_ADVERSARIAL_AUDIT_ADDENDUM_2026-07-16.md`: competitive-hypothesis
  adjudication and falsifiable next actions;
- `docs/reviews/2026-07/ADVERSARIAL_LEARNING_GEOMETRY_RESPONSE_2026-07-16.md`: zero-basin analysis,
  teacher-forced `U_eta` curriculum, and the registered noninductive 1/8/32
  capacity ladder. These same-ID diagnostics are not held-out performance
  experiments.
- `docs/reviews/2026-07/ELECTRONIC_GENERATOR_ADJUDICATION_2026-07-17.md`:
  exact-clone control, electronic irrep/basis forensics, linearized controls,
  literal nonlinear differential-polarization implementation, capacity gates,
  and development-fold boundary;
- `docs/reviews/2026-07/PREDICTIVE_VALIDITY_REPLAY_PROTOCOL_2026-07-16.md`: frozen separation of the
  physical and macro experiments, conditioning diagnostics, and statistical
  decision rules;
- `docs/reviews/2026-07/WARNING_INVENTORY_2026-07-16.md`: exact known-warning allowlist and
  fail-on-new-warning policy;
- `docs/reviews/2026-07/MAINTAINED_SURFACE_CLEANUP_2026-07-16.md`: canonical role map and the list
  of removed executable fallbacks whose artifacts remain archived;
- `EXPERIMENT_REGISTRY.md`: human-readable experiment ledger covering every
  top-level cohort, including negative, failed, interrupted, partial, running,
  and historical work;
- `outputs/EXPERIMENT_REGISTRY.json`: machine-readable cohort and subrun
  registry with convention/comparability boundaries;
- `outputs/EXPERIMENT_ARTIFACT_INDEX.jsonl`: file-level inventory with
  path/size/time and SHA-256 for lightweight result/configuration records;
- `E:\PAPER\piezojet_equivariant_response_jets\piezojet.tex`: paper source.

Regenerate and validate the ledger after any experiment changes state:

```powershell
$env:PYTHONPATH = 'E:\CODE\PiezoJet\src'
& 'D:\Anaconda\envs\EGNN\python.exe' -m piezojet.experiment_registry
& 'D:\Anaconda\envs\EGNN\python.exe' -m piezojet.experiment_registry --check
```

The registry is an index, not permission to pool results. In particular,
pre-v7 convention variants, removed pInv/ridge observable lifts, historical
protocol A--G runs, and historical v7 direct-`U_eta` replays remain separate
comparability groups.
