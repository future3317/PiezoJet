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
e^{\mathrm{ion}}_U=\frac{c_e}{\Omega}Z^{*\mathsf T}U_{\eta,\delta},
\qquad U_{\eta,\delta}=\mathcal D_\delta(\Phi)\Lambda.
\]

`U_eta` denotes the production regularized internal-displacement response
coordinate `U_{eta,delta}`. It is produced by an independent atom-level
equivariant head and is translation-free. It is not formed from predicted
`Phi` or `Lambda`, so the
ionic macro loss has no inverse, SVD, detached chart, or straight-through
gradient route. Only on the true stable, well-conditioned stratum is the exact
stationary diagnostic `U_eta_stat = Phi_o^-1 Lambda_o` interpreted as
`du/deta`; an unstable regularized target is not an equilibrium derivative.

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

and the inverse-free coupling constraint

\[
(\Phi^2+\delta^2I)U_\eta=\Phi\Lambda.
\]

The total-only GMTNet target and the physical branch decomposition use
independent towers:

- `tensor`: macro total tower, trained on all GMTNet total labels;
- `physical_tensor`: same-OUTCAR electronic plus direct-`U_eta` ionic tower;
- `factorized_ionic_piezo`: `Z*/Phi/Lambda` diagnostic, never substituted for
  the maintained direct-`U_eta` prediction.

This separation is required because total-only data cannot identify the
electronic/ionic allocation. A macro-total gradient cannot enter the physical
encoder, `Z*`, `Phi`, `Lambda`, `U_eta`, or electronic decoder.

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
- `data/processed/full_corpus_multitask_train1603_v1.json` contains 4,961
  formula-disjoint macro training records and preserves the frozen val10/test20
  IDs. Its strict factor train contains 1,603 records; five additional strict
  records sharing a frozen-panel formula are explicitly excluded.

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
response. The macro curve tests whether 4,961 total-only labels train the
independent total predictor. Because the towers are gradient-isolated, the
macro curve is a negative control/software-isolation check and cannot show that
total-only labels improve ionic factors.

## Current evidence

The first matched direct-operator capacity ladder is retained in
`outputs/operator_learning_capacity_v2/summary.json`. It improves most 1- and
8-material same-ID factors, but fails at 32 materials: force-constant relative
error changes from `0.68049` to `2.17519` and cosine from `0.76921` to
`-0.73438`; factorized ionic cosine is `-0.60515` with amplitude ratio
`0.03024`. This is a negative capacity result, so frozen validation was not
opened. A fresh matched replay isolates the independent-`Lambda` scalar-energy
parameterization and material spectral floor while holding operator weights
fixed.

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
  --splits-file data/processed/full_corpus_multitask_train1603_v1.json `
  --epochs 1 --factor-pretrain-epochs 1 `
  --early-stopping-patience 0 `
  --output-dir outputs/direct_u_multistream_smoke_v1
```

Evaluate the frozen physical panel with:

```powershell
& 'D:\Anaconda\envs\EGNN\python.exe' -m piezojet.evaluate_dfpt `
  --checkpoint outputs/direct_u_multistream_smoke_v1/loss_best.pt `
  --splits-file data/processed/full_corpus_multitask_train1603_v1.json `
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
- `src/piezojet/train_direct_baseline.py`: matched macro-only control;
- `docs/reviews/2026-07/DIRECT_U_IDENTIFIABILITY_CORRECTION_2026-07-15.md`: detailed correction
  report;
- `docs/reviews/2026-07/SECOND_ADVERSARIAL_AUDIT_ADDENDUM_2026-07-16.md`: competitive-hypothesis
  adjudication and falsifiable next actions;
- `docs/reviews/2026-07/ADVERSARIAL_LEARNING_GEOMETRY_RESPONSE_2026-07-16.md`: zero-basin analysis,
  teacher-forced `U_eta` curriculum, and the registered noninductive 1/8/32
  capacity ladder. These same-ID diagnostics are not held-out performance
  experiments.
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
