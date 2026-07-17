# PiezoJet agent guide

## Objective and scope

PiezoJet is a physics-constrained, atom-coordinate response model. Preserve
physical units, tensor conventions, and auditable source provenance. Do not
turn an unresolved data/physics issue into a cosmetic architectural claim.

## Runtime and validation

- Use `D:\Anaconda\envs\EGNN\python.exe` for Python, PyTorch, training, and
  tests.
- Set `PYTHONPATH=E:\CODE\PiezoJet\src` before module entry points.
- After code changes, run the relevant tests; use the full suite for shared
  data, tensor, train, or evaluator changes.
- Use `apply_patch` for source and configuration edits. Preserve unrelated
  dirty-worktree changes.
- Treat `data/processed` and `E:\DATA\PiezoJet\processed` as one mapped
  physical store even when Windows reports no `LinkType`. Never recursively
  delete or move either root during repository cleanup.

## Non-negotiable scientific rules

1. Never relax strict internal-strain completion thresholds to increase the
   sample count. A completed Lambda requires symmetry invariance, acoustic
   nullspace, unique identification, redundant-block validation when possible,
   and ionic-response closure.
2. Do not claim No.187 source records are wrong. It is quarantined from strict
   completion pending a source/reference-cell convention resolution; retain
   valid printed partial labels.
3. Do not apply a global Voigt, engineering-shear, or improper-rotation sign
   patch without a controlled reproducer. Internal strain uses the documented
   canonical order `(xx, yy, zz, yz, xz, xy)`.
4. Never expose credentials from `mp.md`, `help.md`, attachments, or local
   configuration.
5. Do not mix Materials Project labels into JARVIS-only benchmarks. Any future
   external-source experiment requires source identity and JARVIS-only test
   reporting.

## Current data protocol

- `data/processed/canonical_datasets.json` is the only maintained role map.
  Code and scripts must not scan older versioned directories as fallbacks.

- The full public archive accounting is complete: 4,995 schema-4 parsed
  payloads are in `E:\DATA\PiezoJet\processed\jarvis_dfpt_v9_full_public`,
  and the three non-labelable raw ZIPs are SHA256-indexed in
  `E:\DATA\PiezoJet\raw\jarvis_dfpt_v9_quarantine`. Together these account
  for all 4,998 requested archives. Never fabricate aligned labels for
  JVASP-1048 (structure mismatch), JVASP-34237, or JVASP-7658 (malformed XML).
- `E:\DATA\PiezoJet\processed\jarvis_strain_completion_v10_zero_dimensional_fix`
  is the current strict cohort: 1,638 accepted of 4,995 audited payloads. Its
  zero-dimensional invariant-space correction treats the unique zero tensor
  as condition number 1 and pseudoinverse norm 0; no acceptance threshold was
  relaxed.
- `data/processed/strict_completion_benchmark_train_v10_full_public.json` is
  the current frozen-panel-safe factor split: train1603/val10/test20. The five
  accepted records sharing frozen validation/test formulas are excluded from
  train. Validation and test IDs remain byte-for-byte frozen.
- `data/processed/full_corpus_multitask_train1603_v1.json` combines the 4,961
  formula-disjoint GMTNet macro train records with the strict train1603 factor
  records. Factor losses remain availability-masked.
- The official 2022 `dft_3d` release at
  `E:\DATA\PiezoJet\jarvis_dft3d_index\jdft_3d-12-12-2022.json.zip` has
  75,993 unique JIDs and is an auxiliary structure/metadata source. Its
  historical GMTNet overlap is 4,937/4,998 and some same-JID relaxed cells
  differ, so it never replaces GMTNet-pinned benchmark structures or DFPT
  response labels. Its integrity/overlap audit is
  `outputs/jarvis_dft3d_official_audit/summary.json`.
- The v10 coverage audit records severe selection bias. In particular strict
  acceptance is 40/1,005 for trigonal records and 0/46 for 1--2 atom records.
  Do not describe strict-complete coverage as uniform or unbiased.
- The v4--v7 cache/completion cohorts below are retained historical convention
  and learning-curve evidence. They are not the current full-public training
  cohort and must not be pooled with v9/v10.

- Historical v4--v7 completion/cache details and their biased acquisition
  cohorts are documented in `docs/EXPERIMENTAL_ARCHIVE.md` and the experiment
  registry, not in the maintained runtime protocol. Their central convention
  result still applies: source `Z[i,j]=dP_i/du_j` becomes internal `Z[j,i]`,
  while printed OUTCAR internal strain is already `dF/deta=Lambda`.
- `outputs/gmtnet_outcar_total_consistency_v1/` records the required total-label
  audit. Raw GMTNet and same-ID OUTCAR totals agree exactly after one common
  conversion for all 610 archives. The Reynolds-projected GMTNet training
  target conflicts with source OUTCAR branch labels only for global-train
  JVASP-42995 and JVASP-28862 under the fixed 0.05 C/m2 and 5% double gate;
  retain their GMTNet total target but mask DFPT macro ionic/electronic/branch
  supervision pending convention resolution. Neither belongs to the frozen
  strict 149/10/20 panel.
- `data/processed/_historical_splits/strict_completion_benchmark_v1.json` is frozen at
  69/10/20 formula-disjoint train/validation/test. Never reassign the 20 test
  IDs as new data arrive.
- `data/processed/_historical_splits/strict_completion_benchmark_train97_v1.json`,
  `data/processed/_historical_splits/strict_completion_benchmark_train108_v1.json`, and
  `data/processed/_historical_splits/strict_completion_benchmark_train149_v1.json` may add only
  strict-complete, frozen-panel-formula-disjoint materials to train. Their
  validation/test IDs must remain byte-for-byte identical to the v1 panel.
- The first nested learning curve uses splits in
  `outputs/strict_learning_curve_v1/splits/`: 19, 23, 35, 50, and 69 complete
  training materials with the same validation/test panel. Its registered
  three-seed anchors (19, 35, 69) are summarized in
  `outputs/strict_learning_curve_v1/report/learning_curve.md`; the 23/50 rows
  are explicitly seed-42 phase-1 diagnostics.
- Every learning-curve point reports full-Lambda, response-active ionic, and
  low-mode metrics. Do not select checkpoints or hyperparameters using test
  outputs.
- `outputs/feedback4_execution_v1/report/feedback4_report.md` is retained as a
  historical fixed-coverage optimization forensic. Its former executable
  protocol and mode-aware branches are not part of the maintained trainer.
- `outputs/feedback5_execution_v1/report_v2/feedback5_report.md` is the
  canonical post-selection replay of A--E. Ionic claims must name the
  regularized operator and material-macro aggregation; component-micro cosine
  is an aggregation audit only. The active ionic panel uses the independent
  18-component threshold `0.05*sqrt(18)` C/m2.
- `outputs/information_gain_cohort_v1/cohort.json` is a dry retrieval ranking,
  not a label cohort. It must not be used to relax strict gates, reassign the
  frozen test panel, or claim model uncertainty unless its optional supplied
  ensemble score was actually computed.
- `outputs/information_gain_cohort_v2_test_crystal_coverage/cohort.json` is a
  second, coverage-constrained retrieval queue, not a label cohort. Its
  `test_crystal_coverage` rule reserves *retrieval* slots only; it must never
  be interpreted as a relaxed completion rule or as accepted-label balance.
- Canonical post-selection B diagnostics for 69, 97, and 108 training
  materials are in `outputs/factor_protected_norm_match_v1/report/`,
  `outputs/strict_completion_train97_protocol_b_v1/report/`, and
  `outputs/strict_completion_train108_protocol_b_v1/report/`, respectively.
  The 108-train coverage-aware addition improves ionic diagnostics but leaves
  total TRS negative, so it is not a production-model promotion.

## Maintained parameterization and training

- The production ionic path is `Z*^T U_{eta,delta}`, where `U_{eta,delta}` is
  the regularized internal-displacement response coordinate predicted by an
  independent translation-free atom-level head. It is not an equilibrium
  `du/deta` on unstable references. `U_eta_stat=Phi_o^-1 Lambda_o` is reserved
  for the true-stable exact diagnostic. Never reconstruct `Lambda`
  from a macroscopic target through a pseudoinverse, ridge lift, active/null
  projector, or detached predicted-factor chart.
- `Phi/Lambda` propagation is an attached physical diagnostic. Strict labels
  supervise `U_eta*=D_delta(Phi)Lambda` and the first-order real block system
  `Phi U-delta V=Lambda`, `Phi V+delta U=0`. The auxiliary `V` is training-only;
  do not reintroduce the squared normal equation.
- `Phi` and `Lambda` are independent coefficients of the same explicit scalar
  quadratic response energy `0.5 u^T Phi u - u^T Lambda eta + 0.5 eta^T C eta`.
  Do not reintroduce the extra restriction `Lambda=B^T K S`; integrability
  does not require sharing the edge stiffness `K`.
- The rejected low-mode/mixed-probe operator auxiliary losses and their
  capacity executor have been removed from the maintained package after the
  failed 32-material gate. Their immutable outputs and read-only summarizer
  remain historical evidence only; do not reintroduce them as fallback losses.
- Total-only GMTNet labels use an independent macro encoder/head. They must not
  backpropagate into the physical encoder, electronic branch, `Z*`, `Phi`,
  `Lambda`, or `U_eta`.
- `PiezoJet.macroscopic_response_density()` is a coefficient-level constitutive
  wrapper for the physical direct-`U` piezo tensor and factor-derived elastic/
  dielectric diagnostics. It must omit the total-only macro tower and must not
  be described as a common microscopic generator or as enforcing
  `U_eta=D_delta(Phi)Lambda`.
- Same-OUTCAR electronic and true-BEC ionic labels supervise their own
  components. `branch_sum` remains a logged closure diagnostic with zero loss
  weight. Its target is algebraically redundant, and the train1603 audit found
  its U-tower gradient norm 0.30024 versus direct-U 0.08167 with cosine -0.55615.
  Do not use this closure residual to make U compensate for electronic error.
- `model_from_config` accepts only `independent_quadratic_response`, isotropic
  background, and the continuous `regularized` operator. Exact propagation is
  an explicit true-DFPT stable diagnostic. There is no `auto` policy.
- Tensor losses reduce complete Cartesian Frobenius norms before pseudo-Huber.
  Do not reintroduce componentwise SmoothL1 for rotated tensor objectives.
- One exposure epoch means complete passes. Factor training traverses branch +
  strict; teacher-U training traverses branch + strict; joint training traverses
  macro + branch + strict. Strict-only losses must not be repeated on the
  branch stream, and all three stages must be counted in exposure metadata.
- `scripts/run_exposure_matched_replay.ps1` registers 1/5/10/20 passes for
  seeds 42/7/1729 and runs a matched macro-only direct control. Checkpoints are
  selected by validation loss only; test outputs never tune the model.
- Intermediate single-seed and one-pass values are implementation diagnostics.
  Three seeds are required before comparing registered exposure points.
- Historical A--G, sketch, mode-aware, pInv/ridge, and architecture-switch
  executors have been removed from the maintained package. Their persisted
  outputs remain historical evidence and must not be presented as current
  code paths.
- The M2.1/implicit-first-32 shortcuts and the fixed-v5 operator-action bundle
  are also removed. Same-ID diagnostics require an explicit material-ID file
  and `--allow-noninductive-overfit`; negative operator artifacts remain
  readable only through their summarizer and registry.

## Current direct-U candidate

- The maintained U tower is isolated from the factor/macro encoders and uses a
  global explicit STF-octupole (`l=3`) readout. Periodic graph construction
  retains the full equal-distance shell at the neighbor-budget boundary.
- `outputs/u_capacity_adjudication_v4_global_l3/` is the positive same-ID
  capacity gate. At samples32/200 epochs without consistency, U relative error
  is 0.15827, U cosine 0.95703, active true-BEC ionic cosine 0.99829, and
  amplitude ratio 1.01418. This is noninductive capacity evidence only.
- `outputs/global_l3_joint_optimizer_adjudication_v1/` is the completed seed42
  train1603/val10 loss-geometry adjudication. Removing the redundant branch-sum
  objective yields validation-selected epoch 7 with loss 0.97701, direct-U
  loss 0.24816, ionic loss 0.13628, and total TRS 0.38696. It is a post-freeze
  single-seed validation diagnostic, not a test result or production claim.
- `outputs/global_l3_no_redundant_sum_multiseed_v1/` is the completed
  seeds42/7/1729 validation-only replication. Validation-selected mean/sample
  SD are: total TRS 0.29165/0.08272, direct-U loss 0.25054/0.00485, ionic loss
  0.13820/0.01368, and electronic loss 0.29781/0.00049.
- `outputs/global_l3_matched_direct_validation_v1/` is the completed matched
  direct-total val10 control. Its TRS is 0.37382 +/- 0.08634; paired physical
  macro minus direct TRS is -0.08217 +/- 0.03399 and negative for all seeds.
  This rejects a total-tensor advantage while retaining the separate positive
  direct-U/ionic mechanism result. Frozen test20 remains unread and no
  production promotion is authorized.
- Teacher-U AdamW state is preserved at the joint boundary. The isolated U/V
  tower uses joint LR 5e-4; remaining parameters use the registered 1e-3 LR.
- Production CUDA kernels batch global nonlocal attention and `Z*^T U` with
  masked GEMM/einsum-scatter. Inactive macro/optical paths are omitted from
  branch/strict training and constant-zero losses keep AdamW from decaying
  towers on the wrong stream. `num_workers` remains zero.

## Electronic-generator adjudication

- `data/processed/electrostatic_development_folds.json` is the current
  response-generator development map. It contains 4,939 formula-safe records
  with BEC, same-OUTCAR electronic piezo, electronic dielectric, and force
  constants, without requiring strict Lambda. All 4,995 parsed payloads pass
  explicit finite/shape gates for these fields. Its five formula-disjoint
  development folds have 988/989/987/988/987 materials. Frozen val10/test20
  labels are not read and the map does not replace the production split.
- `data/processed/strict_train1603_development_folds_v1.json` remains the
  development map only for tasks that require strict-complete Lambda.
- The current Cartesian electronic head fails the samples32 same-ID capacity
  gate (active relative error 0.60814, cosine 0.39285), with the dominant
  residual in `l=3`. The explicit global-irrep `l<=3` control passes on the same
  panel (0.04492, 0.99973). These are train-only model-class diagnostics.
- `ElectromechanicalJetHead` is the exact displacement--strain part of the
  first-order electromechanical jet.
  It directly emits BEC/electronic coefficients and defines their common
  first-order polarization increment. Do not demote it to a fidelity control
  or describe it as a finite-perturbation polarization state. Its samples32
  electronic/BEC relative errors are 0.03942/0.01933. Adding algebraically
  redundant response-jet probes gives 0.04167/0.02104 and is not retained.
- `IndependentElectrostaticHeads` is A0: statistically independent BEC and
  electronic-piezo generators with no shared parameter tensors.
- `NonlinearDifferentialPolarizationTower` has exactly two explicit candidates:
  A2 Cartesian polarization and A3 reduced polarization
  `P0=det(F)F^-1 P`, `F=I+eta`. There is no automatic variable switch. It uses
  `Delta P=P_theta(T_eta(x+u_o))-P_theta(x)`, with positions, cell, and periodic
  shifts deformed consistently and no absolute Berry-phase target. BEC and
  electronic piezo are three-output reverse-mode Jacobians of this map. Do not
  wrap coefficient evaluation in `inference_mode`. With only zero-point
  Jacobian labels, its higher-order degrees of freedom have no additional
  identifiable content; finite-perturbation/field labels are needed to test a
  genuinely nonlinear advantage.
- Differentiable reciprocal geometry is never retained in the fixed-geometry
  cache across optimizer steps. Large same-ID cohorts use material-count-
  weighted gradient accumulation; one optimizer update remains the exact
  cohort-mean objective. The samples8/200 CUDA gate passes with electronic
  relative error/cosine 0.14986/0.99961 and BEC 0.02105/0.99937. The
  preregistered samples32/200 gate also passes the joint strong threshold:
  electronic active relative error/cosine are 0.09839/0.99786, l=3 stabilized
  relative error is 0.07766, and BEC relative error/cosine are
  0.04673/0.99710. It used 16+16 material-weighted microbatches, 4,443 CUDA
  optimizer seconds, and 14.09 GiB peak allocated memory. These remain
  noninductive capacity results and are not a production promotion. Because
  all five development folds contain some samples32 IDs, this fitted
  checkpoint must not initialize a held-out-fold experiment.
- The fresh literal-autodiff samples8 response-jet control uses weight 0.25
  and three probes. It changes electronic active error 0.14986 -> 0.13981 and
  l=3 error 0.11294 -> 0.10529, while BEC error worsens 0.02105 -> 0.02455 and
  optimizer time rises 964 -> 1,018 seconds. Preserve this mixed result, but
  do not add the algebraically redundant probe objective to the maintained
  candidate.
- The A0--A3 formula-disjoint runner is
  `piezojet.electrostatic_fold_adjudication`. Every architecture uses the same
  fold-train-only structure checkpoint, random response heads, stochastic
  mini-batches, `num_workers=0`, and development-only selection. A0 initializes
  both independent encoder copies from the same checkpoint but shares no
  trainable parameters. The diagnostic batch is fixed, response-active, and
  norm-stratified; reports contain both all-task and shared-parameter gradient
  norms/cosine.
- A0 backpropagates its two disjoint task losses sequentially within one
  optimizer step. This is exactly the gradient of their sum, verified against
  joint backward parameter by parameter, while only one tower's activation
  graph is resident at a time. It is a memory optimization, not a changed
  objective.
- `piezojet.prepare_electrostatic_adjudication` only writes an auditable command
  plan and can never launch training. The current Stage-A plan is
  `outputs/electromechanical_jet_fold_adjudication/stage_a_n100_fold0_seed42_plan.json`.
  It requires a later explicit user request before execution. The fold-only
  pretrainer derives schema-2 train IDs from the global population minus the
  development subset; it must not expect a duplicated `fold["train"]` field.
- The random-initialized A1 N=100/fold0/seed42 pilot is a retained negative
  control: selected update 25 has electronic active relative error 0.99826,
  cosine 0.06239, amplitude 0.00406, and BEC relative error about 0.99616.
  Its response-active read-only audit has electronic/BEC all-parameter
  gradient norms 0.04598/0.03224, shared norms 0.04547/0.03217, and shared
  cosine -0.01883. The earlier 5,700-fold ratio came from a weak-target prefix
  batch and must not be cited as global loss-scale evidence. It is not the fair
  structure-pretrained Stage-A result. The corresponding A0
  attempt was interrupted on 2026-07-17 at the user's request and has no
  performance result; do not resume or overwrite its directory.

- `outputs/operator_learning_capacity_v2/summary.json` is a retained negative
  same-ID capacity result: the operator bundle helps 1/8 materials but fails
  at 32, including a Phi direction reversal. It does not authorize validation.
- Historical independent-Lambda/material-spectral-floor replays retain their
  fixed v5 weights and negative/partial outputs. They are superseded as the
  current architecture gate by the global-l3 capacity adjudication and must not
  be reused or presented as the maintained candidate.

- `config.yaml` uses the 4,961-item formula-disjoint macro train pool, the
  convention-corrected 4,995-payload v9 cache, strict v10 completions, and the full-
  corpus inductive structural checkpoint. Frozen validation10/test20 IDs are
  unchanged.
- `src/piezojet/train_direct_baseline.py` is the matched control. It uses the
  same macro passes, pretraining, graph, tensor convention, split, seed, and
  validation-selection rule, without physical response factors.
- The exposure replay is two parallel experiments: branch610/strict149 test
  physical `U_{eta,delta}` learning, while macro4961 tests only the isolated
  total predictor. Macro non-degradation is a negative control and must never
  be described as evidence that total-only labels improve ionic factors.
- `outputs/direct_u_multistream_smoke_v1/` is a one-pass end-to-end smoke, not
  a performance claim. Total TRS is -0.00405; direct-U ionic macro cosine is
  -0.01505 and amplitude ratio is 0.00537.
- The same smoke's strict true-factor regularized closure has component MAE
  0.00452 C/m2 and component-micro cosine 0.99997. This validates the declared
  data/operator closure while locating the one-pass failure in learned factors
  and displacement response.
- `scripts/run_teacher_forced_zero_basin_probes.ps1` is the registered
  noninductive 1/8/32-material capacity ladder. It first fits direct factors,
  then supervises `U_{eta,delta}` with true `Phi,Lambda` and true-BEC ionic
  contraction, and holds the homogeneous normal equation at zero during the
  joint phase. Its same-ID results can falsify a zero-basin/capacity failure
  but can never support a formula-disjoint performance claim. Do not expand
  strict training data unless a subsequent matched validation-only comparison
  improves; do not inspect frozen test outputs for that decision.
- Its schema-6 audit finds all 480 true and 480 predicted test optical modes at
  `|lambda| >= 3 delta`, so there is no evidence for a predicted-soft-mode
  shortcut at this checkpoint. The rank-4 true-U oracle retains 95.98% of
  displacement singular energy but has 19.81% true-BEC response error; never
  interpret rank(U)<=6 as six physical phonon modes. True-BEC `Z*^T U_pred`
  has mean cosine 0.1153 and amplitude ratio 0.0563. These remain one-pass
  diagnostics, not performance claims.
- The current periodic e3nn control is incomplete: its train149 structural
  pretraining completed, but a desktop-session interruption stopped direct
  seed 42 at update 29/100 before any test JSON was produced. It has no
  performance result and must be rerun from a fresh output directory before
  being cited or compared.
- `data/processed/_historical_splits/full_corpus_multitask_train149_v1.json` is the registered
  full-corpus multitask split: 4,961 GMTNet train records are formula-disjoint
  from the frozen val10/test20 formulas, while val/test IDs remain identical to
  the strict panel. It may use total GMTNet supervision on the full train pool;
  DFPT and strict-Lambda terms must remain availability-masked. Its driver is
  `scripts/run_full_corpus_multitask_replay.ps1`; it fixes 100 structural,
  50 direct-factor, and 100 joint updates. Its completed v2 three-seed
  diagnostic has factorized total TRS `0.00095 +/- 0.00708`, matched-direct
  TRS `0.00444 +/- 0.01999`, and paired difference `-0.00349 +/- 0.01567`.
  Factorized ionic macro MAE skill is `-0.00670 +/- 0.00216` with amplitude
  ratio `0.03662 +/- 0.00261`; it is a negative diagnostic, not a promotion.
- Production forward propagation must use
  `AtomCoordinateResponsePotential.apply_optical_operator`, not materialized
  dense inverse matrices. `optical_operator()` is diagnostic-only. Stable,
  soft-positive, and unstable DFPT strata must remain separate in reports.

## Artifacts and paper

- Put diagnostic outputs under `outputs/`; do not overwrite prior cohorts.
- Every top-level output cohort must be present in
  `outputs/EXPERIMENT_REGISTRY.json`; every persisted file must be indexed in
  `outputs/EXPERIMENT_ARTIFACT_INDEX.jsonl`. Regenerate both with
  `python -m piezojet.experiment_registry` after a run changes state.
- Preserve negative, failed, blocked, interrupted, partial, and running runs.
  Never delete or reuse their output directories to make a result table look
  complete. New attempts require a fresh cohort or run directory.
- A performance number in the paper must point to a registered summary/test
  artifact and its run-local split, seed, convention, and selection rule.
  Directory existence or a training checkpoint without held-out evaluation is
  not a performance result.
- Paper source is `E:\PAPER\piezojet_equivariant_response_jets\piezojet.tex`.
  Only add evidence-backed statements and distinguish post-freeze diagnostics
  from production performance tables.
- Compile with `latexmk -pdf -interaction=nonstopmode -halt-on-error
  -outdir=output\pdf piezojet.tex`; visually inspect the rendered PDF after a
  material change.
