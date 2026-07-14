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

- `data/processed/jarvis_strain_completion_v4/` holds 99 schema-validated,
  accepted strict completions.
- `data/processed/jarvis_strain_completion_v5/` adds the 28 accepted labels
  from the first information-gain cohort (127 total); its cohort is strongly
  cubic-heavy and must not be described as a uniform scaling result.
- `data/processed/jarvis_strain_completion_v6/` is the current fresh union
  (138 accepted). It adds 11 accepted labels from a coverage-aware JARVIS-only
  queue of 100 formulas. That queue reserved retrieval slots by the frozen
  test-panel crystal-system frequencies, excluded previously audited IDs, and
  retained every strict gate. Of 98 available archives it accepted 5 cubic, 3
  hexagonal, 2 tetragonal, and 1 orthorhombic materials; it accepted none of
  the retrieved trigonal, monoclinic, or triclinic materials. Do not claim
  coverage of those unresolved classes.
- `data/processed/strict_completion_benchmark_v1.json` is frozen at
  69/10/20 formula-disjoint train/validation/test. Never reassign the 20 test
  IDs as new data arrive.
- `data/processed/strict_completion_benchmark_train97_v1.json` and
  `data/processed/strict_completion_benchmark_train108_v1.json` may add only
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
- `outputs/feedback4_execution_v1/report/feedback4_report.md` records the
  registered fixed-coverage A/B/C/D forensic, five one-seed resampled
  35-label subsets, and one mode-aware smoke. Treat the A/B/C/D result as a
  post-freeze optimization diagnostic, not a production performance table.
- The gauge-safe `mode_aware_strain_loss_weight` is disabled by default. Its
  one-seed smoke did not improve predicted ionic skill; do not enable it for a
  production claim without a preregistered multi-seed comparison.
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

## Training conventions

- Production curve protocol: 100 fixed gradient updates (`batch_size=128`, so
  one update per epoch for these subsets), no early stop, checkpoint chosen by
  validation loss only.
- Direct-factor curve protocol: 50 direct-factor updates plus 50 frozen-factor
  joint updates; same seed, optimizer, batch size, and frozen panel.
- Intermediate single-seed values are diagnostics. Three seeds are required
  before comparing learning-curve points; do not describe the 23/50 phase-1
  rows as three-seed results.
- Protocol E is a registered negative follow-up: 100 direct-factor updates,
  validation-factor restore, then 50 frozen-factor joint updates. Its three
  seeds preserve factor direction but worsen total TRS; do not make it a
  production default or tune it using test results. `--protocol all` in
  `protocol_ablation` deliberately continues to mean only the legacy A--D set.
- Protocols F/G are prospective three-seed factor-protected gradient
  diagnostics. F projects conflicting response gradients off the direct-factor
  direction on the B/E factor stack; G also unit-norm matches the projected
  response gradient. Neither produces positive ionic skill or total TRS, so
  neither is a production default or a loss-weight tuning basis. Their
  canonical A--G report is
  `outputs/factor_protected_norm_match_v1/report/feedback5_report.md`.

## Artifacts and paper

- Put diagnostic outputs under `outputs/`; do not overwrite prior cohorts.
- Paper source is `E:\PAPER\piezojet_equivariant_response_jets\piezojet.tex`.
  Only add evidence-backed statements and distinguish post-freeze diagnostics
  from production performance tables.
- Compile with `latexmk -pdf -interaction=nonstopmode -halt-on-error
  -outdir=output\pdf piezojet.tex`; visually inspect the rendered PDF after a
  material change.
