# PiezoJet agent guide (maintained protocol)

This file describes only the current implementation. Historical experiments,
rejected architectures, and old metrics belong in `docs/EXPERIMENTAL_ARCHIVE.md`
and the experiment registry; do not revive them as fallbacks.

## Runtime and repository rules

- Use `D:\Anaconda\envs\EGNN\python.exe` for Python, PyTorch, tests, and smoke runs.
- Set `PYTHONPATH=E:\CODE\PiezoJet\src` for module entry points.
- Use `apply_patch` for edits. Preserve unrelated dirty files.
- Run focused tests after every change and the full suite for shared model,
  tensor, data, training, or evaluator changes.
- Keep data, caches, checkpoints, and credentials outside Git. Never commit
  `mp.md`, `.env`, API keys, `E:\DATA`, or generated model outputs.
- `data/processed` and `E:\DATA\PiezoJet\processed` are mapped stores;
  never recursively delete, move, or re-index either root during cleanup.
- On the remote server, operate only under `/home/workspace/lrh`; do not merge
  a dirty server worktree into local `main`. Sync through Git commits.

## Scientific and data invariants

- Preserve JARVIS provenance, units, tensor conventions, and source IDs.
- Internal-strain Voigt order is `(xx, yy, zz, yz, xz, xy)`. Do not apply a
  global Voigt, engineering-shear, transpose, or sign patch without a minimal
  controlled reproducer and a regression test.
- A strict `Lambda` label requires symmetry invariance, acoustic nullspace,
  unique identification, redundant-block validation when available, and
  ionic-response closure. Never relax these gates to increase coverage.
- No.187 remains quarantined pending reference-cell/convention resolution;
  never claim its source record is wrong.
- Materials Project metadata must never be mixed into JARVIS-only benchmarks
  without explicit source identity and separate reporting.
- The canonical role map is `data/processed/canonical_datasets.json`; do not
  scan older versioned directories as runtime fallbacks.
- Current public accounting: 4,995 parsed schema-4 payloads plus three
  quarantined raw archives account for all 4,998 requested JARVIS archives.
  Current strict completion cohort is v10 (1,638 accepted); its coverage is
  selection-biased and must not be described as uniform.
- The maintained inductive split is `train1595/val10/test20` with the frozen
  panels unchanged. Never read frozen `test20` unless the user explicitly
  authorizes it. Development maps and validation10 are not test substitutes.

## Current production mathematics

The only maintained factor architecture is
`independent_quadratic_response` with isotropic background. The explicit scalar
factor energy is

`E = 1/2 u^T Phi u - u^T Lambda eta + 1/2 eta^T C eta`.

`Phi` and `Lambda` are independent coefficients; do not impose
`Lambda = B^T K S`, active/null pseudoinverse lifts, ridge charts, or detached
factor reconstructions.

The production ionic response is the differentiable factor/Schur path

`e_ionic = (c_e/Omega) Z*^T D_delta(Phi) Lambda`,

where, in the translation-free optical basis,
`D_delta(Phi) = Q Phi_o (Phi_o^2 + delta^2 I)^(-1) Q^T`.
This is the real Tikhonov resolvent with signed filter
`lambda/(lambda^2 + delta^2)`. The default policy is `tikhonov`; the old
`regularized` spelling is compatibility-only and has identical semantics.
There is no automatic predicted-spectrum branch. Exact inversion is a
true-DFPT stable diagnostic only.

The model prediction is:

`e_phys = e_electronic + e_ionic`.

The independent translation-free direct-`U` head is an amortized-solver
diagnostic, not a second production path. Its agreement with the factor path
and the first-order `U/V` residual may be reported, but it must not silently
replace the factor response or be combined with unrelated factor blocks into a
new constitutive energy.

Total-only GMTNet labels use a separate macro tower and must not backpropagate
into physical factors, BEC, electronic response, `Phi`, `Lambda`, or direct-`U`.
`branch_sum` is a zero-weight closure diagnostic only.

## Training and checkpoint rules

- Factor, electronic, BEC, dielectric, and direct-`U` losses must use explicit
  availability masks and source-matched labels.
- The ionic loss must evaluate the production factor/Schur response whenever
  `ionic_piezo_loss_weight != 0`; do not prune that path accidentally.
- Tensor losses reduce complete Cartesian Frobenius norms before robust losses;
  do not use duplicated componentwise shear metrics.
- Keep material schedules, fold identity, data hashes, code commit, graph
  schema, model configuration, optimizer state, and selection metric in every
  checkpoint. Reject mismatched resume or pretrained initializers.
- Select checkpoints using development/validation only; never tune on frozen
  test20. Same-ID capacity checkpoints cannot initialize inductive folds.
- Preserve failed and interrupted cohorts as immutable evidence, but do not
  let automatic directory scans treat them as current candidates.
- Avoid redundant fallback branches and long loops when a vectorized,
  provenance-equivalent implementation is available. Do not start long
  training before a short smoke, checkpoint-load test, and metric write test.

## Data-download and external-task rules

- Download public JARVIS/DFPT data into `E:\DATA\PiezoJet`, with URL, SHA256,
  parser version, and archive ID recorded. Never fabricate missing labels.
- A new dataset or downloader must have an availability report, schema/units
  test, source-overlap audit, and an explicit statement of whether it may enter
  training, auxiliary supervision, or only diagnostics.
- Do not silently overwrite the canonical dataset map or frozen split.

## Handoff checklist

Before handing work to another agent (including Claude Code): report changed
files, exact mathematical path, tests run, data/checkpoint provenance, known
limitations, and commit hash. Push only intentional source/documentation
commits; leave temporary plans, patches, credentials, and generated outputs
uncommitted.
