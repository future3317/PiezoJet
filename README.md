# PiezoJet

PiezoJet learns an O(3)-equivariant piezoelectric tensor as the mixed derivative of one scalar response potential, `Phi(x, E, eta)`. The MVP uses only GMTNet's released JARVIS-DFT piezoelectric data; it does not silently substitute another source.

## Current audited state (2026-07-14)

The current work is a physical-data implementation and diagnosis, not a
state-of-the-art accuracy claim.

- GMTNet/JARVIS piezoelectric index coverage: 4,998 / 4,998 records.
- Same-source JARVIS DFPT strict-cache lineage: the immutable base v4 has 99
  completions; the cubic-heavy information-gain v1 addition yields v5 with
  127; the current v6 has 138 completions after a coverage-aware addition.
- Historical completion-likelihood batch: 247 parseable DFPT archives, 71
  strict acceptances (28.7%); three malformed OUTCAR strain blocks remain
  recorded as acquisition failures, not relaxed labels.
- Coverage-aware JARVIS-only acquisition: a 100-formula queue reserved
  retrieval slots by the frozen test-panel crystal-system frequencies and
  excluded previously audited IDs. Of 98 available archives, 11 strictly pass
  (5 cubic, 3 hexagonal, 2 tetragonal, 1 orthorhombic). No retrieved trigonal,
  monoclinic, or triclinic material passed; strict thresholds were unchanged.
- Frozen complete-factor benchmark:
  `data/processed/strict_completion_benchmark_v1.json`, with formula-disjoint
  69 / 10 / 20 train / validation / test materials. The 20-material test panel
  must never be reassigned as data expands.
- No.187 (`P-6m2`) is quarantined from *strict completion* only. Synthetic
  recovery, non-orthogonal fractional/Cartesian round trips, and affine group
  closure pass; raw operation-level sign conflicts remain unresolved. Its
  printed partial labels are retained for their valid masked supervision.

The first registered nested learning curve is complete. Its three-seed anchors
at `N_Lambda = 19, 35, 69` use the same fixed panel and update budget for the
full production and direct-factor protocols. The full report is
`outputs/strict_learning_curve_v1/report/learning_curve.md`. The direct-factor
protocol improves held-out full-Lambda learning at 69 labels
(`0.210 +/- 0.058` cosine versus `0.027 +/- 0.045` for production), but ionic
skill remains below the zero predictor. This is evidence of an optimization and
response-subspace bottleneck in addition to limited certified coverage, not a
general performance claim.

The registered fourth-feedback forensics refine that conclusion without
changing the frozen panel.  On the same 69 labels, 100 factor-only updates (A)
reach full-Lambda cosine `0.326 +/- 0.067`; the current 50-factor plus
50-frozen-joint protocol (B) reaches `0.210 +/- 0.058`.  Holding the first 50
factor updates fixed but allowing joint training to rewrite the factor stack
(C) drops this to `0.067 +/- 0.162`; in seed 42, validation cosine falls from
`0.387` after factor training to `0.178` after the unfrozen joint stage, while
the frozen counterpart remains `0.362`.  This is evidence of factor-path
interference, not evidence that B solves ionic response. Five response-matched
35-label subsets span cosine `0.048--0.258`, so material composition is also a
large uncertainty. The complete report is
`outputs/feedback4_execution_v1/report/feedback4_report.md`. The new
degeneracy-safe mode-aware loss is implemented but remains disabled by default:
its one-seed smoke did not improve predicted ionic skill.

The fifth-feedback audit fixes a previously ambiguous ionic metric. Every DFPT
evaluation now reports (i) material-balanced cosine, (ii) component-micro
cosine, (iii) an active-material cosine using the preregistered independent
18-component threshold `0.05 * sqrt(18) C/m2`, and (iv) material-balanced
amplitude. The canonical true-`Z*`/true-`Phi`/predicted-`Lambda` oracle uses
the same signed regularized operator as training. The frozen-panel A--D replay
shows why this distinction matters: B has ionic micro cosine `0.965 +/- 0.008`
but material-balanced oracle cosine only `0.017 +/- 0.131`. Therefore the
former must not be described as material-level ionic direction generalization.
The response decomposition instead finds that C/D's near-zero total TRS
coincides with collapsed predicted total norms (0.172 and 0.104 of true-total
norm), not demonstrated successful branch cancellation.

The evidence-driven protocol E (100 factor updates, validation-factor restore,
then 50 frozen-factor joint updates) preserves the factor-only full-Lambda
result (`0.326 +/- 0.067`) but has total TRS `-2.179 +/- 0.395`; it is a
negative result, not the new default. Canonical diagnostics are in
`outputs/feedback5_execution_v1/report_v2/feedback5_report.md`. The dry
information-gain retrieval queue is
`outputs/information_gain_cohort_v1/cohort.json`: it preserves the frozen
panel, excludes previously completed/frozen-formula candidates and No.187,
and does not invent ensemble uncertainty when no ensemble prediction exists.

Two subsequent prospective, validation-selected gradient-surgery diagnostics
also remain non-default. Protocol F applies a one-sided factor-protected
projection to the B/E factor stack; it executes on 97/150 joint updates but
still has amplitude-collapse characteristics (full-Lambda `0.142 +/- 0.027`,
total TRS `-0.038 +/- 0.030`). Protocol G follows the same projection with
per-update unit-norm matching on that stack, recovering full-Lambda
`0.228 +/- 0.099` and a larger predicted total norm ratio (`0.325`), but it
does not produce positive total skill (`-0.141 +/- 0.137`) or positive
predicted ionic skill. Both are auditable negative controls, not production
settings. Their combined canonical report is
`outputs/factor_protected_norm_match_v1/report/feedback5_report.md`.

Two JARVIS-only, post-freeze train expansions now test the data diagnosis under
the identical B schedule (50 direct-factor plus 50 frozen-factor joint
updates), with loss-only checkpoint selection and the original 10/20
validation/test IDs intact. The 69/97/108-train canonical macro results are:

| Train materials | Full-Lambda cosine | Oracle ionic cosine | Predicted ionic skill | Total TRS |
| ---: | ---: | ---: | ---: | ---: |
| 69 | 0.210 +/- 0.058 | 0.017 +/- 0.131 | -0.054 +/- 0.052 | -0.470 +/- 0.386 |
| 97 (cubic-heavy addition) | 0.194 +/- 0.129 | 0.079 +/- 0.078 | 0.025 +/- 0.016 | -0.368 +/- 0.162 |
| 108 (coverage-aware addition) | 0.223 +/- 0.044 | 0.143 +/- 0.094 | 0.042 +/- 0.019 | -0.258 +/- 0.151 |

The coverage-aware labels improve the ionic path and its amplitude diagnostics,
but total response remains below the zero predictor. These are frozen-panel,
post-selection data-regime diagnostics, not a production accuracy table or a
claim of low-symmetry coverage.

## Reproduce

```powershell
& 'D:\Anaconda\envs\EGNN\python.exe' -m pip install -e .
python scripts/download_data.py --output data/raw/gmtnet
python scripts/inspect_data.py --root data/raw/gmtnet
& 'D:\Anaconda\envs\EGNN\python.exe' -m pytest -q
# Public, same-source DFPT labels; use --limit first to verify connectivity.
python -m piezojet.jarvis_dfpt --data-root data/raw/gmtnet --output-dir data/processed/jarvis_dfpt_v1 --limit 1
python -m piezojet.train --config config.yaml --loss full --overfit-32
python scripts/run_pipeline.py --config config.yaml --loss full
python scripts/run_pipeline.py --config config.yaml --loss sketch
python -m piezojet.evaluate --checkpoint outputs/best.pt --split test
# Audited physical-unit metrics on the formula-disjoint DFPT subset.
python -m piezojet.evaluate_dfpt --checkpoint outputs/best.pt --split test
```

For the frozen complete-factor benchmark, use the explicit split file rather
than the global 4,998-material split:

```powershell
$env:PYTHONPATH = 'E:\CODE\PiezoJet\src'
& 'D:\Anaconda\envs\EGNN\python.exe' -m piezojet.train `
  --config config.yaml `
  --splits-file data/processed/strict_completion_benchmark_v1.json `
  --seed 42 --output-dir outputs/strict_completion_v4_seed42

& 'D:\Anaconda\envs\EGNN\python.exe' -m piezojet.evaluate_dfpt `
  --checkpoint outputs/strict_completion_v4_seed42/loss_best.pt `
  --splits-file data/processed/strict_completion_benchmark_v1.json `
  --split test --output outputs/strict_completion_v4_seed42/dfpt_test.json
```

The strict-completion tools are deliberately separate from model training:

```powershell
# Audit only; no threshold is changed by this command.
& 'D:\Anaconda\envs\EGNN\python.exe' -m piezojet.strain_completion `
  --material-ids-file outputs/jarvis_dfpt_expansion_v1/completion_likelihood_cohort.json `
  --output-dir outputs/jarvis_dfpt_expansion_v1/completion_likelihood_strict

# Build nested train subsets while preserving the frozen validation/test panel.
& 'D:\Anaconda\envs\EGNN\python.exe' -m piezojet.select_learning_curve_subsets

# Registered fourth-feedback diagnostics. These never alter the frozen test IDs.
& 'D:\Anaconda\envs\EGNN\python.exe' -m piezojet.resample_stratified_subsets
& 'D:\Anaconda\envs\EGNN\python.exe' -m piezojet.protocol_ablation `
  --config config.yaml `
  --splits-file outputs/strict_learning_curve_v1/splits/strict_lambda_n69.json `
  --output-root outputs/optimization_ablation_v1 `
  --protocol all --seeds 42,43,44 --factor-updates 50 --joint-updates 50
& 'D:\Anaconda\envs\EGNN\python.exe' -m piezojet.summarize_feedback4

# P0/P2: replay already selected checkpoints with canonical macro/micro ionic
# reporting. This is post-selection only.
& 'D:\Anaconda\envs\EGNN\python.exe' -m piezojet.reevaluate_protocol_checkpoints `
  --root outputs/optimization_ablation_v1 `
  --splits-file data/processed/strict_completion_benchmark_v1.json `
  --device cpu --output-dir outputs/feedback5_execution_v1/canonical_all_v3
& 'D:\Anaconda\envs\EGNN\python.exe' -m piezojet.summarize_feedback5 `
  --input outputs/feedback5_execution_v1/canonical_all_v3 `
  --output-dir outputs/feedback5_execution_v1/report_v2

# P1: protocol E is explicit; `--protocol all` intentionally remains A--D.
& 'D:\Anaconda\envs\EGNN\python.exe' -m piezojet.protocol_ablation `
  --config config.yaml --splits-file data/processed/strict_completion_benchmark_v1.json `
  --protocol E --seeds 42,43,44 --device cpu `
  --output-root outputs/feedback5_execution_v1/protocol_e

# Prospective factor-protected gradient diagnostics. These use validation-only
# checkpoint selection; neither is a production setting.
& 'D:\Anaconda\envs\EGNN\python.exe' -m piezojet.protocol_ablation `
  --config config.yaml --splits-file data/processed/strict_completion_benchmark_v1.json `
  --protocol F --seeds 42,43,44 --device cpu `
  --output-root outputs/factor_protected_projection_v1
& 'D:\Anaconda\envs\EGNN\python.exe' -m piezojet.protocol_ablation `
  --config config.yaml --splits-file data/processed/strict_completion_benchmark_v1.json `
  --protocol G --seeds 42,43,44 --device cpu `
  --output-root outputs/factor_protected_norm_match_v1

# P3: rank only. It downloads nothing and cannot accept a label or edit a split.
& 'D:\Anaconda\envs\EGNN\python.exe' -m piezojet.rank_information_gain_cohort `
  --output outputs/information_gain_cohort_v1/cohort.json

# Coverage-aware retrieval excludes prior strict audits and reserves queue slots
# by frozen-test crystal system. It still cannot change a completion gate.
& 'D:\Anaconda\envs\EGNN\python.exe' -m piezojet.rank_information_gain_cohort `
  --strict-completion-manifest data/processed/jarvis_strain_completion_v6/manifest.json `
  --dfpt-dir data/processed/jarvis_dfpt_v3 `
  --prior-audit-manifest data/processed/jarvis_strain_completion_information_gain_v1/manifest.json `
  --prior-audit-manifest data/processed/jarvis_strain_completion_information_gain_v2_test_crystal_coverage/manifest.json `
  --selection-policy test_crystal_coverage `
  --output outputs/information_gain_cohort_v2_test_crystal_coverage/cohort.json
```

The production pipeline always performs masked-species and translation-free
coordinate-denoising structural pretraining before piezoelectric fine-tuning.
The local encoder is a channelized Cartesian periodic environment: an invariant
chemical backbone weights vector and traceless-quadrupole edge bases, then a
learned channel-space interaction matrix forms local many-body modes without
Clebsch--Gordan feature propagation. This is an efficient backbone rather than
the paper's central novelty. The response model combines local equivariant polar
motifs with a physical reciprocal-vector shell, an origin-invariant polar--chemical
cross-spectrum, and PBC radial fluctuation correlations. The cross-spectrum
directly constructs a rank-three polar correction, so global information affects
tensor direction as well as invariant amplitudes. Six-frame polar-decomposition
synthesis has been removed from the production path.

The Cartesian encoder has a new pretraining checkpoint lineage:
`outputs/pretrain_cartesian/best_encoder.pt`. Earlier spherical-encoder
checkpoints are intentionally rejected instead of being silently misloaded.

`inspect_data.py` is a required gate: it records the raw fields, units, source Voigt order, split status, finite-value check, and atom counts before training. The production split is formula-grouped and stratified by the rotation-invariant piezoelectric tensor norm, preventing formula leakage while preserving zero, weak, moderate, high, and very-high response populations in every partition. Source labels are Reynolds-projected to each structure's point group once and cached under `data/processed/piezo_symmetry_targets_v1/`.
When data is manually copied rather than cloned by `download_data.py`, create `data/raw/gmtnet/SOURCE_COMMIT.txt` containing the exact 40-character GMTNet commit SHA. Training refuses to start without it, so results remain reproducible.

## Conventions

GMTNet labels are `piezoelectric_C_m2` with source columns `[xx, yy, zz, xy, yz, xz]`. PiezoJet converts them once at ingestion to `[xx, yy, zz, yz, xz, xy]`, using engineering shear strain `[exx, eyy, ezz, 2eyz, 2exz, 2exy]`. Internally the tensor is `e_ijk=e_ikj`, represented through `e3nn.io.CartesianTensor("ijk=ikj")` (18 dimensions).

The GMTNet Piezo block obtains its source columns by differentiating with respect to one entry of a symmetric Cartesian strain matrix. Therefore the source shear coefficient is `e_ij`, not `e_ij/2`; PiezoJet stores both Cartesian entries as `e_ij` and puts the factor of two only in the engineering-strain conversion. This is enforced by `test_engineering_shear_matches_single_symmetric_component_derivative`.

The response model now works in the physical atom-coordinate space. It predicts
node-aligned Born tensors `Z*`, a variable-size force-constant Hessian `Phi`,
and node-aligned strain forces `Lambda`. An exact Cartesian projector removes
the three uniform translations for every crystal, after which the optical
solve yields `e_ion = (16.02176634 / volume) Z* Phi_opt^-1 Lambda` in C/m2.
There is no fixed mode count and no phonon padding. The force-constant head
enforces block-transpose symmetry and the acoustic sum rule while retaining
real negative/unstable DFPT modes. A signed damped pseudoinverse
`lambda/(lambda^2+delta^2)` makes soft-mode crossings finite without changing
their sign; BEC and internal strain obey their corresponding zero-sum
constraints. The same relaxation also supplies lattice dielectric and elastic
corrections with explicit unit conversion. GMTNet's summary records do not embed BEC or phonon arrays,
but every current `JVASP-*` label has a matching public JARVIS `raw_files`
DFPT archive. `piezojet.jarvis_dfpt` downloads, parses, and structure-validates
those archives into atomic BEC targets, the raw ionic piezoelectric tensor,
and cached Gamma-point dynamical eigenpairs, force constants, and
symmetry-inequivalent internal-strain blocks. Training directly supervises BEC,
the cleaned physical Hessian, only the internal-strain blocks actually printed
by VASP, and the complete ionic response product. It does not fabricate a
symmetry expansion whose VASP convention has not been numerically validated.
Variable-length phonon arrays are retained without padding for audits. The parser deliberately
stores VASP dynamical eigenvalues rather than treating potentially repeated
`OUTCAR` blocks as a frequency label.

`piezojet.evaluate_dfpt` is deliberately separate from the full-dataset tensor
evaluator. It accepts either an audited-ID restriction of the global split or a
frozen explicit `--splits-file`, and reports BEC, force-constant, printed internal-strain, ionic-piezo,
electronic-piezo, total-piezo, and dielectric errors in physical units, with
zero-predictor skills and per-material CSV rows. Ionic outputs explicitly
separate material-macro from component-micro aggregation, label the optical
operator policy, and audit electronic/ionic magnitudes plus cancellation; the
micro legacy aliases are never the canonical material-level result. The optional direct-factor
curriculum is enabled with `--factor-pretrain-epochs`; its selected factor
stack can be protected during a response ablation with
`--freeze-factors-during-joint`. Both are off by default; the 99-label frozen
benchmark shows that direct-factor pretraining can improve full-Lambda
direction while ionic response remains a separate response-active bottleneck.
`mode_aware_strain_loss_weight` adds gauge-safe true-DFPT optical-subspace
supervision for complete strict labels: it compares block projections rather
than individual eigenvectors across exact or near degeneracies. It is disabled
by default because the registered one-seed smoke establishes implementation
validity, not an ionic-response improvement.

`--loss full` is the production balanced robust tensor objective. `--loss
sketch` differentiates the relaxed potential with JVPs and is retained as a
diagnostic rather than claimed as a speed or memory improvement for this
18-dimensional output.
