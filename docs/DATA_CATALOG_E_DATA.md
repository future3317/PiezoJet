# PiezoJet data catalog and external storage policy

## Physical location

All project data now physically resides under `E:\DATA\PiezoJet`.  The
workspace paths `E:\CODE\PiezoJet\data\raw` and
`E:\CODE\PiezoJet\data\processed` are Windows junctions to that location,
so existing relative configuration remains reproducible.  The user's
`E:\DATA\.env` is not copied, indexed, or committed.

`data/processed/canonical_datasets.json` is the authoritative role map.
Versioned directories below are immutable provenance cohorts; maintained code
does not scan them as ordered fallbacks.

## JARVIS/GMTNet response sources

| Source | Stored artifact | Coverage | Intended role | Current rule |
|---|---|---:|---|---|
| GMTNet JARVIS piezo/dielectric | `raw\gmtnet\data\jarvis_diele_piezo.pkl` | 5,000 source rows; 4,998 usable GMTNet records | Structural inputs, total piezoelectric tensor, dielectric fields | Macro total uses the published GMTNet total target. |
| GMTNet JARVIS elastic | `raw\gmtnet\data\jarvis_elastic.pkl`; `processed\jarvis_elastic_auxiliary_v2_reynolds` | 14,500 rows; 3,346 exact piezo intersections; 3,291 audited targets | Availability-masked relaxed-elastic auxiliary supervision | Raw kbar stiffness is retained; target is converted once to GPa and point-group Reynolds projected. The default production weight remains zero pending a separately registered validation-selected auxiliary run. |
| JARVIS public raw DFPT | `processed\jarvis_dfpt_v9_full_public`; `raw\jarvis_dfpt_v9_quarantine` | 4,998/4,998 GMTNet-aligned archives accounted for: 4,995 schema-4 tensor payloads and 3 raw-byte quarantines | BEC, force constants, Gamma dynamical data, dielectric, OUTCAR ionic/total piezo, printed internal strain | Every one of the 4,995 parsed payloads has finite BEC, masses, Gamma eigensystem, raw dynamical matrix, force constants, ionic/total piezo, and `epsilon`/`epsilon_ion`/`epsilon_rpa`. Thirty-nine VASP numeric-overflow records retain those valid fields and only mask irrecoverable printed Lambda blocks. JVASP-1048 is structure-mismatched; JVASP-34237 and JVASP-7658 contain malformed XML. Their raw zips and SHA256 are retained but never exposed as aligned labels. |
| Full-public electrostatic development pool | `processed\electrostatic_development_folds.json` | 4,939 formula-safe materials; development folds 988/989/987/988/987 | Formula-disjoint BEC/electronic-piezo/electronic-dielectric generator development | Uses all parsed records with same-archive electrostatic labels after frozen-formula, malformed/alignment, and two-record convention exclusions. Strict `Lambda` is not required; frozen val10/test20 labels are unread. |
| Official JARVIS 2022 `dft_3d` release | `jarvis_dft3d_index\jdft_3d-12-12-2022.json.zip` | 75,993 unique JIDs; 4,937 overlap with the 4,998 GMTNet IDs | Auxiliary structures and scalar metadata | ZIP CRC and SHA256 are audited. It cannot replace GMTNet-pinned structures or DFPT labels because 61 historical IDs are absent and some same-JID relaxed cells differ. |
| Observed-only DFPT partial audit | `audits\jarvis_dfpt_v10_full_public_label_coverage` | One row per usable GMTNet record | High-quality BEC, force-constant, branch, and *printed* internal-strain coverage | A qualified partial requires finite, shape-valid, archive-provenanced source arrays. It never asserts that unprinted `Lambda` entries are known and is never substituted for strict completion. |
| Strict internal-strain completion | canonical `processed\jarvis_strain_completion_v10_zero_dimensional_fix`; historical v7--v9 retained | 1,638 accepted of 4,995 audited payloads; corrected train1595 after reduced-formula exclusion | Full `Lambda` only after complete source-block parsing, symmetry, acoustic, identification, redundant-block, and ionic closure gates | Never relax gates to increase count. The v10 zero-dimensional invariant-space correction changes audit correctness, not thresholds. Numeric-overflow blocks categorically prevent strict completion. |

The GMTNet elastic intersection is not the full 5,000 piezo set.  It is an
availability-masked JARVIS auxiliary candidate, not a license to fabricate
elastic labels for missing IDs.  The v2 audit retains raw residuals and accepts
only targets that pass the projected point-group, engineering-shear,
stiffness/compliance, and source bulk/shear-modulus checks.

## Materials Project summary auxiliary data

`external\materials_project_summary_v1\materials_project_summary.jsonl` is
a separately source-tagged table.  It contains MP summary fields:
`is_stable`, `energy_above_hull`, `band_gap`, `formation_energy_per_atom`,
`total_magnetization`, `ordering`, and `theoretical`.

Mapping is accepted only if JARVIS `dft_3d` metadata has an explicit
`reference = mp-<id>`, then passes a reduced-formula check.  No formula-only or
nearest-structure matching is used.  The manifest records all missing MP
summaries and mismatches.  This table may support a separately registered,
availability-masked auxiliary task, but it must never change JARVIS-only
frozen validation/test labels or be silently pooled into their reported
metrics.

`jarvis_raw_index` and `jarvis_dft3d_index` must be distinct directories.
`jarvis-tools` manages `raw_files` and `dft_3d` downloads independently, and
sharing a cache directory can replace one dataset ZIP while another reader is
active.

## Maintained audit commands

```powershell
python -m piezojet.audit_responses --data-root E:\DATA\PiezoJet\raw\gmtnet --output E:\DATA\PiezoJet\audits\gmtnet_response_v1
python -m piezojet.dfpt_label_coverage --data-root E:\DATA\PiezoJet\raw\gmtnet --dfpt-dir E:\DATA\PiezoJet\processed\jarvis_dfpt_v9_full_public --strict-completion-dir E:\DATA\PiezoJet\processed\jarvis_strain_completion_v10_zero_dimensional_fix --output-dir <fresh-audit-directory>
python -m piezojet.materials_project_auxiliary --data-root E:\DATA\PiezoJet\raw\gmtnet --dft3d-cache-dir E:\DATA\PiezoJet\jarvis_dft3d_index --env-file E:\DATA\.env --output-dir E:\DATA\PiezoJet\external\materials_project_summary_v1
```

The full-public acquisition is complete, so its one-off shard/finalizer
wrappers were removed from the maintained script directory. The underlying
`piezojet.jarvis_dfpt` command remains atomic and restartable for an explicitly
fresh output directory. Never overwrite the canonical cache. The coverage
audit reports numerator, denominator, and rate by crystal system, atom-count
bin, and GMTNet response-magnitude bin; it is not a new supervision threshold.
