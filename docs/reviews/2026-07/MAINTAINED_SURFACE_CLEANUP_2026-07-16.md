# Maintained-surface cleanup

## Policy

PiezoJet now distinguishes immutable provenance from executable surface area.
Versioned data cohorts and experiment outputs are retained because they are
needed to reproduce negative results, convention changes, and paper tables.
They are not searched as fallbacks. The one maintained role map is
`data/processed/canonical_datasets.json`.

The repository-facing `data/processed` path maps into the physical
`E:\DATA\PiezoJet\processed` store; it is not an independent duplicate tree.
An attempted duplicate-directory cleanup on 2026-07-16 exposed that Windows
did not report the mapping reliably and traversed into the target. The
canonical schema-4 DFPT payloads were therefore regenerated from the intact
raw-files index, the strict cohort was regenerated with unchanged gates, and
the elastic v2 target was restored byte-for-byte from its independent audit
archive (SHA256
`D83E723A3A66CCE845F2159524A562063E8B598C031BC7ED4CB046205DDC543A`).
Future cleanup must never recursively delete or move either mapped data root.

## Recovery verification (2026-07-16)

The canonical data were regenerated and verified after the mapped-directory
incident. `jarvis_dfpt_v9_full_public/manifest.json` now reports
`requested=4998`, `cached=4995`, and exactly three retained failures:
`JVASP-1048` (archive/GMTNet structure mismatch), `JVASP-34237`, and
`JVASP-7658` (malformed XML). Thus its 4,995 schema-4 payloads plus the three
SHA256-indexed raw quarantines account for every public archive.

The strict completion was run without a condition-number cutoff, rounding
bootstrap, or any other gate relaxation. A four-way concurrent attempt was
resource-interrupted before it wrote any shard manifest, so the same four
deterministic shards were re-run serially. The shard finalizer verified that
the accepted JID set exactly equals the on-disk `JVASP-*.pt` set and wrote the
canonical manifest: `audited=4995`, `accepted=1638`. The recovery logs are
preserved in `outputs/data_recovery_2026-07-16/`.

Legacy split JSON files are grouped under
`data/processed/_historical_splits/`; small version-controlled historical
manifests/fixtures are grouped under `data/processed/_historical_data/`; and
the regenerable schema-1 symmetry cache was removed. The processed-data root
otherwise retains only current role manifests and current regenerable caches.

## Removed executable paths

The following replaced paths are no longer part of the package:

- protocol A--G, mode-aware, sketch-loss, pInv/ridge and observable-lift
  trainers/evaluators and their unit tests;
- the random sketch-gradient benchmark;
- the old response-operator inverse-action capacity launcher;
- predicted-resolvent supervision, direct-U/factorized-ionic consistency, and
  factor-freezing joint-training switches;
- post-freeze test-informed information-gain ranking;
- pre-full-corpus DFPT candidate selection and cache/completion merge tools;
- pre-freeze panel builders, legacy learning-curve subset resamplers, and the
  superseded hexagonal/completion-expansion report executors;
- M3/M5 milestone launchers and the old train149/BEC-transpose replay wrappers;
- unused zero/mean/composition/unconstrained-scalar model classes.

Their existing `outputs/` artifacts remain registered historical evidence.
Removing an executor does not convert a failed result into success and does
not authorize pooling across conventions.

## Current canonical roles

- raw/macro source: GMTNet JARVIS, 4,998 usable records;
- factor cache: JARVIS DFPT v9 full-public, 4,995 parsed payloads;
- raw quarantine: three checksum-indexed non-labelable ZIPs;
- strict completion: v10 zero-dimensional fix, 1,638 accepted;
- factor split: train1603/val10/test20 with the frozen formulas preserved;
- multitask split: macro4961 plus availability-masked strict train1603.

The v9/v10 tokens identify immutable data schemas/cohorts. They are not a
menu of runtime alternatives. New code must use the canonical role map or an
explicit path and must never scan v4--v10 until something loads.

The maintained factor architecture has one name only:
`independent_quadratic_response`. Its implementation is
`IndependentQuadraticResponseHead`, exposed on the model as
`response_factors`. The replaced `energy_learned_strain` name and
`energy_factors` attribute have no compatibility alias; incompatible old
checkpoints fail explicitly instead of selecting a hidden fallback.

## Remaining deliberate diagnostics

Exact stable-DFPT propagation, symfc projection, parser cross-validation, and
historical experiment registry readers remain read-only diagnostics. They are
not production fallbacks and cannot modify cached labels or select a model.
