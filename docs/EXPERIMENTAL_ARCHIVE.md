# Historical experiment archive

The maintained package has one production parameterization: independent
atom-resolved `U_eta`, `energy_learned_strain` physical factors, an isotropic
background, and the continuous regularized optical operator.

The following executable branches were removed after negative or
non-identifiable diagnostics:

- macro-to-`Lambda` Moore--Penrose and ridge lifts;
- attached/detached observable-map gradient policies;
- predicted-spectrum `auto` switching;
- sketch/hybrid and mode-aware training objectives;
- protocol A--G training executors.

Their persisted JSON, CSV, checkpoints, and reports under `outputs/` remain
historical evidence. They describe the code and data convention recorded in
their own resolved configuration; they are not current implementation paths
and must not be pooled with direct-`U_eta` results.

The complete directory-by-directory ledger is `EXPERIMENT_REGISTRY.md`, with
machine-readable cohort/subrun metadata in
`outputs/EXPERIMENT_REGISTRY.json`. Every persisted file, including partial
checkpoints and failure markers, is listed in
`outputs/EXPERIMENT_ARTIFACT_INDEX.jsonl`. The ledger intentionally keeps
negative and interrupted work; a checkpoint without held-out evaluation is
not silently promoted to a completed result.

The generic gradient-conflict math remains in `gradient_diagnostics.py` for
read-only auditing. The production trainer does not import it.

Legacy model classes may still exist where required to deserialize an old
checkpoint or test a tensor identity, but `model_from_config` rejects every
architecture except the maintained one. A new experimental architecture needs
a separate module, output directory, frozen split, validation-only selection
rule, matched control, and report; it must not be added as a silent fallback in
the production config.
