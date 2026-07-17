# Historical experiment archive

The maintained package has one production parameterization: independent
atom-resolved `U_eta`; independent quadratic-response `Phi/Lambda`
coefficients of one scalar factor energy; an isotropic background; and the
continuous regularized optical operator. Direct `U_eta`, factor diagnostics,
and the isolated macro tower are not packaged as one model energy: a post-hoc
quadratic density would not prove a common microscopic generator. The strict
first-order U/V residual is the relevant falsifiable compatibility diagnostic.

The following executable branches were removed after negative or
non-identifiable diagnostics:

- macro-to-`Lambda` Moore--Penrose and ridge lifts;
- attached/detached observable-map gradient policies;
- predicted-spectrum `auto` switching;
- sketch/hybrid and mode-aware training objectives;
- protocol A--G training executors.
- the M2.1/implicit first-32 memorization shortcut;
- the fixed-v5 operator-loss bundle, its `operator_losses.py` module and
  capacity executor, and the single-material action launchers.

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

Historical model classes and CLI aliases are not retained in the production
factory. Diagnostic-only exact optical propagation and output summarizers remain
because they test current tensor identities or read immutable artifacts; they
cannot alter the production configuration. A new experimental architecture
needs a separate module, output directory, frozen split, validation-only
selection rule, matched control, and report; it must not be added as a silent
fallback in the production config.
