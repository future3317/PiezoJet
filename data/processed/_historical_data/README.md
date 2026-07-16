# Historical data fixtures

This directory contains the small, version-controlled manifests and fixtures
needed to audit pre-full-public experiments. They are immutable provenance,
not runtime fallbacks. Maintained code must use `../canonical_datasets.json`
or an explicit path and must never scan this directory automatically.

Large superseded caches are not duplicated in the code repository. Their
registered experiment outputs remain under `outputs/`; the current complete
public-source factors are regenerated from the canonical raw archive index.
