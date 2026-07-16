# Warning inventory and regression policy

Date: 2026-07-16

The pre-allowlist full suite completed with 108 tests and 403 warnings. The
count is not a numerical-training warning count. It decomposes exactly as:

| Source | Category | Count | Status |
| --- | --- | ---: | --- |
| PyTorch/e3nn TorchScript construction | `torch.jit.script` deprecation | 222 | Upstream transition warning; allowlisted by exact message |
| Python AST invoked by TorchScript/e3nn | Instance-level empty-container annotation warning | 179 | Upstream TorchScript type-system warning; allowlisted by exact message |
| symfc 1.x test API | `displacements=` deprecation | 1 | Read-only oracle compatibility; allowlisted until symfc 1.7 migration |
| symfc 1.x test API | `forces=` deprecation | 1 | Read-only oracle compatibility; allowlisted until symfc 1.7 migration |
| Dataset serialization | — | 0 | No warning observed |
| Numerical linear algebra | — | 0 | No warning observed; any future warning fails pytest |
| Pytest collection | — | 0 | No warning observed; any future warning fails pytest |

`pyproject.toml` now sets all warnings to errors except the four exact known
messages above. A new warning category or changed message therefore fails the
test command instead of silently growing the warning total.

## LaTeX inventory

The current 16-page paper compiles successfully. Its log separately contains:

- two underfull `hbox` notices;
- three underfull `vbox` notices;
- three duplicate PDF destinations (`table.1`--`table.3`) produced by the
  conference template/float combination.

The rendered affected pages were inspected: tables, captions, page numbers,
and text are legible and do not overlap or clip. These layout notices are not
mixed with Python warnings. Undefined references, missing citations, overfull
boxes, fatal errors, and any visual clipping are not allowlisted.

