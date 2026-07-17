# Global-l3 capacity and joint-loss geometry adjudication

## Scope

This note records the implementation and validation-only experiments performed
after the external capacity/optimization adjudication.  No frozen test20 label
was loaded for architecture choice, gradient diagnosis, or checkpoint
selection.  The only held-out panel used here is the immutable,
formula-disjoint validation10 panel.

## Representation corrections

The periodic graph cache is now schema 4.  A nominal `max_neighbors` boundary
retains the complete equal-distance shell, preventing arbitrary truncation of
symmetry-equivalent neighbors.

The displacement response is predicted by its own encoder and global readout.
The readout receives local polar features, reciprocal/global context, explicit
strain queries, and an STF octupole (`l=3`) channel.  Both the physical
response `U` and training-only auxiliary `V` are projected against uniform
translations.

The maintained consistency residual is the first-order real block system

\[
\Phi U-\delta V=\Lambda,\qquad
\Phi V+\delta U=0,
\]

not the historical squared normal equation.  During the current stage the
residual uses true `Phi/Lambda`, is warmed/ramped, and is capped by its measured
gradient norm relative to the direct `U*` loss.

## Capacity gate

The prior local/global heads without an independent `l=3` channel remained
poor on samples32 even after 500 epochs.  The global-l3 candidate gave:

| Candidate | U relerr | U cosine | active ionic cosine | amplitude |
|---|---:|---:|---:|---:|
| global, no consistency, 100 epochs | 0.22088 | 0.92454 | 0.99616 | 1.02167 |
| global, no consistency, 200 epochs | 0.15827 | 0.95703 | 0.99829 | 1.01418 |
| global, first-order, 200 epochs | 0.14829 | 0.94524 | 0.99962 | 1.00605 |

The no-consistency 200-epoch run passes the declared samples32 capacity gate.
This is same-ID memorization evidence only.

The corrected output-basis oracle now evaluates the actual head used by each
checkpoint.  The superseded global head has mean/worst minimum relative
residual `0.09498/0.91137` and mean maximum cosine `0.96213`.  Adding the
independent STF octupole reduces those residuals to `0.00366/0.05945` and raises
mean maximum cosine to `0.99989`.  The unrestricted translation-free lookup
has worst residual `8.64e-9`.  Thus the explicit `l=3` family removes a real
high-symmetry output-basis blind spot; the remaining capacity error belongs to
structure-to-coefficient conditioning rather than a missing translation-free
target space.

## Joint-boundary ablations

The isolated-U precursor reached teacher validation losses 0.31775 for direct
U and 0.23902 for true-BEC ionic response, then degraded in joint training.
All following comparisons start from the same teacher checkpoint.

| Mechanism, epoch 1 | val direct-U | val ionic | val loss |
|---|---:|---:|---:|
| fresh AdamW, U LR 5e-4 | 0.40784 | 0.33643 | 1.65746 |
| preserved AdamW, U LR 5e-4 | 0.39187 | 0.29971 | 1.63485 |

Lower learning rate and state continuity help only modestly.  They do not
explain the full teacher-to-joint degradation.

## Gradient localization

The same-batch audit used all 1,603 strict training materials and the entire
isolated U tower:

| Gradient route | Norm | cosine with direct-U |
|---|---:|---:|
| direct-U | 0.08167 | 1.000 |
| true-BEC ionic | 0.13242 | +0.54927 |
| branch sum | 0.30024 | -0.55615 |
| ionic + branch sum | 0.19322 | -0.46044 |

The physical ionic supervision is aligned with direct U.  The conflict comes
from re-optimizing `electronic + ionic = total` while the electronic head is
inaccurate.  The sum residual can reduce itself by making U compensate for the
electronic error.

The matched OUTCAR source audit already verifies this identity after a common
conversion, so it provides no independent target information.  The maintained
configuration therefore sets `branch_sum_loss_weight: 0.0`.  Electronic and
true-BEC ionic component losses remain attached; sum closure remains logged.
No inference fallback or alternate forward path was added.

## Validation-only replication and matched control

The no-redundant-sum candidate was run for ten complete joint passes.  The
first-order term ramped during epochs 4--6.  Validation loss selected epoch 7:

| Run | val loss | direct-U | ionic | total TRS |
|---|---:|---:|---:|---:|
| isolated-U precursor, epoch 9 | 1.54465 | 0.37147 | 0.29212 | 0.05800 |
| no redundant sum, epoch 7 | **0.97701** | **0.24816** | **0.13628** | **0.38696** |

The unoptimized branch-sum closure diagnostic is 0.99172 at the selected
checkpoint.  This is not hidden: it identifies electronic response as a
remaining independent bottleneck.  It is not a reason to corrupt U again.

The same validation-loss selection rule was then applied independently to
seeds 7 and 1729.  The matched direct-total control uses the same 4,961 macro
train records, complete-shell graph, structural checkpoint, ten macro passes,
seed, and formula-disjoint val10 panel.

| Seed | Physical epoch | Physical TRS | direct-U | Ionic | Direct epoch | Direct TRS | Physical - direct |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 7 | 0.38696 | 0.24816 | 0.13628 | 10 | 0.46791 | -0.08096 |
| 7 | 4 | 0.24944 | 0.24734 | 0.12559 | 8 | 0.29824 | -0.04880 |
| 1729 | 6 | 0.23855 | 0.25613 | 0.15274 | 9 | 0.35530 | -0.11675 |
| Mean +/- sample SD | -- | 0.29165 +/- 0.08272 | 0.25054 +/- 0.00485 | 0.13820 +/- 0.01368 | -- | 0.37382 +/- 0.08634 | -0.08217 +/- 0.03399 |

The direct-U and ionic objectives replicate tightly, so the corrected global
readout and removal of the opposing sum gradient solve a real mechanism and
optimization failure.  The electronic loss remains nearly constant at
`0.29781 +/- 0.00049`, and unoptimized branch closure remains
`0.90548 +/- 0.09502`.  Conversely, the direct total regressor exceeds the
physical model's isolated macro tower in every seed.  This rejects a
total-tensor predictive advantage for the current system; it does not compare
away the atom-resolved ionic mechanism because total-only labels have no
gradient route into that tower.  Frozen test20 remains unread.

The canonical aggregate is
`outputs/global_l3_no_redundant_sum_multiseed_v1/report/validation_summary.json`.
Both dirty-worktree training states are protected by run-local SHA256 source
manifests.

## Engineering validation

- `153 passed` under `D:\Anaconda\envs\EGNN\python.exe` after CUDA
  vectorization and stream-pruning tests.
- The direct control ran on `cuda` on an NVIDIA RTX 4060 Ti; live samples were
  36--39% utilization and 1.31--1.48 GB allocated.  Zero utilization after the
  run reflects completed processes and released memory.
- Resume reconstructs one-, two-, or three-group AdamW topology and restores
  teacher-U moments without losing earlier CSV exposure counts.
- `summary.json` reconstructs factor, teacher-displacement, and joint updates
  from persisted metrics after an interrupted/resumed process.
- The paper compiles to 18 pages; rendered method, result, and experiment-ledger
  pages were visually checked for clipping, overflow, and unreadable tables.
