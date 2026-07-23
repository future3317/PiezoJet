# PiezoJet Stage-A training-efficiency review

## Repository access

The connected GitHub app has read access (`pull=true`) but no push/maintain permission.
The attached patch is therefore not pushed to the repository. It is based on the current
`main` implementation inspected on 2026-07-22.

## Highest-confidence findings

### 1. History/checkpoint growth is avoidable and likely contributes to slowdown

At every development evaluation the current runner stores full per-material development
and train metrics in `history`. Every later checkpoint then serializes the entire accumulated
history again. With three tasks and 988+800 materials, each evaluation adds roughly 5,364
per-material metric records. This makes checkpoint payload construction and serialization
grow with the number of evaluation points.

The patch keeps full per-material metrics in each immutable `update_XXXXXXXX.pt`, while
only compact aggregate rows accumulate in `history`, `training_curve.json`, console logs,
and later checkpoints.

### 2. The same checkpoint payload is serialized twice

The current runner performs `torch.save` once for the immutable update checkpoint and a
second time for `progress.pt`. The patch serializes once and atomically hard-links
`progress.pt` to the immutable checkpoint, with a byte-copy fallback.

### 3. Full train evaluation does not participate in selection

The three full train passes account for 2,400 of the 5,364 evaluation graph forwards.
The patch adds `--train-eval-interval`. Development evaluation and guardrails remain
unchanged at every eligible checkpoint. A recommended initial value is 250 updates.

Using the observed 162-second full evaluation, a rough proportional estimate is:

- development: 3 x 988 / 5,364 ≈ 55.3%, about 90 seconds;
- train diagnostic: 3 x 800 / 5,364 ≈ 44.7%, about 72 seconds.

Running the train diagnostic once every five development evaluations should therefore save
roughly 58 seconds per 50-update block on average, before other improvements.

### 4. Worker prefetch is ineffective with per-block DataLoader recreation

`loader_options` already supports persistent workers and prefetching, but the training
DataLoader is destroyed and recreated for every tower and every 50-update block. The patch
creates one deterministic schedule loader/iterator per task for the entire remaining run.
This preserves the exact material order and makes `num_workers>0`, persistent workers,
and `prefetch_factor` meaningful.

Recommended sweep on a fresh deterministic 20-update benchmark:

| workers | prefetch |
|---:|---:|
| 0 | 2 |
| 2 | 2 |
| 2 | 4 |
| 4 | 2 |

Compare material IDs, losses, gradients, first AdamW update, resume, wall time, host RAM,
and pinned-memory use.

### 5. Per-update CUDA event retention is unnecessary

The current runner creates and retains two CUDA events for every update of every tower.
The patch records one event pair per task block. This preserves aggregate GPU optimizer
time while reducing Python/CUDA runtime objects from O(updates) to O(evaluations).

### 6. Runtime geometry caches should be explicitly cleared before device swaps

`CrystalGlobalContext._geometry_cache` is a non-registered attribute, so `module.to("cpu")`
does not move it automatically. The patch clears these references before tower migration.
It also adds an opt-in flag to retain the CUDA allocator cache instead of calling
`torch.cuda.empty_cache()` after every task.

## Recommended command after equivalence checks

```bash
python -m piezojet.electrostatic_a0_fold_adjudication \
  ...existing scientific arguments... \
  --num-workers 2 \
  --prefetch-factor 4 \
  --train-eval-interval 250 \
  --retain-cuda-allocator-cache \
  --matmul-precision high
```

`--matmul-precision high` is intentionally not enabled by the patch. It must pass the
project's tolerance-equivalence and selection-stability checks first.

## Multi-GPU recommendation

A0 is unusually suitable for process-level parallelism because its three towers and three
AdamW states are parameter-disjoint. The clean design is:

1. one worker process per task/GPU;
2. the same persisted material schedule hash and common update checkpoints;
3. task-local model/optimizer checkpoints and full development metrics;
4. an offline or barrier-based coordinator that combines the three task metrics at common
   updates and applies the unchanged selection/guardrail rule.

Running all workers to 1,500 updates and aggregating offline is simpler and more robust than
synchronized early stopping. It should approach a 2.5–3x wall-clock reduction if each task
has a dedicated RTX 4090. This is a second patch because it changes orchestration and
checkpoint layout, although not the mathematical objective.

## What not to enable first

- `torch.compile` on the entire tower: the workload has ragged graph dimensions and Python
  loops in reciprocal geometry. Profile graph breaks and recompilations before promotion.
- CUDA graphs: fixed addresses and static shapes conflict with the current ragged batches.
- global BF16 autocast: high-order equivariant tensor products need explicit accuracy,
  equivariance, and gradient checks.
- training cost-aware reordering: moving materials between logical AdamW updates changes
  the optimization trajectory. Cost bucketing is safe for evaluation, or inside a fixed
  preregistered logical batch with tolerance checks.

## Next compute optimization after this patch

Profile one warm block with CPU and CUDA activities. If spherical harmonics/radial bases
and reciprocal geometry dominate, add a response-only static-geometry cache:

- precompute edge distances, `l<=3` spherical harmonics, and radial bases;
- key the cache by graph schema, cutoff, lmax, radial basis, dtype, and geometry hash;
- never use the static cache in coordinate-denoising or nonlinear geometry-Jacobian paths;
- require prediction, loss, gradient, update, and resume equivalence tests.

This is likely the largest remaining single-GPU optimization, but it is more invasive than
the attached runtime patch.

## Pretraining follow-up (2026-07-22)

The Stage-A A0 runner optimizations above were already present in
`electrostatic_a0_fold_adjudication.py`. The same runtime issues also affected the
response-only pretrainers and are now addressed in the maintained sources:

- `pretrain_electronic_e3nn.py` and `pretrain_bec_e3nn.py` expose
  `--prefetch-factor` and use one deterministic persistent-worker DataLoader for the
  complete resumed epoch range;
- the schedule batches are explicitly cut at epoch boundaries, so a physical batch
  can never mix two epochs or alter logical AdamW objectives;
- `cache_graphs=False` is used for response pretraining, preventing each worker from
  retaining hundreds of ragged graphs in its private process memory;
- best checkpoints are filesystem hard-links to the already serialized `last` payload,
  with a copy fallback, rather than a second `torch.save`;
- an optional, manifest-checked `--graph-cache-key` reuses an existing canonical graph
  cache and avoids re-hashing the full corpus at startup.

The default production command remains conservative (`--num-workers 0`); worker
prefetch is a separate runtime experiment and must be compared by loss trajectory,
schedule/resume equivalence, wall time, and host memory before enabling it for a
registered result. The 2026-07-22 full local suite passed after these changes.

## Maintained-path extension (2026-07-23)

The same bounded runtime changes now cover the remaining maintained training
entry points. `pretrain_e3nn.py` and the Cartesian structure pretrainer accept
`--prefetch-factor` through the common `loader_options` helper, precompute their
trainable parameter lists, and use one serialized `last` checkpoint plus an
atomic hard-link/copy for `best`. The direct baseline and the full factor/
displacement trainer use the same loader helper and checkpoint link operation.
The shared electrostatic fold runner no longer serializes the identical
evaluation payload twice for its progress pointer. These changes are runtime
only: they preserve material order, logical batch boundaries, optimizer
updates, validation selection, and checkpoint payload contents.

For the upcoming three-fold single-seed gate, use the already validated A0
resource-bounded runner with `num_workers=0` as the reproducible baseline. A
worker/prefetch speed comparison may be run separately; it must not silently
become the registered result until loss/gradient/resume equivalence and host
memory are recorded.
