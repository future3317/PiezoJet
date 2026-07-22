# PiezoJet training-efficiency review brief (2026-07-22)

## Request

Please review this implementation for **correctness-preserving speed
improvements**. Do not propose data leakage, a silent change to development
selection, or a change to the material-mean logical objective.

## Current experiment

- Server: six RTX 4090 GPUs (24 GiB), PyTorch + PyTorch Geometric + e3nn.
- Active physical device: **GPU 5**, selected by `CUDA_VISIBLE_DEVICES=5`.
  At observation: 4.3 / 24.6 GiB allocated, 62% sampled GPU utilization, P2.
  The run is not memory-bound and is not saturating the accelerator.
- Runner: `piezojet.electrostatic_a0_fold_adjudication`, architecture
  `a0_parameter_matched_irreps`, fold 0, N=800 response-training materials,
  seed 42, 1,500 maximum updates.
- Development panel: 988 formula-disjoint materials. Frozen `validation10`
  and `test20` are not loaded by this runner.
- Logical batch = physical microbatch = 16 graphs; development/train-eval
  batch = 64; `num_workers=0`; full evaluation every 50 updates; early-stop
  patience = four eligible development evaluations.
- A BEC tower was response-pretrained for 20 epochs on 3,951 fold-train-only
  BEC records. Only that A0-PM tower receives its checkpoint. A compatible
  fold-train-only structural checkpoint is reused, not re-trained.

## Model and objective

The task predicts three independently supervised, availability-valid JARVIS
DFPT responses: per-atom Born effective charge `Z*` (3x3), electronic
piezoelectric tensor (Cartesian rank 3), and electronic dielectric tensor
(symmetric rank 2). A0-PM deliberately contains three parameter-disjoint
`ElectromechanicalJetHead` instances (`born_generator`, `piezo_generator`,
`dielectric_generator`), each with an O(3)-equivariant periodic graph encoder
and task-specific irrep readout. Width is 0.56 and total parameters are 6.36M.
There are three independent AdamW states and no shared trainable tensors: this
is a controlled test of whether hard multitask sharing damages learning.

Definitions: `src/piezojet/model.py:IndependentElectrostaticHeads`; runner:
`src/piezojet/electrostatic_a0_fold_adjudication.py`.

## Exact execution path

For each 50-update block, the runner does the following **serially for each
of the three towers**:

1. Move tower and its AdamW state to GPU.
2. Create a fresh PyG `DataLoader` for the deterministic 800 graph occurrences
   scheduled in that block.
3. Execute 50 forward/backward/AdamW updates (one 16-graph microbatch per
   logical update).
4. Traverse all 988 development graphs to compute full per-material metrics.
5. Traverse all 800 training graphs again to report the train--development gap.
6. Move tower and optimizer to CPU; call `torch.cuda.empty_cache()`.

Thus each checkpoint evaluates 3 x (988 + 800) graph forwards, in addition to
3 x 50 training updates. The full development score is the sum of stabilized
material-relative electronic-piezo, BEC, and dielectric errors. The full train
evaluation is diagnostic only; it never selects checkpoints.

The DataLoader helper sets `pin_memory=True` and non-blocking H2D copies on
CUDA. With `num_workers=0`, it has no persistent workers or prefetching.
Graphs are locally cached but ragged, with variable atom and edge counts. The
evaluator stores predictions on CPU, computes material-level metrics, retains
per-material outputs, and emits a full JSON row at every evaluation.

## Measured wall-clock evidence

From the active run. Evaluation total includes the three complete development
passes plus three complete train diagnostic passes. Block wall is the whole
50-update block.

| Update | Dev. score | Evaluation total | Complete 50-update block |
|---:|---:|---:|---:|
| 50 | 1.574 (ineligible: electronic amplitude) | 76 s | 310 s |
| 400 | 1.431 | 93 s | 447 s |
| 500 | 1.398 | 123 s | 559 s |
| 650 | 1.379 | 149 s | 670 s |
| 800 (best so far) | **1.356** | 157 s | 730 s |
| 850 | 1.376 | 162 s | 747 s |

At update 850 the process was still running. The large and growing block time
is real: evaluation is only roughly 20--25% of the newest block. The remaining
training time is also slow and variable, plausibly from ragged graph cost,
CPU-side batching/cache reads, and serial tower execution. This has not yet
been isolated by a profiler.

## Learning status (not final)

At update 850: electronic stabilized relative error 0.511; active electronic
relative error 0.994, cosine 0.177, amplitude ratio 0.086 (passes the >=0.05
guardrail); BEC stabilized relative error 0.394; score 1.376. The best so far
is 1.356 at update 800. This is development-only, single-fold, single-seed
evidence. It is promising relative to the old no-BEC-pretraining A0-PM score
1.587, but requires completion and paired multi-seed replication.

## Hard constraints

1. Keep formula disjointness and never read frozen val10/test20.
2. Preserve the material-mean logical loss and one AdamW update per logical
   batch. Microbatch changes may be tolerance-equivalent, not bitwise exact.
3. A0 towers must remain parameter-disjoint. Multi-GPU execution is acceptable
   only with explicit deterministic schedules, optimizer updates, metric
   aggregation, and checkpoint provenance.
4. Full-development evaluation and its guardrails remain mandatory whenever a
   checkpoint is eligible for selection.
5. Any optimization needs a small deterministic equivalence test (fixed
   schedule/initial state; loss, gradient, update, and resume checks).

## Questions for a reviewer

1. Give a profiling plan separating CUDA kernels, H2D copies, PyG collation,
   graph-cache reads, CPU metric work, JSON serialization, and model/optimizer
   transfers.
2. Can workers, persistent workers, and pinned-memory prefetching safely be
   enabled without altering sample order/objective? Recommend a small sweep.
3. Is it sound to compute full train-gap diagnostics less often, or stream them
   from training, while retaining full-development selection? Estimate savings.
4. Can the three independent towers run concurrently on three GPUs with a
   simple, robust checkpoint/process design?
5. Would atom/edge-cost-aware batching improve ragged-graph throughput without
   compromising the logical objective? State necessary equivalence tests.
6. Assess `torch.compile`, CUDA graphs, AMP/bfloat16, TF32, fused AdamW, and
   PyG/e3nn kernel upgrades for this dynamic-shape equivariant workload. Rank
   suggestions by expected gain, risk, and protocol compatibility.
