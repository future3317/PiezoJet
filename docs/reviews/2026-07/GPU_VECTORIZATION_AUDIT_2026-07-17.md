# GPU vectorization audit — 2026-07-17

## Scope

This audit addresses low and intermittent CUDA utilization in the maintained
`global_l3` direct-U model.  It does not change `num_workers`, which remains
zero, and it does not read the frozen test20 panel.

The seed-1729 process launched before this refactor retains the Python objects
loaded from its run-local source manifest.  The measurements below were made
concurrently on the same RTX 4060 Ti and are therefore implementation
diagnostics, not clean production throughput claims.

## Implemented changes

1. The per-material nonlocal U-head attention loop is replaced by one padded,
   mask-exact batched attention calculation.  Padding cannot contribute to the
   key softmax.
2. The direct ionic contraction `Z*^T U` is expressed as a node-wise `einsum`
   followed by a graph `scatter` sum.
3. Branch and strict training streams omit the independent macro tower and the
   propagated `Phi/Lambda` response when no active objective consumes those
   outputs.  Validation still evaluates the complete diagnostic response.
4. Inactive losses use a parameter-independent scalar zero.  This prevents an
   inactive tower from receiving an explicit zero gradient and hence prevents
   AdamW weight decay from silently updating that tower on the wrong stream.

## Numerical checks

- Vectorized attention versus the previous loop: forward output and gradients
  agree within the test's FP32 tolerances.
- Vectorized `Z*^T U` versus the previous loop: forward output and gradients
  agree within FP32 tolerances.
- Pruned versus complete forward: all maintained physical outputs agree; the
  expensive disabled methods are monkey-patched to fail if called.
- A strict-like U/Lambda objective leaves macro, electronic, background, and
  BEC towers absent from autograd.
- Full repository suite after the implementation: 153 tests passed.

## Concurrent CUDA microbenchmark

Environment: NVIDIA RTX 4060 Ti 16 GB, PyTorch 2.12.0+cu126, EGNN conda
environment.  Values are medians; seed 1729 was training concurrently.

| Kernel | Previous | Vectorized | Speedup |
|---|---:|---:|---:|
| Global attention, 128 graphs / 1,282 nodes | 77.625 ms | 3.932 ms | 19.74x |
| `Z*^T U`, 128 graphs / 1,282 nodes | 77.634 ms | 0.838 ms | 92.68x |

The synthetic attention dimensions match the production configuration:
48 Cartesian channels, scalar dimension 64, attention dimension 64, and
cross-rank 24.  Graph sizes were uniformly sampled from 4--16 nodes.

## Concurrent end-to-end diagnostic

A 32-graph/315-node actual JARVIS geometry batch used the production three
block model.  A forward/backward step on the same active physical objective
took 789.045 ms with complete diagnostic propagation and 735.305 ms with the
training-stream pruning, a conservative 1.073x speedup.  The physical-output
maximum absolute difference was `5.78e-8`, and the active loss difference was
exactly zero at printed precision.

This comparison already uses vectorized attention on both sides, so it does
not include the large loop-to-batch gain in the first table.  A clean
single-process epoch benchmark should be run after the old seed-1729 process
finishes before quoting an overall training speedup.
