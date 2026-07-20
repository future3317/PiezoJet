# Interrupted fresh structural-pretraining attempt

- Started: 2026-07-19 01:02:40 Asia/Shanghai
- Stopped by explicit user request after approximately 406 seconds.
- Command used physical batch 16 and no logical-batch accumulation.
- GPU allocation reached approximately 16,048/16,380 MiB.
- The first exposure epoch did not complete.
- No encoder checkpoint, optimizer state, history, or performance result was written.
- This directory is immutable interruption evidence and must never be resumed or reused.

The replacement protocol uses a new cohort, physical batch 4, logical batch
32, fresh initialization, stabilized three-task selection with explicit
guardrails, and pinned code/data/graph provenance.
