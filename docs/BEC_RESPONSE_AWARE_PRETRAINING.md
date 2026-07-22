# BEC response-aware initialization gate

## Decision boundary

The completed Stage-A gate is recorded in
`outputs/vnext_stage_a_hierarchical_fairness_server_v1_correct/ARCHITECTURE_GATE_DIAGNOSIS.json`.
It is fold 0, seed 42, formula-disjoint development-only evidence on the fixed
800-material response subset. No frozen validation10 or test20 label was read.

| model | parameters | selected update | stabilized development score | BEC stabilized error |
|---|---:|---:|---:|---:|
| A0-full independent | 19.30M | 1200 | 1.61302 | 0.65336 |
| A0-PM independent | 6.358M | 1200 | **1.58657** | **0.64019** |
| A1 hard shared | 6.454M | 1000 | 1.73414 | 0.75302 |
| A1.6 hierarchical | 6.674M | 800 | 1.72749 | 0.76495 |

The parameter-matched independent control wins. Its electronic error is nearly
tied with A1; the largest deficit of both shared candidates is BEC. The next
test should therefore improve BEC sample efficiency, not add PCGrad, GradNorm,
a new sharing topology, or a new physical response identity.

## Candidate and scientific scope

`piezojet.pretrain_bec_e3nn` performs **supervised BEC response-aware
initialization**. It trains the existing A0-PM BEC tower on source BEC labels
from the complete fold-training side (3,951 records in fold 0), then the
standard A0-PM runner loads only that tower. It does not modify:

- the first-order electrostatic response definition;
- the acoustic sum-rule projection in the BEC decoder;
- electronic-piezoelectric or electronic-dielectric towers;
- the downstream three-task loss, stabilized selection metric, or guardrails;
- frozen validation10/test20 access policy.

This deliberately tests a data/initialization hypothesis: whether abundant,
same-source BEC supervision improves the identified BEC bottleneck when the
downstream multi-response panel remains fixed at `N=800`. A positive single
seed is not a promotion; it must be retained with the same diagnostics and
then independently replicated.

## Provenance and failure policy

The BEC checkpoint contains a strict contract:

- architecture must be `a0_parameter_matched_irreps` with its exact width;
- source IDs are a subset of the current fold-train IDs and have a SHA256;
- the complete development-ID SHA256 is stored, with zero reduced-formula
  overlap;
- canonical data manifest, GMTNet data commit, fold identity, BEC-only task,
  optimizer state, and exact `born_generator` state are saved;
- resume accepts only an exact contract and provenance match;
- downstream loading rejects every mismatch and writes no fallback checkpoint.

The downstream argument `--bec-pretrained-tower` is intentionally unavailable
to A0-full, A1, A1.5, and A1.6. It overwrites `born_generator` only after the
ordinary fold-train-only structural initializer has populated all three A0-PM
towers.

## Required execution order

1. Run local tests, Ruff, and bytecode compilation.
2. Run a fresh two-epoch server smoke BEC pretrain and a two-update downstream
   smoke. Verify `last` and `best` checkpoints, strict reload, saved provenance,
   and `frozen_validation_test_labels_read: false`.
3. Run one fold-0/seed-42 BEC pretrain on the full fold-training BEC panel.
4. Run one fold-0/seed-42 fixed-`N=800` A0-PM downstream comparison under the
   existing 1,500-update/guardrail-aware selection protocol.
5. Preserve every evaluation and report the three stabilized components,
   direction/amplitude guardrails, train--development gap, parameter count,
   throughput, and peak memory. Do not inspect frozen panels to select it.

The implementation is a narrow response-aware transfer gate. A negative result
rules out this particular allocation of BEC labels; it does not establish that
BEC is irreducibly unlearnable or that the tensor conventions are wrong.
