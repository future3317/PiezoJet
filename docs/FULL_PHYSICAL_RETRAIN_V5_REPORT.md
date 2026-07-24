# Full physical retrain v5

This report records the completed single-seed validation replay. Historical
smoke runs and failed launches remain in the experiment ledger and are not
treated as current results.

- Split: canonical reduced-formula-safe `train1595/val10/test20`.
- Seed: 42; frozen `test20` was not read.
- Initialization: compatible Cartesian structural pretraining only.
- Stages: factor 5 epochs, teacher-$U$ 5 epochs, joint 10 epochs.
- Joint stream passes were complete; no update-capped multistream shortcut.
- Checkpoint: `outputs/full_physical_retrain_v5/loss_best.pt`.
- Evaluation: `outputs/full_physical_retrain_v5/validation10.json`.

| metric | value |
|---|---:|
| total signal-weighted relative error | 0.4781 |
| total response skill vs zero | 0.2712 |
| high-response directional cosine | 0.7990 |
| high-response amplitude ratio | 0.4882 |
| ionic macro cosine | -0.2858 |
| ionic amplitude ratio | 0.00285 |
| ionic stabilized relative error | 1.0019 |
| electronic-only response skill | 0.00006 |
| electronic-only amplitude ratio | 8.0e-7 |

The model improves total direction over the zero baseline on this small
validation panel, but ionic and electronic response amplitudes collapse. This
is a validation diagnostic, not a production or test-set claim.
