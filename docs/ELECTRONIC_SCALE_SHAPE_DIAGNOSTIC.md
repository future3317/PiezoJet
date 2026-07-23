# Electronic amplitude / scale--shape diagnostic

This is a development-only audit of the selected single-seed A0-PM checkpoint;
it is not a new loss, a checkpoint-selection rule, or a production fallback.
The diagnostic uses 128 materials from electrostatic development fold 0, with
64 materials used only to fit calibration scalars and a disjoint 64-material
slice used for the reported audit. Frozen validation10/test20 labels are not
read.

The audit compares the unmodified electronic prediction with (i) one global
least-squares scalar, (ii) independent least-squares scalars for the two
`l=1` copies, `l=2`, and `l=3` irrep blocks, and (iii) an oracle that rescales
each audit material to its true norm. The last quantity is a shape-only upper
bound and is intentionally non-deployable because it uses the audit target
norm.

## Result

Artifact: `outputs/electronic_response_pretraining_a0pm_fold0_seed42_v1/`
`electronic_scale_shape_diagnostic_v1/diagnostic.json`.

| audit variant | stabilized relative | active relative | active cosine | active amplitude |
|---|---:|---:|---:|---:|
| raw A0-PM | 0.5992 | 0.9281 | 0.3292 | 0.2188 |
| one global scalar (`alpha=1.2641`) | 0.6037 | 0.9212 | 0.3292 | 0.2766 |
| per-irrep scalar | 0.6234 | 0.9294 | 0.2723 | 0.2916 |
| oracle per-material norm | 0.6358 | 0.9957 | 0.3292 | 1.0000 |

The calibration-slice irrep scalars were `3.9630`, `0.1144`, `0.6557`, and
`1.2622` for `l1_copy0`, `l1_copy1`, `l2`, and `l3`, respectively. Their large
and opposing `l=1` values are evidence of a shape/copy-direction problem, not
one uniform amplitude factor. The global scalar only raises amplitude and
barely changes the direction; per-irrep calibration worsens the held-out
directional score. The oracle norm experiment also leaves the active cosine at
0.3292, so amplitude is not the dominant error on this slice.

This result rules out a simple post-hoc scalar as the next production change.
The next small hypothesis, if pursued, should target the two `l=1` response
copies and their cross-material shape/OOD behavior; any learned scale--shape
head must be tested with a new disjoint development calibration protocol and
must not use target norms at inference.
