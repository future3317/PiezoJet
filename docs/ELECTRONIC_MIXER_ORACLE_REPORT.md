# Global `l=1` multiplicity-mixer oracle

This is a development-only, read-only oracle audit for the electronic A0-PM
generator. It fits a shared 2x2 scalar map between the two `l=1` multiplicity
copies in orthonormal irrep coordinates. The map is never inserted into the
model, loss, checkpoint selection, or production inference.

Artifacts:

- `outputs/electronic_mixer_oracle_fold0_seed42_v1/calibration64_audit64.json`
- `outputs/electronic_mixer_oracle_fold0_seed42_v1/development988_calibration64.json`

On the disjoint calibration64/audit64 slice:

| variant | active relative error | active cosine |
|---|---:|---:|
| baseline | 0.92815 | 0.32924 |
| unconstrained 2x2 | 0.94114 | 0.28864 |
| diagonal-only | available in JSON | available in JSON |
| orthogonal-polar 2x2 | 0.92201 | 0.34411 |

The complete development report (988 records, with the first 64 used only for
calibration and 924 for audit) gives:

| variant | active relative error | active cosine |
|---|---:|---:|
| baseline | 0.87115 | 0.43404 |
| unconstrained 2x2 | 1.00273 | 0.41004 |
| orthogonal-polar 2x2 | 0.87138 | 0.43913 |

The preregistered gate required at least 0.10 cosine improvement and 0.03
relative-error reduction on the audit slice. The mixer fails decisively, so M1
is rejected before same-ID capacity training. This does not support a global
material-independent mixer; it leaves representation/OOD conditioning as the
next hypothesis. Frozen validation10/test20 were not read.
