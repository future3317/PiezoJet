# Production-path migration: factor/Schur response

Approved on 2026-07-24. The maintained production ionic piezoelectric path is

\[
e^{\rm phys}=e^{\rm el}+\frac{c_e}{\Omega}Z^{*\mathsf T}
\mathcal D_\delta(\Phi)\Lambda,
\qquad
\mathcal D_\delta(\Phi)=Q\Phi_o(\Phi_o^2+\delta^2I)^{-1}Q^\mathsf T.
\]

This is the real Tikhonov solution of
\(\min_U \frac12\|\Phi_oU-r\|_F^2+\frac{\delta^2}{2}\|U\|_F^2\), evaluated in
the translation-free optical basis. Negative eigenvalues retain their sign;
there is no predicted-spectrum hard branch.

The independently supervised direct-\(U\) head remains available as an
amortized-solver diagnostic. It is not combined with factor-derived
elastic/dielectric outputs as a second production path.

## Provenance and result status

The migration is implemented in commit `4117e28`, based on the approved server
candidate `1585a93`. Existing direct-\(U\)-production checkpoints and their
validation numbers are historical diagnostics and are not comparable to the
new production semantics. A fresh response training run and validation are
required before making accuracy claims for this path. Frozen `test20` remains
unread.

The migration was verified with the focused tensor/operator suite:
22 tests passed. A full-suite run exceeded the local two-minute smoke budget;
it was stopped and must be rerun as a separate, explicitly budgeted check.
