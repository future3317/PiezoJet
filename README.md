# PiezoJet

PiezoJet learns an O(3)-equivariant piezoelectric tensor as the mixed derivative of one scalar response potential, `Phi(x, E, eta)`. The MVP uses only GMTNet's released JARVIS-DFT piezoelectric data; it does not silently substitute another source.

## Reproduce

```bash
python -m pip install -e .
python scripts/download_data.py --output data/raw/gmtnet
python scripts/inspect_data.py --root data/raw/gmtnet
pytest -q
# Public, same-source DFPT labels; use --limit first to verify connectivity.
python -m piezojet.jarvis_dfpt --data-root data/raw/gmtnet --output-dir data/processed/jarvis_dfpt_v1 --limit 1
python -m piezojet.train --config config.yaml --loss full --overfit-32
python scripts/run_pipeline.py --config config.yaml --loss full
python scripts/run_pipeline.py --config config.yaml --loss sketch
python -m piezojet.evaluate --checkpoint outputs/best.pt --split test
# Audited physical-unit metrics on the formula-disjoint DFPT subset.
python -m piezojet.evaluate_dfpt --checkpoint outputs/best.pt --split test
```

The production pipeline always performs masked-species and translation-free
coordinate-denoising structural pretraining before piezoelectric fine-tuning.
The local encoder is a channelized Cartesian periodic environment: an invariant
chemical backbone weights vector and traceless-quadrupole edge bases, then a
learned channel-space interaction matrix forms local many-body modes without
Clebsch--Gordan feature propagation. This is an efficient backbone rather than
the paper's central novelty. The response model combines local equivariant polar
motifs with a physical reciprocal-vector shell, an origin-invariant polar--chemical
cross-spectrum, and PBC radial fluctuation correlations. The cross-spectrum
directly constructs a rank-three polar correction, so global information affects
tensor direction as well as invariant amplitudes. Six-frame polar-decomposition
synthesis has been removed from the production path.

The Cartesian encoder has a new pretraining checkpoint lineage:
`outputs/pretrain_cartesian/best_encoder.pt`. Earlier spherical-encoder
checkpoints are intentionally rejected instead of being silently misloaded.

`inspect_data.py` is a required gate: it records the raw fields, units, source Voigt order, split status, finite-value check, and atom counts before training. The production split is formula-grouped and stratified by the rotation-invariant piezoelectric tensor norm, preventing formula leakage while preserving zero, weak, moderate, high, and very-high response populations in every partition. Source labels are Reynolds-projected to each structure's point group once and cached under `data/processed/piezo_symmetry_targets_v1/`.
When data is manually copied rather than cloned by `download_data.py`, create `data/raw/gmtnet/SOURCE_COMMIT.txt` containing the exact 40-character GMTNet commit SHA. Training refuses to start without it, so results remain reproducible.

## Conventions

GMTNet labels are `piezoelectric_C_m2` with source columns `[xx, yy, zz, xy, yz, xz]`. PiezoJet converts them once at ingestion to `[xx, yy, zz, yz, xz, xy]`, using engineering shear strain `[exx, eyy, ezz, 2eyz, 2exz, 2exy]`. Internally the tensor is `e_ijk=e_ikj`, represented through `e3nn.io.CartesianTensor("ijk=ikj")` (18 dimensions).

The GMTNet Piezo block obtains its source columns by differentiating with respect to one entry of a symmetric Cartesian strain matrix. Therefore the source shear coefficient is `e_ij`, not `e_ij/2`; PiezoJet stores both Cartesian entries as `e_ij` and puts the factor of two only in the engineering-strain conversion. This is enforced by `test_engineering_shear_matches_single_symmetric_component_derivative`.

The response model now works in the physical atom-coordinate space. It predicts
node-aligned Born tensors `Z*`, a variable-size force-constant Hessian `Phi`,
and node-aligned strain forces `Lambda`. An exact Cartesian projector removes
the three uniform translations for every crystal, after which the optical
solve yields `e_ion = (16.02176634 / volume) Z* Phi_opt^-1 Lambda` in C/m2.
There is no fixed mode count and no phonon padding. The force-constant head
enforces block-transpose symmetry and the acoustic sum rule while retaining
real negative/unstable DFPT modes. A signed damped pseudoinverse
`lambda/(lambda^2+delta^2)` makes soft-mode crossings finite without changing
their sign; BEC and internal strain obey their corresponding zero-sum
constraints. The same relaxation also supplies lattice dielectric and elastic
corrections with explicit unit conversion. GMTNet's summary records do not embed BEC or phonon arrays,
but every current `JVASP-*` label has a matching public JARVIS `raw_files`
DFPT archive. `piezojet.jarvis_dfpt` downloads, parses, and structure-validates
those archives into atomic BEC targets, the raw ionic piezoelectric tensor,
and cached Gamma-point dynamical eigenpairs, force constants, and
symmetry-inequivalent internal-strain blocks. Training directly supervises BEC,
the cleaned physical Hessian, only the internal-strain blocks actually printed
by VASP, and the complete ionic response product. It does not fabricate a
symmetry expansion whose VASP convention has not been numerically validated.
Variable-length phonon arrays are retained without padding for audits. The parser deliberately
stores VASP dynamical eigenvalues rather than treating potentially repeated
`OUTCAR` blocks as a frequency label.

`piezojet.evaluate_dfpt` is deliberately separate from the full-dataset tensor
evaluator. It intersects the persisted audited IDs with the formula-disjoint
split and reports BEC, force-constant, printed internal-strain, ionic-piezo,
electronic-piezo, total-piezo, and dielectric errors in physical units, with
zero-predictor skills and per-material CSV rows. The optional direct-factor
curriculum is enabled with `--factor-pretrain-epochs`; its selected factor
stack can be protected during a response ablation with
`--freeze-factors-during-joint`. Both are off by default because the current
128-material experiment improves BEC/force prediction but does not improve the
predeclared total-response skill metric.

`--loss full` is the production balanced robust tensor objective. `--loss
sketch` differentiates the relaxed potential with JVPs and is retained as a
diagnostic rather than claimed as a speed or memory improvement for this
18-dimensional output.
