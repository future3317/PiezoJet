# PiezoJet

PiezoJet learns an O(3)-equivariant piezoelectric tensor as the mixed derivative of one scalar response potential, `Phi(x, E, eta)`. The MVP uses only GMTNet's released JARVIS-DFT piezoelectric data; it does not silently substitute another source.

## Reproduce

```bash
python -m pip install -e .
python scripts/download_data.py --output data/raw/gmtnet
python scripts/inspect_data.py --root data/raw/gmtnet
pytest -q
python -m piezojet.train --config config.yaml --loss full --overfit-32
python scripts/run_pipeline.py --config config.yaml --loss full
python scripts/run_pipeline.py --config config.yaml --loss sketch
python -m piezojet.evaluate --checkpoint outputs/best.pt --split test
```

The production pipeline always performs masked-species and translation-free
coordinate-denoising structural pretraining before piezoelectric fine-tuning.
The response model combines local equivariant polar motifs with their periodic
reciprocal-space coherence spectrum and continuous lattice-frame synthesis.

`inspect_data.py` is a required gate: it records the raw fields, units, source Voigt order, split status, finite-value check, and atom counts before training.
When data is manually copied rather than cloned by `download_data.py`, create `data/raw/gmtnet/SOURCE_COMMIT.txt` containing the exact 40-character GMTNet commit SHA. Training refuses to start without it, so results remain reproducible.

## Conventions

GMTNet labels are `piezoelectric_C_m2` with source columns `[xx, yy, zz, xy, yz, xz]`. PiezoJet converts them once at ingestion to `[xx, yy, zz, yz, xz, xy]`, using engineering shear strain `[exx, eyy, ezz, 2eyz, 2exz, 2exy]`. Internally the tensor is `e_ijk=e_ikj`, represented through `e3nn.io.CartesianTensor("ijk=ikj")` (18 dimensions).

The GMTNet Piezo block obtains its source columns by differentiating with respect to one entry of a symmetric Cartesian strain matrix. Therefore the source shear coefficient is `e_ij`, not `e_ij/2`; PiezoJet stores both Cartesian entries as `e_ij` and puts the factor of two only in the engineering-strain conversion. This is enforced by `test_engineering_shear_matches_single_symmetric_component_derivative`.

The response potential is exactly `Phi=-E_i e_ijk eta_jk`. `--loss full` fits Cartesian tensor MSE; `--loss sketch` applies one Gaussian mixed-Hessian JVP projection per sample; `--loss hybrid` combines them. Outputs contain only `best.pt`, `last.pt`, resolved config, metric history, and a summary.
