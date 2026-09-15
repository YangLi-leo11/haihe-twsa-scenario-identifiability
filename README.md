# Climate scenarios cannot uniquely determine century-scale terrestrial water storage

This repository contains custom code, compact processed inputs, and configuration
files supporting the principal computational analyses and reported numerical
results in the associated manuscript. It enables reproduction of the bundled
numerical checks and principal analytical figure workflows. Raw-data reruns require
the external datasets described below. Principal analytical figure workflows are
provided for Figures 2–6.

Development, exploratory, deprecated, manuscript-formatting,
submission-preparation, and unrelated research code are intentionally excluded.

## Main reproduced results

| Result | Value |
|---|---:|
| Strict recursive GRU, November 2020-October 2025 (n=60) | R = 0.691; RMSE = 54.000 mm; NSE = 0.286 |
| 84-month analogue median R | 0.710 |
| Residual-groundwater Pearson R | 0.861 |
| Groundwater moving-block bootstrap 95% interval | 0.759-0.921 |
| 2100 stationary-residual maximum inter-SSP separation | 6.76 mm |
| 2100 maximum within-SSP residual-pathway envelope | approximately 306 mm |

The validation starts from the observed October 2020 TWSA state and advances for 60
months without observed-state updating. The groundwater interval uses paired,
overlapping, non-circular six-month moving blocks, 2,000 replicates, seed 120 and
empirical 2.5th and 97.5th percentiles.

## Data access and redistribution

Raw groundwater records are not redistributed because of source-data restrictions.
GRACE/GRACE-FO and CMIP6 data must be obtained from their original providers.
HydroBASINS v1c Asia level 9 is an external dependency for rendering Figure 4 and
is not redistributed. Acquisition information and expected paths are provided in
`data/README.md` and `external_data_manifest.csv`.

## Installation and execution

```bash
conda env create -f environment.yml
conda activate cee-publication-code
python reproduce/run_core_analysis.py
python reproduce/run_figures.py
python -m unittest discover -s tests -v
```

`python scripts/run_all.py` runs the contained figure workflows, numerical checks
and regression tests. If the HydroBASINS boundary is absent, Figure 4 reports the
missing files and exits through the documented external-dependency path.

Figure 2 numerical results and publicly distributable source series are verified
from the bundled processed inputs. The complete figure is supplied as a reference
image because panels relying on restricted groundwater observations cannot be
regenerated from raw records in the public package. Pixel-level or binary-identical
rendering is environment-dependent; numerical values and plotted source data are
the primary reproduction checks.

## License

Original code and documentation in this repository are available under the MIT
License. The MIT License does not relicense GRACE/GRACE-FO, CMIP6, HydroBASINS, the
continuous TWSA product, restricted groundwater observations or other third-party
material. Those resources remain governed by their providers' terms.

## Authors and citation

The repository metadata lists Yang Li, Qi Liu, Xiaohui Wu, Qipei Pang, Sulan Liu,
Xihui Gu, and Yunlong Wu, in that order. Citation metadata are provided in
`CITATION.cff`.

The five residual pathways are deterministic conditional assumptions. Their ranges
are descriptive and are not confidence or prediction intervals. Groundwater is used
only as an independent consistency check and is not used to train or update the
projections.
