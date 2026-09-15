# Reproduction guide

## Contained checks

Run from the repository root:

```bash
python reproduce/run_core_analysis.py
python reproduce/run_figures.py
python -m unittest discover -s tests -v
```

The core-analysis entry point recomputes the reported validation metrics, the
84-month analogue statistic, groundwater summary checks and the 2100 separation and
pathway-envelope values from the bundled processed tables. The figure entry point
generates Figures 3, 5 and 6, verifies Figure 2's publicly distributable source
series and numerical results, and materializes the included Figure 2 reference
image. Figure 2 panels that depend on restricted groundwater observations cannot be
regenerated from raw records in this public package.

Pixel-level or binary-identical rendering can vary with the operating system,
Matplotlib, FreeType, fonts and backend. Reference-image hashes are therefore an
environment diagnostic rather than a reproduction failure gate. Numerical values
and plotted source data provide the primary checks.

## Figure 4

Figure 4 uses included numerical tables but requires an external HydroBASINS v1c
Asia level-9 boundary for map rendering. Follow `data/README.md` to prepare the five
shapefile components. The script reports each missing file explicitly.

## Expected outputs

- `figures/reproduced/Fig2_CEE_harmonized_v3.png`
- `figures/reproduced/Fig3_CEE_harmonized_v2.*`
- `figures/reproduced/Fig5_CEE_harmonized_v2.*`
- `figures/reproduced/Fig6_CEE_harmonized_v2.*`
- `outputs/reproduction_summary.csv`
- `outputs/figure_checks.csv`

Raw-data preprocessing requires provider-supplied GRACE/GRACE-FO and CMIP6 archives.
Groundwater reruns require access governed by the originating monitoring authority;
the public package provides only the reported summary values and bootstrap
specification.
