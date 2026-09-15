# Data layout and external inputs

`processed/` contains compact, author-generated tables included in this release.
Large upstream archives and restricted groundwater observations are excluded.
Groundwater monitoring records remain subject to the data-use conditions of the originating monitoring authority. Contact the corresponding author regarding access conditions; any access or redistribution remains subject to the originating authority's restrictions and approval, where applicable.
`external/` is the expected location for user-supplied licensed inputs.

The repository's MIT License applies to original repository code and documentation.
It does not relicense GRACE/GRACE-FO, CMIP6, HydroBASINS, the continuous TWSA product
or restricted groundwater observations. Each external dataset remains subject to its
provider's license, citation and access requirements.

## HydroBASINS boundary required by Fig. 4

Fig. 4 uses **HydroBASINS version 1c, Asia, level 9, with inserted lakes**, whose
upstream distribution layer is `hybas_lake_as_lev09_v1c`. HydroBASINS is maintained
by HydroSHEDS. Download the Asia level-9 shapefile from the official
[HydroBASINS product page](https://www.hydrosheds.org/products/hydrobasins) and
consult the [version 1c technical documentation](https://data.hydrosheds.org/file/technical-documentation/HydroBASINS_TechDoc_v1c.pdf).

The manuscript renderer needs the Haihe subset dissolved to one EPSG:4326 boundary.
Supply the following five-file shapefile bundle without committing it:

```text
data/external/Haihe_basin_HydroBASINS_L9_boundary.shp
data/external/Haihe_basin_HydroBASINS_L9_boundary.shx
data/external/Haihe_basin_HydroBASINS_L9_boundary.dbf
data/external/Haihe_basin_HydroBASINS_L9_boundary.prj
data/external/Haihe_basin_HydroBASINS_L9_boundary.cpg
```

The expected `.shp` fingerprint is
`f47dca4b19ceede39d473547cc44edc56dea56e17601410848f1c1ca6452cf84`.
The boundary is not redistributed here; users must obtain HydroBASINS and comply
with the HydroSHEDS license and attribution terms. `scripts/figure4.py` lists every
missing component and exits with status 2 before plotting.

Other external inputs and restrictions are recorded in
`../external_data_manifest.csv` and `../data_manifest.csv`.
