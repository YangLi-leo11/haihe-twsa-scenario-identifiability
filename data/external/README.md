# External data inputs

This directory is intentionally empty in the release. Do not commit credentials,
raw CMIP6/GRACE archives, or restricted groundwater observations here.

The spatial Fig. 4 renderer expects a HydroBASINS level-9 Haihe boundary named
`Haihe_basin_HydroBASINS_L9_boundary.shp` plus its sidecar files. The groundwater
workflow requires records subject to the originating monitoring authority's
data-use conditions, as described in `../../external_data_manifest.csv`.
Paths are resolved relative to this directory.
