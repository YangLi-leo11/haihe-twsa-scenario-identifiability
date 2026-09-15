"""Residual storage state defined in monthly-change space."""

from __future__ import annotations
import pandas as pd


def construct_residual_storage_state(
    dates,
    continuous_reference_twsa,
    climate_sensitive_component,
    center_start: str = "2019-01-01",
    center_end: str = "2023-12-01",
) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "continuous_reference_twsa": continuous_reference_twsa,
            "climate_sensitive_component": climate_sensitive_component,
        }
    ).sort_values("date")
    result["continuous_reference_monthly_change"] = result[
        "continuous_reference_twsa"
    ].diff()
    result["climate_sensitive_monthly_change"] = result[
        "climate_sensitive_component"
    ].diff()
    result["monthly_change_residual"] = (
        result["continuous_reference_monthly_change"]
        - result["climate_sensitive_monthly_change"]
    )
    result["residual_storage_state_raw"] = result["monthly_change_residual"].fillna(0).cumsum()
    overlap = result["date"].between(center_start, center_end)
    result["residual_storage_state"] = (
        result["residual_storage_state_raw"]
        - result.loc[overlap, "residual_storage_state_raw"].mean()
    )
    return result

