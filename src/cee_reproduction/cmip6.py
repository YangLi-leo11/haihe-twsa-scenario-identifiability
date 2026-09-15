"""Purpose-specific CMIP6 processing helpers."""

from __future__ import annotations
import numpy as np
import pandas as pd


def model_specific_spatial_delta(historical, future):
    """Route 1: mean(2081–2100 P-ET-Q) minus mean(1995–2014 P-ET-Q)."""
    return np.nanmean(future["pr"] - future["et"] - future["runoff"], axis=0) - np.nanmean(
        historical["pr"] - historical["et"] - historical["runoff"], axis=0
    )


def six_model_median_and_sign(model_deltas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(model_deltas, axis=0)
    matching = np.sum(np.sign(model_deltas) == np.sign(median)[None, ...], axis=0)
    return median, matching >= 5


def qdm_adjust(values, future_reference, historical_reference, *, multiplicative: bool):
    """Route 2 monthly QDM transformation for one variable and calendar month."""
    values = np.asarray(values, float)
    future_reference = np.asarray(future_reference, float)
    historical_reference = np.asarray(historical_reference, float)
    result = np.empty_like(values)
    for index, value in enumerate(values):
        quantile = (np.sum(values <= value) + 0.5) / (len(values) + 1.0)
        hq = np.quantile(historical_reference, np.clip(quantile, 0.001, 0.999))
        fq = np.quantile(future_reference, np.clip(quantile, 0.001, 0.999))
        result[index] = max(0.0, hq * value / max(fq, 1e-6)) if multiplicative else hq + value - fq
    return result


def cosine_latitude_weights(latitude) -> np.ndarray:
    return np.cos(np.deg2rad(np.asarray(latitude, dtype=float)))

