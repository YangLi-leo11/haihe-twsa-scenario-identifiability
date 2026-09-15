"""Independent groundwater-consistency diagnostics.

Raw groundwater observations are intentionally absent. The functions operate
only when an authorized local file is supplied by the user.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def storage_proxy(source_values: pd.Series) -> pd.Series:
    return -pd.to_numeric(source_values, errors="coerce")


def zscore(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    return (values - values.mean()) / values.std(ddof=0)


def moving_block_bootstrap_correlation(
    x: pd.Series,
    y: pd.Series,
    *,
    block_length: int = 6,
    repetitions: int = 2000,
    seed: int = 120,
) -> np.ndarray:
    """Paired, overlapping, non-circular moving-block bootstrap.

    This is the implementation that reproduces the manuscript interval from the
    included 60-month pair: blocks may start at 0..n-block_length and therefore never
    wrap December 2023 back to January 2019. The result uses the empirical 2.5th and
    97.5th percentiles (NumPy's default linear quantile interpolation).
    """
    paired = pd.DataFrame({"x": x, "y": y}).dropna()
    x0, y0 = paired["x"].to_numpy(float), paired["y"].to_numpy(float)
    rng = np.random.default_rng(seed)
    n = len(paired)
    correlations = []
    for _ in range(repetitions):
        indices = []
        while len(indices) < n:
            start = int(rng.integers(0, n - block_length + 1))
            indices.extend(start + np.arange(block_length))
        selected = np.asarray(indices[:n])
        correlations.append(stats.pearsonr(x0[selected], y0[selected]).statistic)
    return np.asarray(correlations)
