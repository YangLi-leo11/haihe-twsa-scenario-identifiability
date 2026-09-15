"""Climate-storage risk index used in the manuscript."""

from __future__ import annotations
import numpy as np
import pandas as pd


def trailing_water_balance(frame: pd.DataFrame) -> pd.Series:
    monthly = frame["pr"] - frame["et"] - frame["runoff"]
    return monthly.rolling(12, min_periods=3).mean()


def historical_scaling(historical: pd.DataFrame) -> dict[str, float]:
    balance = trailing_water_balance(historical)
    threshold = float(balance.mean())
    deficit = np.maximum(0.0, threshold - balance)
    cumulative = pd.Series(deficit, index=historical.index).cumsum()
    return {
        "threshold": threshold,
        "cumulative_mean": float(cumulative.mean()),
        "cumulative_sd": float(cumulative.std(ddof=0)),
    }


def future_risk_index(future: pd.DataFrame, scaling: dict[str, float]) -> pd.Series:
    """Start a new cumulative deficit in January 2026; no historical state carry-over."""
    balance = trailing_water_balance(future)
    deficit = np.maximum(0.0, scaling["threshold"] - balance)
    future_cumulative = pd.Series(deficit, index=future.index).cumsum()
    return (future_cumulative - scaling["cumulative_mean"]) / scaling["cumulative_sd"]

