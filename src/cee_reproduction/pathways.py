"""Five deterministic residual assumptions and conditional TWSA reconstruction."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


PATHWAYS = (
    "stationary_residual",
    "trend_capped_persistence_residual",
    "risk_conditioned_residual",
    "moderation_residual",
    "consistency_reference_residual",
)


@dataclass(frozen=True)
class ResidualStatistics:
    mean: float
    sd: float
    p05: float
    p25: float
    p50: float
    p75: float
    p95: float
    iqr: float
    tail_mean: float
    tail_slope_mm_per_year: float


def summarize_residual(dates, residual_state) -> tuple[ResidualStatistics, pd.Series]:
    frame = pd.DataFrame({"date": pd.to_datetime(dates), "state": residual_state}).dropna()
    frame["month"] = frame["date"].dt.month
    values = frame["state"].to_numpy(float)
    p05, p25, p50, p75, p95 = np.percentile(values, [5, 25, 50, 75, 95])
    tail = frame.sort_values("date").tail(min(24, len(frame)))
    x = (tail["date"] - tail["date"].min()).dt.days.to_numpy(float) / 365.25
    slope = float(np.polyfit(x, tail["state"], 1)[0]) if len(tail) >= 6 else 0.0
    stats = ResidualStatistics(
        mean=float(values.mean()), sd=float(values.std(ddof=0)),
        p05=float(p05), p25=float(p25), p50=float(p50),
        p75=float(p75), p95=float(p95), iqr=float(p75 - p25),
        tail_mean=float(tail["state"].mean()),
        tail_slope_mm_per_year=float(np.clip(slope, -2.0, 2.0)),
    )
    climatology = frame.groupby("month")["state"].mean().reindex(range(1, 13))
    return stats, climatology.interpolate(limit_direction="both")


def build_pathways(future: pd.DataFrame, stats: ResidualStatistics, monthly_climatology: pd.Series) -> pd.DataFrame:
    """Return deterministic assumptions; the rows are not probabilities or forecasts."""
    frame = future.sort_values("date").copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["month"] = frame["date"].dt.month
    frame["years"] = (frame["date"] - frame["date"].min()).dt.days / 365.25
    monthly = frame["month"].map(monthly_climatology).to_numpy(float)
    early = frame[frame["date"].dt.year.between(2026, 2035)]["risk_index"].to_numpy(float)
    risk_reference = float(np.nanmedian(early))
    risk_iqr = float(np.nanpercentile(early, 75) - np.nanpercentile(early, 25))
    risk_norm = np.clip((frame["risk_index"].to_numpy(float) - risk_reference) / risk_iqr, -3, 3)
    lower_wide, upper_wide = stats.p05 - stats.iqr, stats.p95 + stats.iqr
    persistence = np.clip(
        stats.tail_mean + stats.tail_slope_mm_per_year * frame["years"].to_numpy(float),
        lower_wide, upper_wide,
    )
    risk_conditioned = np.clip(
        monthly - 0.25 * stats.sd * risk_norm,
        stats.p05 - 0.75 * stats.sd,
        stats.p95 + 0.75 * stats.sd,
    )
    decay = np.exp(-frame["years"].to_numpy(float) / 30.0)
    moderation = np.clip(monthly + (stats.tail_mean - monthly) * decay, lower_wide, upper_wide)
    consistency = np.clip(
        0.5 * persistence + 0.5 * moderation,
        stats.p05 - 0.5 * stats.iqr,
        stats.p95 + 0.5 * stats.iqr,
    )
    arrays = dict(zip(PATHWAYS, [monthly, persistence, risk_conditioned, moderation, consistency]))
    return pd.concat(
        [frame.assign(pathway=name, residual_storage_state=value) for name, value in arrays.items()],
        ignore_index=True,
    )


def conditional_twsa(pathways: pd.DataFrame) -> pd.DataFrame:
    result = pathways.copy()
    result["conditional_twsa"] = (
        result["climate_sensitive_component"] + result["residual_storage_state"]
    )
    result["conditional_twsa_ma12"] = result.groupby(
        ["scenario", "pathway"], sort=False
    )["conditional_twsa"].transform(lambda values: values.rolling(12, min_periods=1).mean())
    return result

