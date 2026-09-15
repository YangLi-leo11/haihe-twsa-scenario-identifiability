"""Historical-analogue estimator used for the climate-sensitive component."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def calendar_features(dates) -> tuple[np.ndarray, np.ndarray]:
    month = pd.DatetimeIndex(dates).month.to_numpy()
    angle = 2 * np.pi * (month - 1) / 12
    return np.sin(angle), np.cos(angle)


def adaptive_q(n_train: int) -> int:
    return int(min(12, max(3, n_train // 12)))


@dataclass
class HistoricalAnalogue:
    scaler: StandardScaler
    neighbours: NearestNeighbors
    target: np.ndarray


def fit(features: np.ndarray, target: np.ndarray, *, q: int | None = None) -> HistoricalAnalogue:
    features = np.asarray(features, dtype=float)
    target = np.asarray(target, dtype=float)
    if q is None:
        q = adaptive_q(len(features))
    scaler = StandardScaler().fit(features)
    neighbours = NearestNeighbors(n_neighbors=min(q, len(features))).fit(
        scaler.transform(features)
    )
    return HistoricalAnalogue(scaler, neighbours, target)


def predict(model: HistoricalAnalogue, features: np.ndarray) -> np.ndarray:
    standardized = model.scaler.transform(np.asarray(features, dtype=float))
    distances, indices = model.neighbours.kneighbors(standardized)
    weights = 1.0 / (distances + 1e-6)
    weights /= weights.sum(axis=1, keepdims=True)
    return np.sum(model.target[indices] * weights, axis=1)


def past_only_candidates(table: pd.DataFrame, origin) -> pd.DataFrame:
    """Matched-origin tests admit only rows strictly before the forecast origin."""
    return table.loc[pd.to_datetime(table["date"]) < pd.Timestamp(origin)].copy()

