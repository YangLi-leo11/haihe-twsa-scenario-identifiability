from __future__ import annotations

import numpy as np


def _paired(observed, predicted) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(predicted)
    return observed[valid], predicted[valid]


def pearson_r(observed, predicted) -> float:
    observed, predicted = _paired(observed, predicted)
    if len(observed) < 3 or observed.std() == 0 or predicted.std() == 0:
        return float("nan")
    return float(np.corrcoef(observed, predicted)[0, 1])


def rmse(observed, predicted) -> float:
    observed, predicted = _paired(observed, predicted)
    return float(np.sqrt(np.mean((observed - predicted) ** 2)))


def nse(observed, predicted) -> float:
    observed, predicted = _paired(observed, predicted)
    denominator = np.sum((observed - observed.mean()) ** 2)
    return float(1 - np.sum((observed - predicted) ** 2) / denominator)

