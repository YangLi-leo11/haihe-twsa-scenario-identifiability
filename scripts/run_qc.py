from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    validation = pd.read_csv(ROOT / "data/processed/fig2_public/strict_recursive_validation_predictions.csv")
    obs, pred = validation.twsa_target_mm.to_numpy(float), validation.twsa_pred.to_numpy(float)
    gru_r = float(np.corrcoef(obs, pred)[0, 1])
    gru_rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
    gru_nse = float(1 - np.sum((pred - obs) ** 2) / np.sum((obs - obs.mean()) ** 2))
    horizons = pd.read_csv(ROOT / "data/processed/fig3/horizon_6month_summary.csv")
    r84 = float(horizons.loc[horizons.forecast_horizon_months.eq(84), "median_R"].iloc[0])
    pathways = pd.read_csv(ROOT / "data/processed/fig5_fig6/conditional_twsa_monthly.csv", parse_dates=["date"])
    endpoint = pathways.loc[pathways.date.eq(pd.Timestamp("2100-12-01"))]
    stationary = endpoint.loc[endpoint.assumption.eq("A_stationary_residual"), "total_twsa_ma12_mm"]
    inter_ssp = float(stationary.max() - stationary.min())
    envelope = float(endpoint.groupby("scenario").total_twsa_ma12_mm.agg(lambda x: x.max() - x.min()).max())
    groundwater = pd.read_csv(ROOT / "data/processed/groundwater/consistency_summary.csv")
    gw_r = float(groundwater.loc[groundwater.metric.eq("Pearson R"), "value"].iloc[0])
    bootstrap = pd.read_csv(ROOT / "data/processed/groundwater/bootstrap_summary.csv").iloc[0]
    lower, upper = float(bootstrap.reproduced_lower), float(bootstrap.reproduced_upper)
    checks = [
        ("GRU Pearson R", 0.691381409468, gru_r, 5e-12),
        ("GRU RMSE mm", 54.000010579487, gru_rmse, 5e-12),
        ("GRU NSE", 0.285512462049, gru_nse, 5e-12),
        ("84-month analogue median R", 0.709656, r84, 5e-7),
        ("Groundwater Pearson R", 0.861, gw_r, 0.001),
        ("Bootstrap lower", 0.759329, lower, 5e-7),
        ("Bootstrap upper", 0.921299, upper, 5e-7),
        ("2100 maximum inter-SSP separation mm", 6.756205, inter_ssp, 5e-7),
        ("2100 maximum pathway envelope mm", 306.096855, envelope, 5e-7),
    ]
    rows = [(name, expected, actual, abs(actual - expected), "PASS" if abs(actual - expected) <= tolerance else "FAIL") for name, expected, actual, tolerance in checks]
    output = ROOT / "outputs/reproduction_summary.csv"
    output.parent.mkdir(exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["check", "expected", "reproduced", "absolute_difference", "status"])
        writer.writerows(rows)
    figures = []
    for number, suffix in ((2, "v3"), (3, "v2"), (5, "v2"), (6, "v2")):
        name = f"Fig{number}_CEE_harmonized_{suffix}.png"
        reference, reproduced = ROOT / "figures/reference" / name, ROOT / "figures/reproduced" / name
        if not reproduced.exists():
            status = "NOT_RUN"
        else:
            status = "PASS" if reference.exists() and sha256(reference) == sha256(reproduced) else "WARNING_RENDER_DIFF"
        figures.append((f"Fig. {number}", status))
    figures.insert(2, ("Fig. 4", "EXTERNAL_DEPENDENCY"))
    with (ROOT / "outputs/figure_checks.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["figure", "status"])
        writer.writerows(figures)
    failures = [row for row in rows if row[-1] != "PASS"]
    if failures:
        raise SystemExit(f"Reproduction check failed: {failures}")
    print(f"Strict recursive validation PASS: R={gru_r:.12f}, RMSE={gru_rmse:.12f} mm, NSE={gru_nse:.12f}")
    print(f"84-month analogue median R={r84:.9f}")
    print(f"Groundwater bootstrap CI={lower:.9f}-{upper:.9f}")
    print(f"2100 maximum inter-SSP separation={inter_ssp:.9f} mm")
    print(f"2100 maximum pathway envelope={envelope:.9f} mm")


if __name__ == "__main__":
    main()
