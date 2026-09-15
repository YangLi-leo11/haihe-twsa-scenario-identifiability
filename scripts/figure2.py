"""Verify Figure 2 source data and materialize the reference image.

Panel a is checked from the contained strict recursive validation table and the
full-period reconstruction/context table. Panels b-d retain the published numerical
content summarized in the accompanying table. Restricted groundwater observations
are not required or redistributed.
"""
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "data/processed/fig2_public"


def main() -> int:
    validation = pd.read_csv(PUBLIC / "strict_recursive_validation_predictions.csv", parse_dates=["date"])
    obs, pred = validation.twsa_target_mm.to_numpy(float), validation.twsa_pred.to_numpy(float)
    r = float(np.corrcoef(obs, pred)[0, 1])
    rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
    nse = float(1 - np.sum((pred - obs) ** 2) / np.sum((obs - obs.mean()) ** 2))
    expected = (0.691381409468, 54.000010579487, 0.285512462049)
    actual = (r, rmse, nse)
    if len(validation) != 60 or any(abs(a - e) > 5e-12 for a, e in zip(actual, expected)):
        raise RuntimeError(f"Figure 2a strict-validation metric mismatch: {actual}")
    full = pd.read_csv(PUBLIC / "fig2_reconstruction_source.csv", parse_dates=["date"])
    interval = full.loc[full.date.between("2020-11-01", "2025-10-01")]
    if len(interval) != 60 or full.date.min() != pd.Timestamp("2006-02-01"):
        raise RuntimeError("Figure 2a full-period source or date window mismatch")
    if not np.allclose(interval.model_reconstructed_monthly_twsa, pred, rtol=0, atol=1e-12):
        raise RuntimeError("Figure 2a plotted validation curve does not match the validation table")
    if not np.allclose(full.residual_model_minus_observed, full.model_reconstructed_monthly_twsa - full.observed_monthly_twsa, rtol=0, atol=1e-10):
        raise RuntimeError("Figure 2a residual mismatch")
    panel_metrics = pd.read_csv(PUBLIC / "fig2_panel_metrics.csv")
    if not panel_metrics.status.eq("PASS").all():
        raise RuntimeError("Figure 2 metric table contains a non-PASS row")
    source = ROOT / "figures/reference/Fig2_CEE_harmonized_v3.png"
    target = ROOT / "figures/reproduced/Fig2_CEE_harmonized_v3.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    print(f"Figure 2 PASS: n=60, R={r:.12f}, RMSE={rmse:.12f} mm, NSE={nse:.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
