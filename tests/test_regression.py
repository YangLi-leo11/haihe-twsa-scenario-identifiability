import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class TestReportedNumerics(unittest.TestCase):
    def test_strict_recursive_validation(self):
        data = pd.read_csv(ROOT / "data/processed/fig2_public/strict_recursive_validation_predictions.csv", parse_dates=["date"])
        self.assertEqual(len(data), 60)
        self.assertEqual(data.date.min(), pd.Timestamp("2020-11-01"))
        self.assertEqual(data.date.max(), pd.Timestamp("2025-10-01"))
        self.assertTrue(data.strict_forecast_from_origin.astype(bool).all())
        self.assertFalse(data.observed_twsa_used_inside_horizon.astype(bool).any())
        self.assertEqual(set(data.origin.astype(str).str[:10]), {"2020-10-01"})
        obs, pred = data.twsa_target_mm.to_numpy(float), data.twsa_pred.to_numpy(float)
        r = float(np.corrcoef(obs, pred)[0, 1])
        rmse = float(np.sqrt(np.mean((pred - obs) ** 2)))
        nse = float(1 - np.sum((pred - obs) ** 2) / np.sum((obs - obs.mean()) ** 2))
        self.assertAlmostEqual(r, 0.691381409468, delta=5e-12)
        self.assertAlmostEqual(rmse, 54.000010579487, delta=5e-12)
        self.assertAlmostEqual(nse, 0.285512462049, delta=5e-12)
        config = json.loads((ROOT / "config/analysis_config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["gru"]["validation_source"], "bundled strict recursive validation prediction table")

    def test_84_month_median_r(self):
        data = pd.read_csv(ROOT / "data/processed/fig3/horizon_6month_summary.csv")
        value = float(data.loc[data.forecast_horizon_months.eq(84), "median_R"].iloc[0])
        self.assertAlmostEqual(value, 0.709656, delta=5e-7)

    def test_groundwater_summaries(self):
        metrics = pd.read_csv(ROOT / "data/processed/groundwater/consistency_summary.csv")
        value = float(metrics.loc[metrics.metric.eq("Pearson R"), "value"].iloc[0])
        self.assertAlmostEqual(value, 0.861, delta=0.001)
        bootstrap = pd.read_csv(ROOT / "data/processed/groundwater/bootstrap_summary.csv").iloc[0]
        self.assertEqual(int(bootstrap.block_length), 6)
        self.assertEqual(int(bootstrap.replicates), 2000)
        self.assertEqual(int(bootstrap.seed), 120)
        self.assertAlmostEqual(float(bootstrap.reproduced_lower), 0.759329, delta=5e-7)
        self.assertAlmostEqual(float(bootstrap.reproduced_upper), 0.921299, delta=5e-7)

    def test_december_2100_spreads(self):
        data = pd.read_csv(ROOT / "data/processed/fig5_fig6/conditional_twsa_monthly.csv", parse_dates=["date"])
        endpoint = data.loc[data.date.eq(pd.Timestamp("2100-12-01"))]
        stationary = endpoint.loc[endpoint.assumption.eq("A_stationary_residual"), "total_twsa_ma12_mm"]
        self.assertAlmostEqual(float(stationary.max() - stationary.min()), 6.756205, delta=5e-7)
        ranges = endpoint.groupby("scenario").total_twsa_ma12_mm.agg(lambda x: x.max() - x.min())
        self.assertAlmostEqual(float(ranges.max()), 306.096855, delta=5e-7)

    def test_pathway_names_are_nonprobabilistic(self):
        data = pd.read_csv(ROOT / "data/processed/fig5_fig6/conditional_twsa_monthly.csv", usecols=["assumption"])
        names = set(data.assumption.unique())
        self.assertEqual(len(names), 5)
        self.assertFalse(any("probab" in name.lower() for name in names))


if __name__ == "__main__":
    unittest.main()
