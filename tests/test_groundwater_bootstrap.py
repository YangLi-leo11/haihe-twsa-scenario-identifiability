import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cee_reproduction.groundwater import moving_block_bootstrap_correlation


class TestGroundwaterBootstrap(unittest.TestCase):
    def test_non_circular_algorithm_regression(self):
        x = pd.Series(np.arange(12, dtype=float))
        y = pd.Series([0, 1, 3, 2, 5, 4, 7, 9, 8, 11, 10, 12], dtype=float)
        values = moving_block_bootstrap_correlation(
            x, y, block_length=4, repetitions=100, seed=7
        )
        lower, upper = np.quantile(values, [0.025, 0.975])
        self.assertAlmostEqual(lower, 0.75534341, places=8)
        self.assertAlmostEqual(upper, 0.97917360, places=8)


if __name__ == "__main__":
    unittest.main()
