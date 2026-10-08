"""Acceptance tests against the saved, executed notebook and its numeric artifacts."""

from pathlib import Path
import ast
import hashlib
import json
import unittest

import numpy as np
import pandas as pd
import nbformat
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


class ExecutedNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb = nbformat.read(ROOT / "notebook_code" / "portfolio_risk_2023_2024.ipynb", as_version=4)
        cls.results = json.loads((ROOT/"data"/"processed"/"analysis_results.json").read_text(encoding="utf-8"))
        cls.prices = pd.read_csv(ROOT/"data"/"processed"/"prices_2023_2024.csv", index_col="Date", parse_dates=True)
        cls.returns = pd.read_csv(ROOT/"data"/"processed"/"returns_2023_2024.csv", index_col="Date", parse_dates=True)

    def test_complete_essay_structure_and_execution(self):
        nbformat.validate(self.nb)
        titles = [c.source.splitlines()[0] for c in self.nb.cells if c.cell_type == "markdown"]
        for part, numbers in {"P1": ["1.1","1.2","1.3","2.1","2.2","2.3"],
                              "P2": ["1.1","1.2","1.3","2.1","2.2","2.3","3.1","3.2"],
                              "P3": ["1.1","1.2","1.3","1.4","2.1","2.2","2.3","3.1","3.2","3.3"]}.items():
            for number in numbers:
                self.assertTrue(any(t.startswith(f"## {part} – {number} ") for t in titles))
        self.assertIn("## MỞ ĐẦU", titles)
        self.assertTrue(any(t.startswith("## KẾT LUẬN – 1.") for t in titles))
        self.assertTrue(any(t.startswith("## KẾT LUẬN – 2.") for t in titles))
        code_cells = []
        for i, cell in enumerate(self.nb.cells):
            if cell.cell_type != "code":
                continue
            code_cells.append(cell)
            self.assertEqual(self.nb.cells[i-1].cell_type, "markdown")
            self.assertIsNotNone(cell.execution_count)
            ast.parse("\n".join(line for line in cell.source.splitlines() if not line.startswith("%")))
            self.assertTrue(cell.outputs)
            self.assertFalse(any(o.output_type == "error" for o in cell.outputs))
        self.assertEqual(len(code_cells), 27)
        png_count = sum("image/png" in o.get("data", {}) for c in code_cells for o in c.outputs)
        self.assertGreaterEqual(png_count, 6)

    def test_price_return_contract_and_source_provenance(self):
        self.assertEqual(list(self.prices.columns), ["BTC","GOLD","SP500"])
        self.assertEqual(list(self.returns.columns), list(self.prices.columns))
        self.assertEqual((len(self.prices), len(self.returns)), (502,501))
        self.assertTrue(self.prices.index.is_unique and self.prices.index.is_monotonic_increasing)
        np.testing.assert_allclose(self.returns, np.log(self.prices/self.prices.shift()).dropna(), atol=1e-15)
        for filename, asset in [("Bitcoin Historical Data.csv","BTC"),
                                ("XAU_USD Historical Data.csv","GOLD"),
                                ("S&P 500 Historical Data.csv","SP500")]:
            self.assertEqual(hashlib.sha256((ROOT/"data"/filename).read_bytes()).hexdigest(),
                             self.results["data"]["source_sha256"][asset])

    def test_financial_invariants_and_simulation_against_independent_replay(self):
        w = np.array([.2,.3,.5])
        mu = self.returns.mean().to_numpy()
        cov = self.returns.cov(ddof=1).to_numpy()
        history = self.returns.to_numpy() @ w
        variance = np.var(history, ddof=1)
        portfolio = self.results["portfolio"]
        self.assertAlmostEqual(portfolio["variance_daily"], variance, places=14)
        self.assertAlmostEqual(portfolio["mean_log_return_daily"], np.mean(history), places=14)
        self.assertAlmostEqual(sum(portfolio["risk_contributions_daily"].values()),
                               np.sqrt(variance), places=14)
        self.assertLess(portfolio["volatility_daily"], portfolio["weighted_asset_volatility_daily"])
        # Replay each simulator, then independently obtain the empirical loss threshold.
        L = np.linalg.cholesky(cov)
        normal = (mu + np.random.default_rng(42).standard_normal((100_000,3)) @ L.T) @ w
        rows = np.random.default_rng(42).integers(0,len(history),size=100_000)
        replay = {"Chuẩn đa biến": normal, "Bootstrap lịch sử": history[rows]}
        for name, values in replay.items():
            result = self.results["risk"][name]
            threshold = -100_000*np.expm1(np.quantile(values,.05,method="linear"))
            losses = -100_000*np.expm1(values)
            tail = losses[losses >= threshold]
            np.testing.assert_allclose([result["var_usd"],result["es_usd"]],
                                       [threshold,tail.mean()], rtol=1e-12)
            self.assertEqual(result["tail_count"], len(tail))
            self.assertGreater(result["var_usd"],0)
            self.assertGreaterEqual(result["es_usd"],result["var_usd"])
            reference_sd = np.sqrt(variance) if name == "Chuẩn đa biến" else history.std(ddof=0)
            self.assertLessEqual(abs(values.mean()-history.mean()),5*reference_sd/np.sqrt(len(values)))
            self.assertLessEqual(abs(values.std(ddof=0)/reference_sd-1),.01)
            if name == "Chuẩn đa biến":
                # Compare Monte Carlo VaR with the closed form normal quantile.
                analytical = -100_000*np.expm1(stats.norm.ppf(.05,loc=mu@w,scale=np.sqrt(variance)))
                self.assertLess(abs(result["var_usd"]/analytical-1),.02)
        self.assertTrue(all(self.results["validation"].values()))


if __name__ == "__main__":
    unittest.main()
