"""Meaningful input and calendar-alignment checks for the preprocessing pipeline."""

from pathlib import Path
from datetime import datetime
import csv
import json
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "script"))
from preprocessing import ASSETS, SOURCES, DataValidationError, prepare, read_asset, write_outputs


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="test-input-", dir=ROOT)
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def write_asset(self, asset, records, columns=("Date", "Price")):
        path = self.folder / SOURCES[asset]["file"]
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(records)
        return path

    def date(self, asset, text):
        return datetime.strptime(text, "%Y-%m-%d").strftime(SOURCES[asset]["date_format"])

    def records(self, asset):
        return [(self.date(asset, "2023-01-03"), "1,234.50"),
                (self.date(asset, "2024-12-31"), "2,000.25")]

    def test_explicit_date_formats_thousands_bom_and_outside_period(self):
        for asset in ASSETS:
            with self.subTest(asset=asset):
                records = self.records(asset)
                records.append((self.date(asset, "2022-12-30"), "900.0"))
                series, report = read_asset(self.write_asset(asset, records[::-1]), asset)
                self.assertEqual(len(series), 2)
                self.assertEqual(series.iloc[0], 1234.5)
                self.assertEqual(str(series.index[0].date()), "2023-01-03")
                self.assertEqual(report["outside_period_rows_removed"], 1)

    def test_missing_or_invalid_dates_fail(self):
        for date_text in ["", "31/12/2024", "02/30/2023", "not a date"]:
            with self.subTest(date=date_text):
                records = self.records("BTC") + [(date_text, "100")]
                with self.assertRaisesRegex(DataValidationError, "BTC.*ngày"):
                    read_asset(self.write_asset("BTC", records), "BTC")

    def test_missing_non_numeric_non_finite_and_non_positive_prices_fail(self):
        for bad_price in ["", "NaN", "inf", "-inf", "0", "-10", "not a price"]:
            with self.subTest(price=bad_price):
                records = self.records("BTC") + [("01/04/2023", bad_price)]
                with self.assertRaisesRegex(DataValidationError, "BTC.*giá"):
                    read_asset(self.write_asset("BTC", records), "BTC")

    def test_missing_file_and_columns_fail(self):
        with self.assertRaisesRegex(DataValidationError, "không tìm thấy"):
            read_asset(self.folder / "missing.csv", "BTC")
        with self.assertRaisesRegex(DataValidationError, "thiếu cột"):
            read_asset(self.write_asset("BTC", [("01/03/2023",)], columns=("Date",)), "BTC")

    def test_duplicate_same_price_collapses_but_conflicting_price_fails(self):
        records = self.records("BTC")
        series, report = read_asset(self.write_asset("BTC", records + [records[0]]), "BTC")
        self.assertEqual(len(series), 2)
        self.assertEqual(report["identical_duplicate_rows_removed"], 1)
        with self.assertRaisesRegex(DataValidationError, "trùng khác giá"):
            read_asset(self.write_asset("BTC", records + [(records[0][0], "1235")]), "BTC")

    def test_wrong_years_and_incomplete_month_coverage_fail(self):
        for records in [[("09/08/2026", "100"), ("10/08/2026", "110")],
                        [("02/01/2023", "100"), ("12/31/2024", "110")]]:
            with self.subTest(records=records):
                with self.assertRaises(DataValidationError):
                    read_asset(self.write_asset("BTC", records), "BTC")

    def test_intersection_then_return_keeps_full_btc_weekend_move(self):
        dates = ["2023-01-03", "2023-01-06", "2023-01-09", "2024-12-31"]
        for asset in ASSETS:
            records = [(self.date(asset, d), str(p)) for d,p in zip(dates, [100,110,130,140])]
            if asset == "BTC":
                records += [(self.date(asset, "2023-01-07"), "120"),
                            (self.date(asset, "2023-01-08"), "125")]
            self.write_asset(asset, records)
        prices, report = prepare(self.folder)
        self.assertEqual(len(prices), 4)
        self.assertNotIn(datetime(2023,1,7), prices.index)
        returns = np.log(prices / prices.shift(1)).dropna()
        self.assertAlmostEqual(returns.loc["2023-01-09", "BTC"], np.log(130/110))
        self.assertNotAlmostEqual(returns.loc["2023-01-09", "BTC"], np.log(130/125))
        self.assertEqual(report["sources"]["BTC"]["alignment_rows_removed"], 2)
        self.assertTrue(any(f["date"] == "2023-01-09" for f in report["outlier_flags"]))
        # Extreme returns must remain in prices and in the subsequent return dataset.
        self.assertIn("2023-01-09", prices.index)
        write_outputs(prices, report, self.folder / "processed")
        text = (self.folder / "processed" / "prices_2023_2024.csv").read_text(encoding="utf-8")
        self.assertEqual(text.splitlines()[0], "Date,BTC,GOLD,SP500")
        persisted = json.loads((self.folder / "processed" / "preprocessing_report.json").read_text(encoding="utf-8"))
        self.assertEqual(persisted["alignment"]["price_rows"], 4)

    def test_current_inputs_match_expected_snapshot(self):
        prices, report = prepare(ROOT / "data")
        self.assertEqual([report["sources"][a]["period_rows"] for a in ASSETS], [731,519,502])
        self.assertEqual(len(prices), 502)
        self.assertEqual(report["alignment"]["return_rows"], 501)
        self.assertEqual(report["alignment"]["first_common_date"], "2023-01-03")
        self.assertEqual(report["alignment"]["last_common_date"], "2024-12-31")
        self.assertEqual([report["sources"][a]["alignment_rows_removed"] for a in ASSETS], [229,17,0])


if __name__ == "__main__":
    unittest.main()
