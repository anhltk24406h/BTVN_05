"""Validate Investing.com CSVs and align closing prices without imputation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
START = pd.Timestamp("2023-01-01")
END = pd.Timestamp("2024-12-31")
ASSETS = ("BTC", "GOLD", "SP500")
SOURCES = {
    "BTC": {
        "file": "Bitcoin Historical Data.csv",
        "date_format": "%m/%d/%Y",
        "name": "Bitcoin (BTC/USD)",
        "unit": "USD/BTC",
        "url": "https://www.investing.com/crypto/bitcoin/historical-data",
    },
    "GOLD": {
        "file": "XAU_USD Historical Data.csv",
        "date_format": "%m/%d/%Y",
        "name": "Vàng giao ngay (XAU/USD)",
        "unit": "USD/troy ounce",
        "url": "https://www.investing.com/currencies/xau-usd-historical-data",
    },
    "SP500": {
        "file": "S&P 500 Historical Data.csv",
        "date_format": "%b %d, %Y",
        "name": "S&P 500 (SPX)",
        "unit": "Điểm chỉ số",
        "url": "https://www.investing.com/indices/us-spx-500-historical-data",
    },
}


class DataValidationError(ValueError):
    """An input cannot safely be used for the requested analysis."""


def _dates(index: pd.Index) -> list[str]:
    return [date.strftime("%Y-%m-%d") for date in index]


def _bad_rows(mask: pd.Series) -> list[int]:
    # CSV header is line 1; data rows begin at line 2.
    return (np.flatnonzero(mask.to_numpy()) + 2).tolist()[:10]


def read_asset(path: Path, asset: str) -> tuple[pd.Series, dict]:
    """Validate all input rows, collapse identical Date/Price pairs, filter period."""
    source = SOURCES[asset]
    if not path.is_file():
        raise DataValidationError(f"{asset}: không tìm thấy file {path}.")
    try:
        raw = pd.read_csv(path, encoding="utf-8-sig", dtype="string")
    except (ValueError, OSError) as exc:
        raise DataValidationError(f"{asset}: không đọc được CSV: {exc}") from exc
    missing_columns = {"Date", "Price"} - set(raw.columns)
    if missing_columns:
        raise DataValidationError(f"{asset}: thiếu cột {sorted(missing_columns)}.")

    date_text = raw["Date"].str.strip()
    date = pd.to_datetime(date_text, format=source["date_format"], errors="coerce")
    if date.isna().any():
        raise DataValidationError(
            f"{asset}: ngày thiếu/không đúng {source['date_format']} tại dòng {_bad_rows(date.isna())}."
        )
    price_text = raw["Price"].str.strip().str.replace(",", "", regex=False)
    price = pd.to_numeric(price_text, errors="coerce").astype("float64")
    invalid = price.isna() | ~np.isfinite(price) | (price <= 0)
    if invalid.any():
        raise DataValidationError(
            f"{asset}: giá thiếu/không hợp lệ/không dương tại dòng {_bad_rows(invalid)}."
        )
    validated = pd.DataFrame({"Date": date, "Price": price})
    conflicts = validated.groupby("Date")["Price"].nunique()
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        raise DataValidationError(f"{asset}: ngày trùng khác giá: {_dates(conflicts.index)}.")
    duplicate_rows = int(validated.duplicated("Date").sum())
    unique = validated.drop_duplicates("Date").sort_values("Date")
    filtered = unique.loc[unique["Date"].between(START, END)].copy()
    observed_years = set(filtered["Date"].dt.year)
    if observed_years != {2023, 2024}:
        raise DataValidationError(
            f"{asset}: phải có dữ liệu trong cả 2023 và 2024; hiện có {sorted(observed_years)}."
        )
    if filtered["Date"].min().month != 1 or filtered["Date"].max().month != 12:
        raise DataValidationError(f"{asset}: dữ liệu chưa bao phủ tháng 01/2023 đến 12/2024.")
    missing = {}
    for column in raw:
        values = raw[column].str.strip()
        missing[column] = int((values.isna() | values.isin(["", "-", "N/A", "null"])).sum())
    series = filtered.set_index("Date")["Price"].rename(asset)
    metadata = {
        **source,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "price_column": "Price",
        "raw_rows": len(raw),
        "raw_start": unique["Date"].min().strftime("%Y-%m-%d"),
        "raw_end": unique["Date"].max().strftime("%Y-%m-%d"),
        "identical_duplicate_rows_removed": duplicate_rows,
        "outside_period_rows_removed": len(unique) - len(filtered),
        "period_rows": len(series),
        "period_start": series.index.min().strftime("%Y-%m-%d"),
        "period_end": series.index.max().strftime("%Y-%m-%d"),
        "missing_values_by_column": missing,
    }
    return series, metadata


def prepare(data_dir: Path = ROOT / "data") -> tuple[pd.DataFrame, dict]:
    """Build prices and a serializable QA report; no files are written here."""
    series_list, metadata = [], {}
    for asset in ASSETS:
        series, asset_report = read_asset(data_dir / SOURCES[asset]["file"], asset)
        series_list.append(series)
        metadata[asset] = asset_report
    prices = pd.concat(series_list, axis=1, join="inner").sort_index()
    prices.index.name = "Date"
    if len(prices) < 3 or set(prices.index.year) != {2023, 2024}:
        raise DataValidationError("Không đủ ngày giao dịch chung trong cả hai năm để tính return.")
    assert list(prices.columns) == list(ASSETS)
    assert prices.index.is_unique and prices.index.is_monotonic_increasing
    for asset, series in zip(ASSETS, series_list):
        excluded = series.index.difference(prices.index)
        metadata[asset]["alignment_rows_removed"] = len(excluded)
        metadata[asset]["alignment_dates_removed"] = _dates(excluded)

    returns = np.log(prices / prices.shift(1)).dropna()
    flags = []
    for asset in ASSETS:
        for date, value in returns.loc[returns[asset].abs() > 0.10, asset].items():
            pos = prices.index.get_loc(date)
            flags.append({
                "asset": asset, "date": date.strftime("%Y-%m-%d"),
                "previous_common_date": prices.index[pos - 1].strftime("%Y-%m-%d"),
                "log_return": float(value), "action": "flag_only_keep_observation",
            })
    gaps = prices.index.to_series().diff().dt.days.dropna().astype(int)
    report = {
        "period": {"start": str(START.date()), "end": str(END.date())},
        "sources": metadata,
        "alignment": {
            "rule": "Intersection of valid price dates; no interpolation or forward fill",
            "price_rows": len(prices), "return_rows": len(returns),
            "first_common_date": prices.index.min().strftime("%Y-%m-%d"),
            "last_common_date": prices.index.max().strftime("%Y-%m-%d"),
            "gap_days_histogram": {str(k): int(v) for k, v in gaps.value_counts().sort_index().items()},
            "max_gap_calendar_days": int(gaps.max()),
            "missing_aligned_prices": int(prices.isna().sum().sum()),
        },
        "outlier_rule": "abs(aligned log return) > 0.10; flag only, no winsorization",
        "outlier_flags": flags,
        "warnings": [
            "Missing Vol. does not affect the analysis of Date/Price.",
            "A common session can span multiple calendar days, including BTC weekends.",
            "Date alignment cannot reconcile different closing times across markets.",
        ],
    }
    return prices, report


def write_outputs(prices: pd.DataFrame, report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prices.to_csv(output_dir / "prices_2023_2024.csv", date_format="%Y-%m-%d", encoding="utf-8")
    (output_dir / "preprocessing_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed")
    args = parser.parse_args()
    try:
        prices, report = prepare(args.data_dir.resolve())
    except DataValidationError as exc:
        parser.exit(1, f"Preprocessing error: {exc}\n")
    write_outputs(prices, report, args.output_dir.resolve())
    print(f"Saved {len(prices)} aligned prices to {args.output_dir.resolve()}")
    print(f"Common dates: {prices.index.min().date()} to {prices.index.max().date()}")
    print(f"Flagged extreme returns (retained): {len(report['outlier_flags'])}")


if __name__ == "__main__":
    main()
