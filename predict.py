"""
Stock price prediction with real market data.

Pipeline:
  1. Download real daily OHLCV data with yfinance (Yahoo Finance source).
  2. Engineer lag/rolling/volatility features (fit on the training split only
     via sklearn Pipelines, so there is no look-ahead leakage).
  3. Forecast: next-day closing price vs. next-day direction (up/down).
  4. Compare Linear Regression, Random Forest and Gradient Boosting against a
     naive "prediction == today's close" baseline.
  5. Save predictions, metrics and a plot.

Requires internet access on first run (yfinance fetches live data).
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless: works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RNG_SEED = 42


# ---------------------------------------------------------------- data
def download_data(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    """Download real daily OHLCV data and sanitize it."""
    print(f"Downloading real market data for {ticker} ...")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise SystemExit(
            f"No data returned for {ticker}. Check the ticker symbol and your "
            "internet connection, then try again."
        )

    # yfinance can return MultiIndex columns even for a single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close", "High", "Low", "Open", "Volume"]].copy()
    df = df.dropna()
    df = df[~df.index.duplicated(keep="first")]
    # Drop useless all-zero-volume rows some datasets contain.
    if len(df) > 50:
        df = df[df["Volume"] > 0]
        if df.empty:  # extremely unlikely, but be safe
            raise SystemExit("All rows had zero volume; cannot model this ticker.")
    if len(df) < 300:
        raise SystemExit(
            f"Only {len(df)} usable rows for {ticker}; need at least 300 to "
            "train meaningfully. Pick an earlier start date or a more liquid ticker."
        )
    print(f"Got {len(df)} trading days: {df.index[0].date()} -> {df.index[-1].date()}")
    return df


# ---------------------------------------------------------------- features
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag, return, rolling and volatility features. Index = trade date."""
    f = df.copy()
    close = f["Close"]

    # Tomorrow's close is what we want the model to predict.
    f["target_next_close"] = close.shift(-1)
    # We model the *next-day return* (not the price level) so that tree
    # models can extrapolate: trees never predict outside the training
    # range, so predicting prices directly fails whenever price trends
    # above the training range. Reconstructing the price from the predicted
    # return + today's close sidesteps that.
    f["target_next_ret"] = f["target_next_close"] / close - 1
    # Direction target: 1 if price goes up tomorrow, else 0.
    f["target_next_up"] = (f["target_next_close"] > close).astype(int)

    # Price history features (all built from today's data and the past only).
    for lag in (1, 2, 3, 5, 10):
        f[f"close_lag_{lag}"] = close.shift(lag)
    f["return_1d"] = close.pct_change(1)
    f["return_5d"] = close.pct_change(5)
    f["return_20d"] = close.pct_change(20)
    f["hl_range_pct"] = (f["High"] - f["Low"]) / close
    f["oc_gap_pct"] = (f["Close"] - f["Open"]) / f["Open"]
    f["volume_ratio_5d"] = f["Volume"] / f["Volume"].rolling(5).mean()
    f["ma_10"] = close.rolling(10).mean()
    f["ma_50"] = close.rolling(50).mean()
    f["close_vs_ma50_pct"] = (close - f["ma_50"]) / f["ma_50"]
    f["volatility_20d"] = close.pct_change().rolling(20).std()

    # Momentum indicator: RSI(14), the standard Wilder version.
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss
    f["rsi_14"] = 100 - 100 / (1 + rs)
    f["rsi_14"] = f["rsi_14"].where(loss != 0, 100.0).fillna(50.0)

    # Day of week / month of year — weak but harmless seasonal features.
    f["day_of_week"] = f.index.dayofweek
    f["month"] = f.index.month

    # Rolling means introduce NaNs in the first rows; drop them, and also
    # drop the final row (its target_next_close is NaN because there is no
    # next trading day yet).
    f = f.dropna(subset=["target_next_close"])
    feature_cols = [  # every predictor except our two targets
        c
        for c in f.columns
        if c not in ("Open", "High", "Low", "Volume",  # keep raw prices too
                     "target_next_close", "target_next_ret", "target_next_up")
    ]
    f = f.dropna(subset=feature_cols)
    return f, feature_cols


# ---------------------------------------------------------------- split
def chronological_split(f: pd.DataFrame):
    """Split in time order: oldest 80% to train, last 20% to test."""
    n = len(f)
    cut = int(n * 0.8)
    train = f.iloc[:cut]
    test = f.iloc[cut:]
    # Shuffle the training set (linear regression doesn't need it, but trees
    # trained with subsampling benefit; the split itself already guarantees
    # no time leakage because test comes strictly after train).
    return train, test


# ---------------------------------------------------------------- models
def make_models() -> dict:
    return {
        "naive (predict today's close)": None,  # handled specially
        "Linear Regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "Random Forest": Pipeline([
            ("model", RandomForestRegressor(
                n_estimators=400,
                max_depth=12,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=RNG_SEED,
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("model", GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.9,
                random_state=RNG_SEED,
            )),
        ]),
    }


def evaluate_model(name, model, X_train, y_train, X_test, y_test_ret,
                   y_test_close, x_test_close):
    """Fit, predict returns, reconstruct prices, and score in both spaces."""
    if model is None:  # naive baseline: predict a 0% return (= today's close)
        preds_ret = np.zeros(len(y_test_ret))
    else:
        model.fit(X_train, y_train)
        preds_ret = model.predict(X_test)

    # Reconstruct the predicted next-day close from today's close.
    preds_close = x_test_close * (1 + preds_ret)

    regression = {
        "model": name,
        "MAE": float(mean_absolute_error(y_test_close, preds_close)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test_close, preds_close))),
        "MAPE_pct": float(mean_absolute_percentage_error(y_test_close, preds_close) * 100),
        "R2": float(r2_score(y_test_close, preds_close)),
        "return_MAE_pct": float(mean_absolute_error(y_test_ret, preds_ret) * 100),
    }
    return regression, preds_close, preds_ret


# ---------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol, e.g. AAPL, MSFT, ^GSPC")
    parser.add_argument("--start", default="2016-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--out", default="results", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = download_data(args.ticker, args.start, args.end)
    features, feature_cols = build_features(df)
    train, test = chronological_split(features)
    print(f"Train: {len(train)} days ({train.index[0].date()} -> {train.index[-1].date()})")
    print(f"Test:  {len(test)} days  ({test.index[0].date()} -> {test.index[-1].date()})")

    X_train = train[feature_cols]
    y_train = train["target_next_ret"]
    X_test = test[feature_cols]
    y_test_ret = test["target_next_ret"]
    y_test_close = test["target_next_close"]
    x_test_close = X_test["Close"].values

    models = make_models()
    all_preds = pd.DataFrame({"actual": y_test_close}, index=y_test_close.index)
    metrics_rows = []

    for name, model in models.items():
        metrics, preds_close, preds_ret = evaluate_model(
            name, model, X_train, y_train, X_test, y_test_ret,
            y_test_close, x_test_close,
        )
        metrics_rows.append(metrics)
        all_preds[name] = preds_close

    metrics_df = pd.DataFrame(metrics_rows).set_index("model")
    print("\n=== Next-day close: metrics on the held-out test set ===")
    print("(MAE/RMSE in USD, reconstructed from predicted returns)")
    print(metrics_df.round(3).to_string())

    # ---- Direction accuracy (up vs down tomorrow) ----
    print("\n=== Direction classification ===")
    direction = {}
    for name, model in models.items():
        if model is None:
            continue
        # A predicted return above 0 means "price goes up tomorrow".
        implied_dir = (all_preds[name].values > X_test["Close"].values).astype(int)
        direction[name] = float(accuracy_score(test["target_next_up"], implied_dir))
    # The naive baseline is always flat: its best case is "always guess up",
    # which scores the share of days that actually rose.
    direction["naive (always guess up)"] = float(test["target_next_up"].mean())
    for name, acc in direction.items():
        print(f"  {name}: {acc:.1%}")
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump({"metrics": metrics_rows, "direction_accuracy": direction}, fh, indent=2)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(y_test_close.index, y_test_close.values, label="Actual close", color="black", linewidth=1.5)
    for name in ("naive (predict today's close)", "Linear Regression",
                 "Random Forest", "Gradient Boosting"):
        if name in all_preds:
            style = {"linewidth": 1, "alpha": 0.85}
            if name.startswith("naive"):
                style.update({"linestyle": "--", "color": "gray"})
            ax.plot(all_preds.index, all_preds[name], label=name, **style)
    # Show where the split happened.
    ax.axvline(test.index[0], color="blue", alpha=0.2)
    ax.set_title(f"{args.ticker} — next-day close: actual vs predicted (test period)")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "predictions.png", dpi=150)

    # ---- Save artifacts ----
    all_preds.to_csv(out_dir / "predictions.csv")
    # Save the best *trained* model (the naive baseline is not a model file).
    trained = metrics_df.loc[metrics_df.index != "naive (predict today's close)"]
    best_name = trained["MAE"].idxmin()
    best = models[best_name]
    best.fit(X_train, y_train)
    joblib.dump(best, out_dir / "model.pkl")
    beats_naive = metrics_df.loc[best_name, "MAE"] < metrics_df.loc["naive (predict today's close)", "MAE"]
    verdict = "beats the naive baseline" if beats_naive else "does NOT beat the naive baseline"
    print(f"\nSaved: {out_dir}/model.pkl (best trained model: {best_name}; {verdict})")
    print(f"Saved: {out_dir}/predictions.csv, metrics.json, predictions.png")


if __name__ == "__main__":
    main()
