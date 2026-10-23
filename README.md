# Stock Price Prediction

Predicts the **next-day closing price** of a stock using a machine learning model trained on real historical market data (Yahoo Finance, via the `yfinance` library).

## Quick start

```bash
cd stock_predictor
pip install -r requirements.txt

# predict Apple's next-day close (default)
python predict.py

# or any other ticker / date range
python predict.py --ticker MSFT --start 2016-01-01
python predict.py --ticker ^GSPC          # S&P 500 index
```

## What it does

1. **Downloads real data** — daily OHLCV bars for your ticker (defaults to AAPL, starting 2016) from Yahoo Finance. No fake or synthetic data.
2. **Engineers features** a model can actually learn from, all using only *past* information at prediction time:
   - previous closes at lags 1–10 days
   - 1-day, 5-day, 20-day returns
   - intraday range (`high−low` %) and open→close gap
   - volume relative to its 5-day average
   - close vs. its 10-day and 50-day moving averages
   - 20-day rolling volatility
   - RSI(14) momentum indicator
   - day-of-week / month-of-year seasonality
3. **Splits chronologically** — oldest 80% trains, most recent 20% tests. Never shuffles across the split, so the model can't "see the future".
4. **Compares 3 models + a naive baseline** (Linear Regression, Random Forest, Gradient Boosting, and "predict today's price = tomorrow's price"). The naive baseline is the honest bar any model must beat.
5. **Evaluates** with MAE, RMSE, MAPE, R², plus **direction accuracy** (how often it calls tomorrow up vs. down correctly) — and saves `results/predictions.csv`, `results/metrics.json`, `results/predictions.png` and the best `model.pkl`.

## Results (AAPL, real Yahoo Finance data)

Trained on 2,133 trading days (2016–2024), tested on 534 unseen days (Sept 2024 – Oct 2026), including the 2025 drawdown and recovery. Models predict the next-day **return** and the price is reconstructed from it.

| Model | MAE (USD) | MAPE | Direction acc. |
|---|---|---|---|
| **naive (predict today's close)** | **2.96** | **1.19%** | 53.9% (share up) |
| Linear Regression | 2.98 | 1.19% | 51.9% |
| Random Forest | 3.12 | 1.24% | 47.8% |
| Gradient Boosting | 4.34 | 1.68% | 47.4% |

Read this honestly: none of the ML models beat simply predicting *"tomorrow = today"* on price error — that is the expected, textbook outcome for daily stock prices, which are close to a random walk. Direction accuracy likewise hovers around a coin flip. The value of this project is the working pipeline (real data, leakage-free split, honest baseline), not a trading edge.

## Honest caveats

- **Prices are mostly predictable in level (they follow yesterday), but hard to beat directionally.** Models will look impressive on MAE/R² while barely beating the naive baseline — markets are close to a random walk at the daily horizon, and any real edge gets competed away. Direction accuracy near 50–55% is normal and is roughly what the literature reports; ~60% sustained would be remarkable.
- **No transaction costs or slippage** are modeled. A 52% direction hit rate earns nothing after fees.
- **Survivorship / lookahead risk is minimized here** (features only use past data, chronological split), but the test period is still just one market regime. Re-run on different tickers and date ranges before trusting any result.
- **This is not financial advice** — treat outputs as a learning tool, not a trading signal.

## Files

| File | Purpose |
|---|---|
| `predict.py` | the full pipeline (download → features → train → evaluate → plot) |
| `requirements.txt` | Python dependencies |
| `results/` | created on first run: metrics, predictions, plot, saved model |

## Extending it

Ideas if you want to go further: add more tickers as training features, use `--horizon` for multi-day forecasts, or replace price prediction with return prediction + a classification head for up/down. (Currently the script is single-ticker, single-horizon by design to keep it readable.)
