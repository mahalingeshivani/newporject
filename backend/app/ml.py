import numpy as np, pandas as pd
from datetime import timedelta
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sqlalchemy import text
from .db import engine

MODELS: dict = {}
META: dict = {}


def _features(dates: pd.DatetimeIndex, t0: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame({
        "t": (dates - t0).days,
        "dow": dates.dayofweek,
        "month": dates.month,
        "doy_sin": np.sin(2 * np.pi * dates.dayofyear / 365),
        "doy_cos": np.cos(2 * np.pi * dates.dayofyear / 365),
        "weekend": (dates.dayofweek >= 5).astype(int),
    })


def load_daily() -> pd.DataFrame:
    df = pd.read_sql(text("SELECT date, product, SUM(units) AS units, SUM(revenue) AS revenue FROM sales GROUP BY date, product ORDER BY date"), engine)
    df["date"] = pd.to_datetime(df["date"])
    return df


def train_all() -> dict:
    df = load_daily()
    metrics = {}
    for product, g in df.groupby("product"):
        g = g.sort_values("date")
        if len(g) < 120:
            continue
        t0 = g["date"].min()
        X, y = _features(pd.DatetimeIndex(g["date"]), t0), g["units"].values
        split = len(g) - 60
        model = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42)
        model.fit(X[:split], y[:split])
        pred = model.predict(X[split:])
        metrics[product] = {"mae": round(float(mean_absolute_error(y[split:], pred)), 2), "r2": round(float(r2_score(y[split:], pred)), 3)}
        model.fit(X, y)  # refit on all data
        MODELS[product] = model
        META[product] = {"t0": t0, "last": g["date"].max(), "avg_price": float(g["revenue"].sum() / max(g["units"].sum(), 1))}
    return metrics


def forecast(days: int = 30, product: str | None = None) -> list[dict]:
    out = []
    for p, model in MODELS.items():
        if product and p != product:
            continue
        m = META[p]
        dates = pd.date_range(m["last"] + timedelta(days=1), periods=days)
        preds = np.maximum(model.predict(_features(dates, m["t0"])), 0)
        for d, u in zip(dates, preds):
            out.append({"date": d.strftime("%Y-%m-%d"), "product": p, "units": round(float(u), 1), "revenue": round(float(u) * m["avg_price"], 2)})
    return out
