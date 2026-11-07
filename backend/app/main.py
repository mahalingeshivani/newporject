import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from .db import init_db, engine, SessionLocal, Sale
from .seed import seed_if_empty
from . import ml

app = FastAPI(title="AI Sales Prediction API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
STATE = {"metrics": {}}


@app.on_event("startup")
def startup():
    init_db()
    seed_if_empty()
    STATE["metrics"] = ml.train_all()


def q(sql: str, **params):
    with engine.connect() as c:
        return [dict(r._mapping) for r in c.execute(text(sql), params)]


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/summary")
def summary():
    k = q("SELECT COALESCE(SUM(revenue),0) AS revenue, COALESCE(SUM(units),0) AS units, COUNT(DISTINCT date) AS days FROM sales")[0]
    last30 = q("SELECT COALESCE(SUM(revenue),0) AS r FROM sales WHERE date > (SELECT MAX(date) FROM sales) - 30")[0]["r"]
    prev30 = q("SELECT COALESCE(SUM(revenue),0) AS r FROM sales WHERE date > (SELECT MAX(date) FROM sales) - 60 AND date <= (SELECT MAX(date) FROM sales) - 30")[0]["r"]
    growth = round((last30 - prev30) / prev30 * 100, 1) if prev30 else 0
    nxt = sum(r["revenue"] for r in ml.forecast(30))
    return {"revenue": round(k["revenue"], 2), "units": int(k["units"]), "days": k["days"],
            "last30_revenue": round(last30, 2), "growth_pct": growth, "forecast30_revenue": round(nxt, 2)}


@app.get("/api/sales/by-product-region")
def by_product_region():
    return q("SELECT product, region, ROUND(SUM(revenue)::numeric,2)::float AS revenue, SUM(units)::int AS units FROM sales GROUP BY product, region ORDER BY product, region")


@app.get("/api/sales/timeseries")
def timeseries(days: int = 90, product: str | None = None):
    sql = "SELECT date::text AS date, ROUND(SUM(units)::numeric,1)::float AS units, ROUND(SUM(revenue)::numeric,2)::float AS revenue FROM sales WHERE date > (SELECT MAX(date) FROM sales) - :d"
    if product:
        sql += " AND product = :p"
    sql += " GROUP BY date ORDER BY date"
    return q(sql, d=days, p=product)


@app.get("/api/products")
def products():
    return [r["product"] for r in q("SELECT DISTINCT product FROM sales ORDER BY product")]


@app.get("/api/forecast")
def forecast(days: int = 30, product: str | None = None):
    if not 1 <= days <= 365:
        raise HTTPException(400, "days must be 1-365")
    daily = {}
    for r in ml.forecast(days, product):
        d = daily.setdefault(r["date"], {"units": 0, "revenue": 0})
        d["units"] += r["units"]; d["revenue"] += r["revenue"]
    return {"forecast": [{"date": k, "units": round(v["units"], 1), "revenue": round(v["revenue"], 2)} for k, v in sorted(daily.items())]}


@app.post("/api/train")
def train():
    STATE["metrics"] = ml.train_all()
    return STATE["metrics"]


@app.get("/api/metrics")
def metrics():
    return STATE["metrics"]


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """CSV columns: date,product,region,units,price"""
    df = pd.read_csv(io.BytesIO(await file.read()))
    need = {"date", "product", "region", "units", "price"}
    if not need.issubset(df.columns):
        raise HTTPException(400, f"CSV needs columns: {sorted(need)}")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    db = SessionLocal()
    try:
        db.bulk_save_objects([Sale(date=r.date, product=str(r.product), region=str(r.region), units=int(r.units), price=float(r.price), revenue=float(r.units) * float(r.price)) for r in df.itertuples()])
        db.commit()
    finally:
        db.close()
    STATE["metrics"] = ml.train_all()
    return {"inserted": len(df), "metrics": STATE["metrics"]}
