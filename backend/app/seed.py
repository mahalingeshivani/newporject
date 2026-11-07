"""Generates a realistic synthetic dataset (2 years daily) if the table is empty."""
import numpy as np
from datetime import date, timedelta
from .db import SessionLocal, Sale

PRODUCTS = {"Laptop": 900, "Smartphone": 600, "Headphones": 120, "Smartwatch": 250}
REGIONS = {"North": 1.0, "South": 0.8, "East": 1.2, "West": 0.9}


def seed_if_empty(days: int = 730):
    db = SessionLocal()
    try:
        if db.query(Sale).count() > 0:
            return
        rng = np.random.default_rng(42)
        start = date.today() - timedelta(days=days)
        rows = []
        for d in range(days):
            day = start + timedelta(days=d)
            for p, base_price in PRODUCTS.items():
                for r, rmult in REGIONS.items():
                    season = 1 + 0.25 * np.sin(2 * np.pi * day.timetuple().tm_yday / 365)
                    weekend = 1.3 if day.weekday() >= 5 else 1.0
                    festive = 1.6 if day.month in (11, 12) else 1.0
                    trend = 1 + d / days * 0.4
                    lam = 12 * rmult * season * weekend * festive * trend * (900 / base_price) ** 0.35
                    units = int(rng.poisson(max(lam, 1)))
                    price = round(base_price * rng.uniform(0.95, 1.05), 2)
                    rows.append(Sale(date=day, product=p, region=r, units=units, price=price, revenue=round(units * price, 2)))
        db.bulk_save_objects(rows)
        db.commit()
    finally:
        db.close()
