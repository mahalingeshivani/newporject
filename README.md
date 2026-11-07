# AI Sales Prediction (FastAPI + scikit-learn + PostgreSQL + 3D dashboard)

## Run
    docker compose up --build
Open http://localhost:3000 (API docs: http://localhost:8000/docs).
First start seeds ~2 years of synthetic sales and trains the models (~20-40s).

## Structure
- backend/  FastAPI, SQLAlchemy, Gradient Boosting per product, Dockerfile
- frontend/ Three.js 3D bar map + Chart.js forecast, served by nginx (proxies /api), Dockerfile
- docker-compose.yml  db (Postgres 16) + backend + frontend

## Use your own data
Click "Upload CSV" with columns: date,product,region,units,price
(e.g. 2026-01-05,Laptop,North,14,899.0). Models retrain automatically.
Public datasets to try: Kaggle "Store Item Demand Forecasting", "Walmart Sales Forecast".

## API
GET /api/summary · /api/forecast?days=30&product=Laptop · /api/sales/by-product-region · /api/sales/timeseries · /api/metrics · POST /api/train · POST /api/upload
