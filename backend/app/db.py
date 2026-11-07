import os, time
from sqlalchemy import create_engine, Column, Integer, String, Float, Date
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://sales:salespass@localhost:5432/salesdb")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False)
Base = declarative_base()


class Sale(Base):
    __tablename__ = "sales"
    id = Column(Integer, primary_key=True)
    date = Column(Date, index=True, nullable=False)
    product = Column(String(60), index=True, nullable=False)
    region = Column(String(60), index=True, nullable=False)
    units = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    revenue = Column(Float, nullable=False)


def init_db(retries: int = 20):
    for _ in range(retries):
        try:
            Base.metadata.create_all(engine)
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError("Database not reachable")
