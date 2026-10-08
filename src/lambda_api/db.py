import os
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine

ENV_PATH = Path(__file__).parent.parent.parent / ".env"
load_dotenv(ENV_PATH)


def get_connection():
    return psycopg2.connect(os.getenv("DATABASE_URL"))


def get_engine():
    url = os.getenv("DATABASE_URL").replace("postgresql://", "postgresql+psycopg2://", 1)
    return create_engine(url)


def get_simulated_date(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT simulated_date FROM sim_state WHERE id = %s", (1,))
        row = cur.fetchone()
    return row[0]


def get_sales_history(engine, store, item, simulated_date):
    query = """
        SELECT store, item, sale_date, sale
        FROM sales_history
        WHERE store = %s AND item = %s AND sale_date <= %s
        ORDER BY sale_date DESC
        LIMIT 365
    """
    df = pd.read_sql(query, engine, params=(store, item, simulated_date))
    df = df.rename(columns={"sale_date": "date", "sale": "sales"})
    df["date"] = pd.to_datetime(df["date"])
    return df