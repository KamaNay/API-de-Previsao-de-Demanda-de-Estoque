import os
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv

from features import HISTORY_DAYS

ENV_PATH = Path(__file__).parent.parent.parent / ".env"
load_dotenv(ENV_PATH)


def get_connection():
    return psycopg2.connect(os.getenv("DATABASE_URL"))


def get_simulated_date(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT simulated_date FROM sim_state WHERE id = %s", (1,))
        row = cur.fetchone()
    return row[0]


def get_sales_history(conn, store, item, simulated_date, history_days=HISTORY_DAYS):
    query = """
        SELECT store, item, sale_date, sale
        FROM sales_history
        WHERE store = %s AND item = %s AND sale_date <= %s
        ORDER BY sale_date DESC
        LIMIT %s
    """

    with conn.cursor() as cur:
        cur.execute(query, (store, item, simulated_date, history_days))
        rows = cur.fetchall()

    df = pd.DataFrame(rows, columns=["store", "item", "date", "sales"])
    df["date"] = pd.to_datetime(df["date"])
    df["sales"] = df["sales"].astype(float)
    return df
