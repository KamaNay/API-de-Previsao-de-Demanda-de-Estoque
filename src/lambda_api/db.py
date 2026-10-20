import logging
import os
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv

from features import HISTORY_DAYS

ENV_PATH = Path(__file__).parent.parent.parent / ".env"
load_dotenv(ENV_PATH)

_conn = None
_logger = logging.getLogger(__name__)

def get_connection():
    global _conn
    if _conn is None or _conn.closed:
        _conn = psycopg2.connect(os.getenv("DATABASE_URL"), connect_timeout=5)
        _conn.autocommit = True
    return _conn

def reset_connection():
    global _conn
    if _conn is not None and not _conn.closed:
        try:
            _conn.close()
        except psycopg2.Error:
            _logger.exception("Failed to close database connection")
    _conn = None
    
class DatabaseUnavailableError(Exception):
    pass
    
def fetch_all(query, params=None):
    for attempt in range(3):
        try:
            conn = get_connection()
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchall()
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            _logger.warning(f"Database connection error: {e}. Attempt {attempt + 1} of 3.")
            reset_connection()
            if attempt == 2:
                raise DatabaseUnavailableError("Database is unavailable after 3 attempts.") from e
            
def get_simulated_date():
    query = "SELECT simulated_date FROM sim_state WHERE id = %s"
    rows = fetch_all(query, (1,))
    return rows[0][0] if rows else None


def get_sales_history(store, item, simulated_date, history_days=HISTORY_DAYS):
    query = """
        SELECT store, item, sale_date, sale
        FROM sales_history
        WHERE store = %s AND item = %s AND sale_date <= %s
        ORDER BY sale_date DESC
        LIMIT %s
    """

    rows = fetch_all(query, (store, item, simulated_date, history_days))
    df = pd.DataFrame(rows, columns=["store", "item", "date", "sales"])
    df["date"] = pd.to_datetime(df["date"])
    df["sales"] = df["sales"].astype(float)
    return df
