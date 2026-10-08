import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

DATA_PATH = Path(__file__).parent.parent / "data" / "train.csv" 

def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.rename(columns={"date": "sale_date", "sales": "sale"})  

    url = os.getenv("DATABASE_URL").replace("postgresql://", "postgresql+psycopg2://", 1)
    engine = create_engine(url)

    df.to_sql(
        "sales_history",
        engine,
        if_exists="append",
        index=False,
        chunksize=10000,
        method="multi",
    )
    print(f"{len(df)} linhas inseridas.")

if __name__ == "__main__":
    main()