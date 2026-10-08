from db import (
    get_connection,
    get_engine,
    get_sales_history,
    get_simulated_date,
)

conn = get_connection()
engine = get_engine()
simulated_date = get_simulated_date(conn)
df = get_sales_history(engine, store=1, item=1, simulated_date=simulated_date)

print(df.dtypes)
print(df.head())