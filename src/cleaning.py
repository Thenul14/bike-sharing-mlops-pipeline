import pandas as pd
from sqlalchemy import create_engine
from src.config import redis_client
import pyarrow as pa
import pyarrow.ipc as ipc

class Cleaning:
    def __init__(self, db_url, table_name):
        self.db_url = db_url
        self.table_name = table_name
        self.redis_client = redis_client()



    def clean(self):
        engine= create_engine(self.db_url)
        df = pd.read_sql(self.table_name, con=engine)

        #extract day from dtdey column
        df['dteday'] = pd.to_datetime(df['dteday'])
        df.insert(5, 'day', df['dteday'].dt.day)
        print(f"----New day column added.----\n")

        #rename columns
        df.rename(columns={'weathersit': 'weather_situation'}, inplace=True)
        df.rename(columns={'dteday': 'date'}, inplace=True)
        print(f"----Columns renamed----\n")

        #storing cleaned dataset in mariadb
        df.to_sql("cleaned_dataset", con=engine, if_exists='replace', index=False)
        print(f"----Cleaning completed----\n")

        #storing cleaned dataset in redis store
        table = pa.Table.from_pandas(df)

        sink = pa.BufferOutputStream()
        writer = pa.ipc.new_stream(sink, table.schema)
        writer.write_table(table)
        writer.close()

        self.redis_client.set("cleaned_dataset",sink.getvalue().to_pybytes())

        print(f"----Cleaned dataset stored in Redis----\n")
