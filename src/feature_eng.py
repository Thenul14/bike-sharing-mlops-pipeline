import pandas as pd
from src.config import redis_client
import pyarrow as pa
import pyarrow.ipc as ipc

class Feature:
    def __init__(self):
        self.redis_client = redis_client()

    def feature_engineering(self):
        retrieved_data = self.redis_client.get("cleaned_dataset")
        reader = pa.ipc.open_stream(pa.BufferReader(retrieved_data))
        table = reader.read_all()

        df = table.to_pandas()

        #dropping irrelavant columns
        df = df.drop(columns=["instant", "date"])

        # Convert to PyArrow Table
        table = pa.Table.from_pandas(df)

        # Serialize using BufferOutputStream
        sink = pa.BufferOutputStream()
        writer = pa.ipc.new_stream(sink, table.schema)
        writer.write_table(table)
        writer.close()

        # Store in Redis
        self.redis_client.set("feature_engineered_data", sink.getvalue().to_pybytes())

        print(f"----Feature Engineering stored in Redis successfully---- \n")
