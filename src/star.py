import pandas as pd
from sqlalchemy import create_engine
from src.config import redis_client
import pyarrow as pa
import pyarrow.ipc as ipc

class Star:
    def __init__(self, db_url):
        self.db_url = db_url
        self.redis_client = redis_client()

    def star_schema(self):
        engine = create_engine(self.db_url)

        retrieved_data = self.redis_client.get("cleaned_dataset")
        reader = pa.ipc.open_stream(pa.BufferReader(retrieved_data))
        table = reader.read_all()

        df = table.to_pandas()

        #time dimension
        dim_time=df[['date', 'yr', 'mnth', 'day', 'hr', 'weekday', 'workingday', 'holiday', 'season']].drop_duplicates().reset_index(drop=True)
        dim_time['time_id'] = dim_time.index
        print(f"----Time dimension table created successfully----\n")

        #weather dimension
        dim_weather = df[['weather_situation', 'temp', 'atemp', 'hum', 'windspeed']].drop_duplicates().reset_index(drop=True)
        dim_weather['weather_id'] = dim_weather.index
        print(f"----Weather dimension table created successfully----\n")

        df = df.merge(dim_time, on=['date', 'yr', 'mnth', 'day', 'hr', 'weekday', 'workingday', 'holiday', 'season'])
        df = df.merge(dim_weather, on=['weather_situation', 'temp', 'atemp', 'hum', 'windspeed'])

        #rentals fact table
        fact = df[['cnt', 'casual', 'registered', 'time_id', 'weather_id']]
        print(f"----Fact table created successfully----\n")

        #storing them in mariadb
        dim_time.to_sql('dim_time', con=engine, if_exists='replace', index=False)
        dim_weather.to_sql('dim_weather', con=engine, if_exists='replace', index=False)
        fact.to_sql('fact_rentals', con=engine, if_exists='replace', index=False)
        print(f"----Saved successfully----\n")
