import pandas as pd
from sqlalchemy import create_engine

class Ingestion:
    def __init__(self, csv_path: str, db_url: str, table_name: str):
        self.csv_path = csv_path
        self.db_url = db_url
        self.table_name = table_name

    def read_csv(self):
        #read csv file into dataframe
        df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(df)} rows from {self.csv_path}")
        return df

    def write_to_mariadb(self, df):
        #Writes DataFrame to MariaDB using SQLAlchemy
        engine = create_engine(self.db_url)
        df.to_sql(name=self.table_name, con=engine, if_exists="replace", index=False)
        print(f"Stored DataFrame into MariaDB table: {self.table_name}")
