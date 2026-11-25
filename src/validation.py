import pandas as pd
from sqlalchemy import create_engine

class Validation:
    def __init__(self, db_url, table_name):
        self.db_url = db_url
        self.table_name = table_name

    def validate(self):
        engine = create_engine(self.db_url)
        #read raw data into a dataframe
        df = pd.read_sql(self.table_name, con=engine)

        #dataset shape
        print(f"Shape of the dataset: {df.shape} \n")

        #columns in the dataset
        print("----Columns in the dataset----")
        print(f"{df.columns} \n")

        #dataset information and datatypes
        print("----Data types and dataset information----")
        print(f"{df.info()} \n")

        #check for nulls
        print(f"----Number of missing values in each column----")
        print(f"{df.isnull().sum()} \n")

        #check for duplicate values
        print("----Check for duplicated values----")
        print(f"No of duplicates: {df.duplicated().sum()} \n")
