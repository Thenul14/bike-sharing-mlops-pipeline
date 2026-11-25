"""Run the original pipeline sequentially without Airflow for local development."""
from src.config import DATA_PATH, database_url
from src.ingestion import Ingestion
from src.validation import Validation
from src.cleaning import Cleaning
from src.star import Star
from src.feature_eng import Feature
from src.linear_regression import LinearRegression
from src.random_forest import RandomForest
from src.gradient_boost import GradientBoost
from src.evaluate_models import ModelComparer


def main():
    url = database_url()
    table = "raw_dataset"
    ingestion = Ingestion(str(DATA_PATH), url, table)
    ingestion.write_to_mariadb(ingestion.read_csv())
    Validation(url, table).validate()
    Cleaning(url, table).clean()
    Star(url).star_schema()
    Feature().feature_engineering()
    LinearRegression().train()
    RandomForest().train()
    GradientBoost().train()
    ModelComparer().compare_and_select()


if __name__ == "__main__":
    main()
