from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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


# ---------------- CONFIG ---------------- #
rawdata_table = "raw_dataset"


# ---------------- TASK FUNCTIONS ---------------- #

def run_ingestion():
    ingestion = Ingestion(str(DATA_PATH), database_url(), rawdata_table)
    df = ingestion.read_csv()
    ingestion.write_to_mariadb(df)


def run_validation():
    validation = Validation(database_url(), rawdata_table)
    validation.validate()


def run_cleaning():
    cleaning = Cleaning(database_url(), rawdata_table)
    cleaning.clean()


def run_star():
    star = Star(database_url())
    star.star_schema()


def run_feature():
    feature = Feature()
    feature.feature_engineering()


def train_lr():
    lr = LinearRegression()
    lr.train()


def train_rf():
    rf = RandomForest()
    rf.train()

def train_gb():
    gb = GradientBoost()
    gb.train()


def evaluate_models():
    evaluate = ModelComparer()
    evaluate.compare_and_select()


# ---------------- DAG ---------------- #

default_args = {
    "owner": "bikeshare_team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}

with DAG(
    dag_id="bike_ml_pipeline",
    default_args=default_args,
    description="Bike sharing ML pipeline",
    schedule_interval=None,
    start_date=datetime(2026, 4, 8),
    catchup=False,
    tags=["bikeshare", "mlops"],
    max_active_runs=1,
) as dag:

    ingestion_task = PythonOperator(
        task_id="ingestion",
        python_callable=run_ingestion
    )

    validation_task = PythonOperator(
        task_id="validation",
        python_callable=run_validation
    )

    cleaning_task = PythonOperator(
        task_id="cleaning",
        python_callable=run_cleaning
    )

    star_task = PythonOperator(
        task_id="star_schema",
        python_callable=run_star
    )

    feature_task = PythonOperator(
        task_id="feature_engineering",
        python_callable=run_feature
    )

    lr_task = PythonOperator(
        task_id="train_linear_regression",
        python_callable=train_lr
    )

    rf_task = PythonOperator(
        task_id="train_random_forest",
        python_callable=train_rf
    )

    gb_task = PythonOperator(
        task_id="train_gradient_boost",
        python_callable=train_gb
    )

    evaluate_task = PythonOperator(
        task_id="model_evaluation",
        python_callable=evaluate_models
    )


# ---------------- DEPENDENCIES ---------------- #

    ingestion_task >> validation_task >> cleaning_task >> star_task >> feature_task

    feature_task >> lr_task
    feature_task >> rf_task
    feature_task >> gb_task

    [lr_task, rf_task, gb_task] >> evaluate_task
