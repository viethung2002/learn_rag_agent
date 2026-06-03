from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator  # type: ignore
from arxiv_ingestion.user_upload import process_user_uploaded_paper

default_args = {
    "owner": "arxiv-curator",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "user_paper_processing",
    default_args=default_args,
    description="Process user-uploaded PDF papers: parse → index → store",
    schedule=None,  # Manual trigger only (via API)
    catchup=False,
    tags=["user-upload", "paper-processing"],
    start_date=datetime(2025, 1, 1),
)

process_task = PythonOperator(
    task_id="process_user_paper",
    python_callable=process_user_uploaded_paper,
    dag=dag,
)
