"""LabMatch end-to-end DAG.

grants (FundedLabs NIH/NSF extracts) -> Iceberg bronze ----\
OpenAlex resolve -> OpenAlex works -> Iceberg bronze/silver --> dbt build -> embed (SPECTER2)
                                                                     -> experiment (BM25, dense, hybrid, LambdaRank) -> demo export
Runs weekly; every task is idempotent (Iceberg createOrReplace, OpenAlex checkpoints).
"""
from datetime import datetime, timedelta
import os

from airflow import DAG
from airflow.operators.bash import BashOperator

ROOT = os.environ.get("LABMATCH_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
PY = os.path.join(ROOT, ".venv", "bin", "python")
ENV = {"LABMATCH_ROOT": ROOT, "PATH": os.environ.get("PATH", "")}


def task(task_id, cmd, **kw):
    return BashOperator(task_id=task_id, bash_command=f"cd {ROOT} && {cmd}", env=ENV, append_env=True, **kw)


with DAG(
    dag_id="labmatch",
    start_date=datetime(2026, 11, 1),
    schedule="@weekly",
    catchup=False,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=10)},
    tags=["labmatch"],
) as dag:
    bronze_grants = task("bronze_grants", f"{PY} -m pipeline.spark.bronze_grants")
    dbt_seed_models = task("dbt_build_grants", f"cd dbt && {ROOT}/.venv/bin/dbt build --profiles-dir . --threads 1")
    oa_resolve = task("openalex_resolve", f"{PY} -m pipeline.openalex.ingest resolve")
    oa_works = task("openalex_works", f"{PY} -m pipeline.openalex.ingest works")
    bronze_oa = task("bronze_openalex", f"{PY} -m pipeline.spark.bronze_openalex")
    dbt_all = task("dbt_build_all", f"cd dbt && {ROOT}/.venv/bin/dbt build --profiles-dir . --threads 1")
    embed = task("embed_specter2", f"{PY} -m ranking.embed", execution_timeout=timedelta(hours=4))
    experiment = task("train_and_evaluate", f"{PY} -m ranking.experiment")
    export = task("export_demo", f"{PY} -m ranking.export_demo")

    bronze_grants >> dbt_seed_models >> oa_resolve >> oa_works >> bronze_oa >> dbt_all >> embed >> experiment >> export
