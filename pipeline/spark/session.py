"""Spark session wired to a local Iceberg JDBC (SQLite) catalog named `lake`.

The same catalog is read by pyiceberg (dbt-duckdb plugin, ranking code), so Spark and dbt
share one set of Iceberg tables.
"""
import os
from pyspark.sql import SparkSession

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WAREHOUSE = os.path.join(ROOT, "data", "lake", "warehouse")
CATALOG_DB = os.path.join(ROOT, "data", "lake", "catalog.db")
PKGS = ",".join([
    "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.6.1",
    "org.xerial:sqlite-jdbc:3.46.1.0",
])


def spark(app="labmatch"):
    os.makedirs(WAREHOUSE, exist_ok=True)
    return (SparkSession.builder.appName(app)
            .master(os.environ.get("SPARK_MASTER", "local[*]"))
            .config("spark.jars.packages", PKGS)
            .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions")
            .config("spark.sql.catalog.lake", "org.apache.iceberg.spark.SparkCatalog")
            .config("spark.sql.catalog.lake.catalog-impl", "org.apache.iceberg.jdbc.JdbcCatalog")
            .config("spark.sql.catalog.lake.uri", f"jdbc:sqlite:{CATALOG_DB}")
            .config("spark.sql.catalog.lake.jdbc.schema-version", "V1")
            .config("spark.sql.catalog.lake.warehouse", f"file://{WAREHOUSE}")
            .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "8g"))
            .config("spark.sql.shuffle.partitions", "16")
            .getOrCreate())
