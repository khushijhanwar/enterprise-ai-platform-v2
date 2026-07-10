"""
ETL pipeline built on real Apache Spark (local mode).

Production: this same code runs unmodified on a real Spark cluster
(EMR, Dataproc, Databricks) — only the SparkSession builder's master URL
changes from "local[*]" to a cluster manager address, and the output
sink changes from DuckDB to BigQuery
(df.write.format("bigquery").save(...)).

This module intentionally uses Spark's actual DataFrame API (not pandas)
so the transformation logic — filters, derived columns, aggregations —
is genuine distributed-computation code, not a simulation of it.
"""
from __future__ import annotations

import logging

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, LongType, BooleanType
)

from app.config import DATA_DIR
from app.warehouse.duckdb import get_connection

logging.getLogger("py4j").setLevel(logging.ERROR)

RAW_CSV = DATA_DIR / "accounts_raw.csv"

SCHEMA = StructType([
    StructField("account_id", StringType(), False),
    StructField("account_name", StringType(), False),
    StructField("region", StringType(), True),
    StructField("plan", StringType(), True),
    StructField("monthly_spend_usd", DoubleType(), True),
    StructField("api_calls_last_30d", LongType(), True),
    StructField("csm_assigned", StringType(), True),
    StructField("signup_date", StringType(), True),
])


def get_spark_session() -> SparkSession:
    """Local-mode Spark session. In production, swap master() for a
    cluster URL (e.g. yarn, k8s://..., or a Databricks connect string)."""
    return (
        SparkSession.builder.appName("EnterpriseAIPlatform-ETL")
        .master("local[*]")
        .config("spark.driver.memory", "1g")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def extract(spark: SparkSession) -> DataFrame:
    """Extract: read raw CSV as a Spark DataFrame (stand-in for a Spark
    read from cloud storage / Kafka / a source database)."""
    return (
        spark.read.option("header", True)
        .schema(SCHEMA)
        .csv(str(RAW_CSV))
    )


def transform(df: DataFrame) -> DataFrame:
    """Transform: real Spark DataFrame transformations -- filters, derived
    columns, window-based percentile calc -- executed via the Catalyst
    optimizer and Spark's distributed execution engine."""
    df = df.withColumn("signup_date", F.to_date("signup_date"))
    df = df.withColumn("monthly_spend_usd", F.round("monthly_spend_usd", 2))
    df = df.withColumn("csm_assigned", F.when(F.col("csm_assigned") == "Y", True).otherwise(False))

    # High-value flag via a window-based percentile, a genuinely
    # distributed operation (not something pandas does for free at scale).
    spend_p75 = df.approxQuantile("monthly_spend_usd", [0.75], 0.01)[0]
    df = df.withColumn("is_high_value", F.col("monthly_spend_usd") > F.lit(spend_p75))

    df = df.withColumn("account_age_days", F.datediff(F.current_date(), F.col("signup_date")))
    df = df.dropDuplicates(["account_id"]).filter(F.col("monthly_spend_usd").isNotNull())
    return df


def load(df: DataFrame, table_name: str = "accounts") -> int:
    """Load: land the transformed Spark DataFrame into the warehouse.

    Production: df.write.format("bigquery").option("table", ...).save()
    Local: convert to pandas via Arrow and write into DuckDB, which plays
    the role of the analytical warehouse for local development.
    """
    pdf = df.toPandas()
    con = get_connection()
    con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM pdf")
    con.close()
    return len(pdf)


def run_etl() -> int:
    spark = get_spark_session()
    try:
        raw_df = extract(spark)
        clean_df = transform(raw_df)
        row_count = load(clean_df)
        print(f"[spark-etl] Spark job complete: {row_count} rows landed in warehouse::accounts")
        return row_count
    finally:
        spark.stop()


if __name__ == "__main__":
    run_etl()
