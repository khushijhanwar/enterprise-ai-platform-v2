"""
Warehouse layer: DuckDB locally, BigQuery-compatible interface.

DuckDB is used here specifically because it speaks near-identical SQL to
BigQuery (both are columnar, both support standard analytical SQL), so
queries written against this layer port to BigQuery with no rewrite --
only the connection/execute methods below change.

Production swap:
  get_connection()  -> bigquery.Client(project=...)
  execute_query()   -> client.query(sql).to_dataframe()
"""
from __future__ import annotations

import duckdb
import pandas as pd

from app.config import DUCKDB_PATH


def get_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(DUCKDB_PATH)


def execute_query(sql: str) -> pd.DataFrame:
    """Run a SQL query against the warehouse and return a DataFrame.
    This is the single choke point that would change to a BigQuery client
    call in production -- every caller in the codebase goes through here."""
    con = get_connection()
    try:
        return con.execute(sql).fetchdf()
    finally:
        con.close()


def table_exists(table_name: str) -> bool:
    con = get_connection()
    try:
        result = con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name],
        ).fetchone()
        return result[0] > 0
    finally:
        con.close()


SCHEMA_DESCRIPTION = """
Table: accounts
Columns:
  account_id VARCHAR, account_name VARCHAR, region VARCHAR, plan VARCHAR,
  monthly_spend_usd DOUBLE, api_calls_last_30d BIGINT,
  csm_assigned BOOLEAN, signup_date DATE, is_high_value BOOLEAN,
  account_age_days INTEGER
"""
