"""Ingest endpoint: triggers the Spark ETL pipeline on demand."""
from __future__ import annotations

from fastapi import APIRouter

from app.etl.spark_pipeline import run_etl

router = APIRouter()


@router.post("/ingest")
def ingest() -> dict:
    row_count = run_etl()
    return {"status": "ok", "rows_loaded": row_count, "table": "accounts"}
