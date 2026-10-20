"""SQLite staging only. Task 3 publishes one manifest with the matching vector index."""

import json
from pathlib import Path
import re
import sqlite3
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings
from app.data.dictionary import BOOLEAN_FIELDS, FIELD_DICTIONARY, INTEGER_FIELDS, TIMESTAMP_FIELDS
from app.data.quality import BOOLEAN_VALUES, USER_ID_PATTERN, csv_columns, invalid_month_headers, is_missing, normalize_timestamp, read_chunks, source_checksum, strict_integer, valid_month
from app.errors import DependencyError, ValidationFailure


class StagedDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: str
    sqlite_path: Path
    row_count: int = Field(ge=0)
    schema_version: str
    source_sha256: str
    sample_limit: int | None = None
    publication_state: Literal["staged"] = "staged"


def _payload(raw: dict[str, str], formats: dict[str, str]) -> dict:
    values = {}
    normalized = {}
    field_limitations = {}
    for field, value in raw.items():
        if is_missing(value):
            values[field] = None
            field_limitations[field] = ["missing_field"]
        elif field in INTEGER_FIELDS:
            values[field] = strict_integer(value, field)
        elif field in BOOLEAN_FIELDS:
            if value not in BOOLEAN_VALUES:
                raise ValidationFailure()
            values[field] = BOOLEAN_VALUES[value]
        else:
            values[field] = value
            if field in TIMESTAMP_FIELDS:
                timestamp = normalize_timestamp(value, formats.get(field))
                if timestamp is None:
                    field_limitations[field] = ["unresolved_timestamp"]
                else:
                    normalized[field] = timestamp
            elif field == "LatestRewardMonth" and not valid_month(value):
                field_limitations[field] = ["unresolved_timestamp"]
            elif FIELD_DICTIONARY[field].kind == "code":
                field_limitations[field] = ["unknown_code"]
    return {"values": values, "normalized_timestamps": normalized, "field_limitations": field_limitations, "limitations": sorted({limitation for limitations in field_limitations.values() for limitation in limitations})}


def stage_dataset(settings: Settings, *, limit: int | None = None, version: str | None = None) -> StagedDataset:
    """Use configured source/staging paths; reject invalid data and remove failed staging."""
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValidationFailure()
    version = version if version is not None else uuid4().hex
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", version, flags=re.ASCII):
        raise ValidationFailure()
    columns = csv_columns(settings.dataset_path)
    if "UserID" not in columns or set(columns) - FIELD_DICTIONARY.keys() or invalid_month_headers(columns) or set(settings.timestamp_formats) - TIMESTAMP_FIELDS:
        raise ValidationFailure()
    checksum = source_checksum(settings.dataset_path)
    settings.staging_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = settings.staging_dir / f"{version}.sqlite3"
    # Exclusive creation prevents overwriting active or previously completed artifacts.
    try:
        with sqlite_path.open("xb"):
            pass
    except OSError:
        raise DependencyError() from None
    row_count = 0
    try:
        with sqlite3.connect(sqlite_path) as connection:
            connection.execute("CREATE TABLE customers (UserID TEXT PRIMARY KEY, payload TEXT NOT NULL, source_row INTEGER NOT NULL) WITHOUT ROWID")
            connection.execute("CREATE TABLE dataset_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID")
            for chunk in read_chunks(settings.dataset_path, settings.ingestion_batch_size):
                if limit is not None:
                    chunk = chunk.iloc[: max(0, limit - row_count)]
                batch = []
                for raw in chunk.to_dict(orient="records"):
                    user_id = raw["UserID"]
                    if not USER_ID_PATTERN.fullmatch(user_id):
                        raise ValidationFailure()
                    batch.append((user_id, json.dumps(_payload(raw, settings.timestamp_formats), separators=(",", ":"), ensure_ascii=True), row_count + 2))
                    row_count += 1
                connection.executemany("INSERT INTO customers VALUES (?, ?, ?)", batch)
                if limit is not None and row_count >= limit:
                    break
            metadata = {"version": version, "schema_version": settings.schema_version, "source_sha256": checksum, "row_count": str(row_count), "sample_limit": str(limit) if limit is not None else "", "state": "staged"}
            connection.executemany("INSERT INTO dataset_metadata VALUES (?, ?)", metadata.items())
            connection.commit()
        return StagedDataset(version=version, sqlite_path=sqlite_path.resolve(), row_count=row_count, schema_version=settings.schema_version, source_sha256=checksum, sample_limit=limit)
    except BaseException as error:
        sqlite_path.unlink(missing_ok=True)
        if isinstance(error, sqlite3.IntegrityError):
            raise ValidationFailure() from None
        if isinstance(error, (sqlite3.Error, OSError)):
            raise DependencyError() from None
        raise
