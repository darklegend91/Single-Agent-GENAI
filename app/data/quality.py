"""Chunked, aggregate-only profiling; customer identifiers never enter the report."""

from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
from pathlib import Path
import re
import sqlite3
import tempfile
from typing import Iterator

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from app.data.dictionary import BOOLEAN_FIELDS, CATEGORICAL_FIELDS, FIELD_DICTIONARY, INTEGER_FIELDS, TIMESTAMP_FIELDS
from app.errors import ValidationFailure

USER_ID_PATTERN = re.compile(r"U[0-9]{10}\Z", re.ASCII)
UNSIGNED_PATTERN = re.compile(r"[0-9]+\Z", re.ASCII)
SIGNED_PATTERN = re.compile(r"-?[0-9]+\Z", re.ASCII)
REVENUE_HEADER = re.compile(r"SettledRevenue(\d+)Paise\Z", re.ASCII)
BOOLEAN_VALUES = {"0": False, "1": True, "false": False, "true": True, "False": False, "True": True}


class QualityReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    row_count: int
    columns: list[str]
    missing_columns: list[str]
    unknown_columns: list[str]
    source_sha256: str
    missing_counts: dict[str, int]
    invalid_integer_counts: dict[str, int]
    invalid_boolean_counts: dict[str, int]
    invalid_user_id_count: int
    duplicate_user_id_count: int
    timestamp_shapes: dict[str, dict[str, int]]
    unresolved_timestamp_counts: dict[str, int]
    invalid_timestamp_counts: dict[str, int]
    invalid_month_columns: list[str]
    invalid_month_counts: dict[str, int]
    categorical_distinct_counts: dict[str, int]
    consistency_violation_counts: dict[str, int]
    warnings: list[str] = Field(default_factory=lambda: ["Undocumented category and cancellation codes have no inferred meanings.", "Monthly settled revenue is not equated with lifetime settled or net revenue.", "Missing cells are blank/whitespace only; nonempty sentinel text is validated literally.", "Aggregate comparisons do not establish business consistency, event chronology, or referential integrity."])


def source_checksum(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def csv_columns(path: Path) -> list[str]:
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            columns = next(csv.reader(handle))
        if not columns or len(columns) != len(set(columns)) or any(not column for column in columns):
            raise ValidationFailure()
        return columns
    except (OSError, UnicodeError, csv.Error, StopIteration):
        raise ValidationFailure() from None


def read_chunks(path: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
    if type(chunk_size) is not int or not 1 <= chunk_size <= 100_000:
        raise ValidationFailure()
    try:
        yield from pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False,
                               chunksize=chunk_size, encoding="utf-8-sig", on_bad_lines="error")
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
        raise ValidationFailure() from None


def is_missing(value: str) -> bool:
    return not value.strip()


def strict_integer(value: str, field: str) -> int:
    pattern = SIGNED_PATTERN if FIELD_DICTIONARY[field].signed else UNSIGNED_PATTERN
    if not pattern.fullmatch(value):
        raise ValidationFailure()
    try:
        return int(value)
    except ValueError:
        raise ValidationFailure() from None


def valid_month(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9]{4}(?:0[1-9]|1[0-2])", value, flags=re.ASCII)) and value[:4] != "0000"


def normalize_timestamp(value: str, fmt: str | None) -> str | None:
    if not fmt:
        return None
    try:
        result = datetime.strptime(value, fmt)
        # UTC is explicit in the source column heading, never guessed epoch units.
        result = result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)
        return result.isoformat()
    except (ValueError, OverflowError):
        return None


def timestamp_shape(value: str) -> str:
    if re.fullmatch(r"[0-9]+", value, flags=re.ASCII):
        return f"digits_{min(len(value), 128)}" if len(value) <= 128 else "digits_over_128"
    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}(?:[T ].*)?", value, flags=re.ASCII):
        return "iso_like_unverified"
    return "other"


def invalid_month_headers(columns: list[str]) -> list[str]:
    return [column for column in columns if (match := REVENUE_HEADER.fullmatch(column)) and not valid_month(match[1])]


def profile_csv(path: Path, chunk_size: int = 20_000, *, timestamp_formats: dict[str, str] | None = None) -> QualityReport:
    """Bounded dataframe memory; disk-backed exact key/category distinct counts."""
    formats = timestamp_formats or {}
    if set(formats) - TIMESTAMP_FIELDS:
        raise ValidationFailure()
    columns = csv_columns(path)
    missing = Counter({column: 0 for column in columns})
    invalid_ints = Counter({column: 0 for column in columns if column in INTEGER_FIELDS})
    invalid_bools = Counter({column: 0 for column in columns if column in BOOLEAN_FIELDS})
    shapes = {column: Counter() for column in columns if column in TIMESTAMP_FIELDS}
    unresolved = Counter({column: 0 for column in shapes})
    invalid_dates = Counter({column: 0 for column in shapes})
    month_counts = Counter({"LatestRewardMonth": 0}) if "LatestRewardMonth" in columns else Counter()
    consistency = Counter({"payment_outcomes_exceed_attempts": 0, "risk_outcomes_exceed_events": 0, "ticket_outcomes_exceed_total": 0, "paid_referrals_exceed_registered": 0})
    rows = invalid_ids = duplicates = 0
    with tempfile.TemporaryDirectory(prefix="ka-quality-") as directory, sqlite3.connect(Path(directory) / "distinct.sqlite3") as connection:
        connection.execute("CREATE TABLE ids (value TEXT PRIMARY KEY) WITHOUT ROWID")
        connection.execute("CREATE TABLE categories (field TEXT, value TEXT, PRIMARY KEY(field,value)) WITHOUT ROWID")
        for chunk in read_chunks(path, chunk_size):
            rows += len(chunk)
            for column in columns:
                values = chunk[column]
                absent = values.str.strip().eq("")
                missing[column] += int(absent.sum())
                populated = values[~absent]
                if column in INTEGER_FIELDS:
                    pattern = r"-?[0-9]+" if FIELD_DICTIONARY[column].signed else r"[0-9]+"
                    invalid_ints[column] += int((~populated.str.fullmatch(pattern)).sum())
                elif column in BOOLEAN_FIELDS:
                    invalid_bools[column] += int((~populated.isin(BOOLEAN_VALUES)).sum())
                elif column in TIMESTAMP_FIELDS:
                    shapes[column].update(populated.map(timestamp_shape))
                    if column in formats:
                        invalid = int(populated.map(lambda value: normalize_timestamp(value, formats[column]) is None).sum())
                        invalid_dates[column] += invalid
                        unresolved[column] += invalid
                    else:
                        unresolved[column] += len(populated)
                elif column == "LatestRewardMonth":
                    month_counts[column] += int((~populated.map(valid_month)).sum())
                if column in CATEGORICAL_FIELDS:
                    connection.executemany("INSERT OR IGNORE INTO categories VALUES (?, ?)", ((column, value) for value in populated.unique()))
            if "UserID" not in columns:
                invalid_ids += len(chunk)
            else:
                valid_ids = chunk["UserID"].str.fullmatch(r"U[0-9]{10}")
                invalid_ids += int((~valid_ids).sum())
                identifiers = chunk.loc[valid_ids, "UserID"]
                before = connection.total_changes
                connection.executemany("INSERT OR IGNORE INTO ids VALUES (?)", ((value,) for value in identifiers))
                duplicates += len(identifiers) - (connection.total_changes - before)
            for fields, name in [(("SuccessfulPaymentCount", "FailedPaymentCount", "PaymentAttemptCount"), "payment_outcomes_exceed_attempts"), (("ConfirmedRiskEvents", "PendingRiskEvents", "RiskEventCount"), "risk_outcomes_exceed_events"), (("OpenTicketCount", "ResolvedTicketCount", "SupportTicketCount"), "ticket_outcomes_exceed_total")]:
                if set(fields) <= set(columns):
                    valid = pd.Series(True, index=chunk.index)
                    for field in fields:
                        valid &= chunk[field].str.fullmatch(r"[0-9]+")
                    # Python integers retain precision and avoid fixed-width overflow.
                    values = chunk.loc[valid, list(fields)].map(int).astype(object)
                    consistency[name] += int((values[fields[0]] + values[fields[1]] > values[fields[2]]).sum())
            if {"DirectReferralsPaid", "DirectReferralsRegistered"} <= set(columns):
                valid = chunk["DirectReferralsPaid"].str.fullmatch(r"[0-9]+") & chunk["DirectReferralsRegistered"].str.fullmatch(r"[0-9]+")
                consistency["paid_referrals_exceed_registered"] += int((chunk.loc[valid, "DirectReferralsPaid"].map(int) > chunk.loc[valid, "DirectReferralsRegistered"].map(int)).sum())
            connection.commit()
        categories = {column: 0 for column in columns if column in CATEGORICAL_FIELDS}
        categories.update(dict(connection.execute("SELECT field, COUNT(*) FROM categories GROUP BY field")))
    return QualityReport(row_count=rows, columns=columns, missing_columns=sorted(set(FIELD_DICTIONARY) - set(columns)), unknown_columns=sorted(set(columns) - set(FIELD_DICTIONARY)), source_sha256=source_checksum(path), missing_counts=dict(missing), invalid_integer_counts=dict(invalid_ints), invalid_boolean_counts=dict(invalid_bools), invalid_user_id_count=invalid_ids, duplicate_user_id_count=duplicates, timestamp_shapes={field: dict(count) for field, count in shapes.items()}, unresolved_timestamp_counts=dict(unresolved), invalid_timestamp_counts=dict(invalid_dates), invalid_month_columns=invalid_month_headers(columns), invalid_month_counts=dict(month_counts), categorical_distinct_counts=categories, consistency_violation_counts=dict(consistency))
