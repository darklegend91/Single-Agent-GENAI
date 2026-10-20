"""Read-only source repository. Every customer lookup authorizes before SQL."""

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from typing import Iterator

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr

from app.data.dictionary import FIELD_DICTIONARY
from app.errors import DependencyError, NotFoundError, ValidationFailure
from app.schemas import UserID
from app.security import Principal, authorize

SourceValue = StrictInt | StrictBool | StrictStr | None


class FieldProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset_version: str
    schema_version: str
    source_sha256: str
    source_row: int = Field(ge=2)
    source_field: str


class CustomerRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    user_id: UserID
    values: dict[str, SourceValue] = Field(repr=False)
    provenance: dict[str, FieldProvenance]
    normalized_timestamps: dict[str, str] = Field(default_factory=dict, repr=False)
    limitations: list[str] = Field(default_factory=list)
    field_limitations: dict[str, list[str]] = Field(default_factory=dict)


class SourceRepository:
    """The path is trusted server configuration, never a client tool argument."""

    def __init__(self, sqlite_path: Path):
        self.sqlite_path = Path(sqlite_path)

    @contextmanager
    def _connection(self):
        connection = None
        try:
            connection = sqlite3.connect(f"{self.sqlite_path.resolve().as_uri()}?mode=ro", uri=True)
            yield connection
        except (sqlite3.Error, OSError):
            raise DependencyError() from None
        finally:
            if connection is not None:
                connection.close()

    def _decode(self, row, metadata: dict[str, str]) -> CustomerRecord:
        try:
            payload = json.loads(row[1])
            values = payload["values"]
            if set(values) - FIELD_DICTIONARY.keys() or values.get("UserID") != row[0]:
                raise ValueError
            return CustomerRecord(user_id=row[0], values=values, normalized_timestamps=payload["normalized_timestamps"], limitations=payload["limitations"], field_limitations=payload["field_limitations"], provenance={field: FieldProvenance(dataset_version=metadata["version"], schema_version=metadata["schema_version"], source_sha256=metadata["source_sha256"], source_row=row[2], source_field=field) for field in values})
        except (ValueError, KeyError, TypeError):
            raise DependencyError() from None

    def lookup(self, principal: Principal, user_id: str) -> CustomerRecord:
        authorize(principal, user_id)
        with self._connection() as connection:
            row = connection.execute("SELECT UserID, payload, source_row FROM customers WHERE UserID = ?", (user_id,)).fetchone()
            if row is None:
                raise NotFoundError()
            metadata = dict(connection.execute("SELECT key, value FROM dataset_metadata"))
            return self._decode(row, metadata)

    def iter_records(self, batch_size: int = 1000) -> Iterator[CustomerRecord]:
        """Trusted ingestion/indexing only; never expose as a model tool or API."""
        if type(batch_size) is not int or not 1 <= batch_size <= 100_000:
            raise ValidationFailure()
        with self._connection() as connection:
            metadata = dict(connection.execute("SELECT key, value FROM dataset_metadata"))
            cursor = connection.execute("SELECT UserID, payload, source_row FROM customers ORDER BY source_row")
            while rows := cursor.fetchmany(batch_size):
                for row in rows:
                    yield self._decode(row, metadata)
