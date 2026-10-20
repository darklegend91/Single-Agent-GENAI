"""Validated environment configuration; no usable credentials are defaulted."""

from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping, Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.security import Principal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KA_", env_file=".env", extra="forbid", frozen=True, hide_input_in_errors=True, allow_inf_nan=False)

    credential_mapping: Mapping[str, Principal] = Field(repr=False, min_length=1)
    dataset_path: Path = Path("dataset/customer.csv")
    database_path: Path = Path("output/customers.sqlite3")
    active_manifest_path: Path = Path("output/active.json")
    staging_dir: Path = Path("output/staging")
    ingestion_batch_size: int = Field(default=1000, ge=1, le=100_000)
    timestamp_formats: dict[str, str] = Field(default_factory=dict)

    milvus_uri: str = "http://localhost:19530"
    milvus_token: SecretStr | None = Field(default=None, repr=False)
    milvus_collection: str = Field(default="customer_knowledge", pattern=r"\A[A-Za-z_][A-Za-z0-9_]{0,127}\z")
    embedding_provider: Literal["openai"] = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=1536, ge=1, le=65536)
    schema_version: str = "1"
    model_id: str = "openai:gpt-4.1-mini"
    openai_api_key: SecretStr | None = Field(default=None, repr=False)
    langfuse_public_key: SecretStr | None = Field(default=None, repr=False)
    langfuse_secret_key: SecretStr | None = Field(default=None, repr=False)
    langfuse_base_url: str = "https://cloud.langfuse.com"

    top_k: int = Field(default=3, ge=1, le=100)
    max_documents: int = Field(default=3, ge=1, le=100)
    context_max_chars: int = Field(default=12000, ge=1, le=1_000_000)
    context_max_tokens: int = Field(default=4000, ge=1, le=250_000)
    relevance_threshold: float = Field(default=0.5, ge=-1, le=1)
    question_max_chars: int = Field(default=2000, ge=1, le=2000)
    model_request_limit: int = Field(default=4, ge=1, le=20)
    tool_call_limit: int = Field(default=8, ge=1, le=100)
    output_retries: int = Field(default=1, ge=0, le=1)
    request_timeout_seconds: float = Field(default=30, gt=0, le=300)
    rate_limit_per_minute: int = Field(default=60, ge=1, le=10000)
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)

    @field_validator("credential_mapping")
    @classmethod
    def safe_credentials(cls, mapping: Mapping[str, Principal]) -> Mapping[str, Principal]:
        if any(not token or token != token.strip() or len(token) > 4096 for token in mapping):
            raise ValueError("Credentials must be nonempty bounded tokens without surrounding whitespace")
        try:
            for token in mapping:
                token.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("Credentials must contain valid UTF-8 text") from None
        return MappingProxyType(dict(mapping))

    @model_validator(mode="after")
    def coherent_limits(self) -> Self:
        if self.max_documents > self.top_k:
            raise ValueError("Maximum documents cannot exceed retrieval top-K")
        return self
