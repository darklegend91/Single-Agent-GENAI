import json
import sqlite3

from app.config import Settings
from scripts.run_local import indexed_customer_ids


def test_default_stack_is_local_ollama(credential_mapping):
    settings = Settings(credential_mapping=credential_mapping)
    assert settings.llm_provider == "ollama"
    assert settings.model_id == "ollama:qwen3:4b-instruct"
    assert settings.ollama_base_url == "http://127.0.0.1:11434"
    assert settings.embedding_provider == "ollama"
    assert settings.embedding_model == "nomic-embed-text:latest"
    assert settings.embedding_dimensions == 768


def test_local_customer_scope_matches_published_sample(tmp_path):
    database = tmp_path / "sample.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE customers (UserID TEXT, source_row INTEGER)")
        connection.executemany("INSERT INTO customers VALUES (?, ?)", [("U0000000002", 3), ("U0000000001", 2)])
    manifest = tmp_path / "active.json"
    manifest.write_text(json.dumps({"sqlite_path": str(database)}), encoding="utf-8")

    assert indexed_customer_ids(manifest) == ["U0000000001", "U0000000002"]
