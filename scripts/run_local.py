"""Index the full dataset (or an optional sample) and run the local stack."""
import argparse
import json
import os
import secrets
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import httpx


def indexed_customer_ids(manifest_path="output/active.json"):
    """Return only IDs in the currently published local sample."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    with sqlite3.connect(f"file:{manifest['sqlite_path']}?mode=ro", uri=True) as connection:
        return [row[0] for row in connection.execute("SELECT UserID FROM customers ORDER BY source_row")]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Optional row cap for a quick test index; omit to index every customer")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    # Keep the generated credential private by default. Operators who need to
    # call the API from Swagger can provide a local-only value in advance.
    token = os.environ.get("KA_FRONTEND_TOKEN") or secrets.token_urlsafe(32)
    env = os.environ.copy()
    env.update({
        # The local runner creates a short-lived, loopback-only dataset identity
        # after ingestion. The source repository checks that each ID exists.
        "KA_CREDENTIAL_MAPPING": json.dumps({token: {"subject": "local-analyst", "customer_ids": [], "ingest_admin": False}}),
        "KA_FRONTEND_TOKEN": token,
        "KA_API_URL": "http://127.0.0.1:8000",
        "KA_LLM_PROVIDER": "ollama",
        "KA_OLLAMA_BASE_URL": "http://127.0.0.1:11434",
        "KA_MODEL_ID": "ollama:qwen3:4b-instruct",
        "KA_EMBEDDING_PROVIDER": "ollama",
        "KA_EMBEDDING_MODEL": "nomic-embed-text:latest",
        "KA_EMBEDDING_DIMENSIONS": "768",
        # Full records live in SQLite; a compact seed index avoids embedding a
        # million customer summaries that are fetched exactly by customer ID.
        "KA_VECTOR_INDEX_LIMIT": os.environ.get("KA_VECTOR_INDEX_LIMIT", "100"),
        "KA_MILVUS_URI": "output/milvus.db",
    })

    if args.limit is None:
        print(f"Loading every customer into SQLite and embedding {env['KA_VECTOR_INDEX_LIMIT']} summaries into Milvus…", flush=True)
        ingest_command = [sys.executable, "-m", "scripts.ingest_data"]
    else:
        print(f"Indexing up to {args.limit} rows with local Ollama embeddings…", flush=True)
        ingest_command = [sys.executable, "-m", "scripts.ingest_data", "--limit", str(args.limit)]
    subprocess.run(ingest_command, env=env, check=True)

    env["KA_CREDENTIAL_MAPPING"] = json.dumps({token: {"subject": "local-analyst", "customer_ids": [], "ingest_admin": False, "dataset_scope": True}})
    env.pop("KA_FRONTEND_CUSTOMERS", None)
    env["KA_FRONTEND_CUSTOMER"] = "U0000000001"
    if args.limit is not None:
        customers = indexed_customer_ids()
        if not customers:
            raise RuntimeError("The local index contains no customers.")
        env["KA_FRONTEND_CUSTOMERS"] = json.dumps(customers)
        env["KA_FRONTEND_CUSTOMER"] = customers[0]

    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--factory", "--host", "127.0.0.1", "--port", "8000"], env=env)
    try:
        for _ in range(100):
            if api.poll() is not None:
                raise RuntimeError("FastAPI stopped during startup.")
            try:
                response = httpx.get("http://127.0.0.1:8000/health", timeout=1, trust_env=False)
                if response.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(.2)
        else:
            raise RuntimeError("FastAPI did not become ready in time.")

        print("FastAPI ready at http://127.0.0.1:8000; Streamlit at http://127.0.0.1:8501", flush=True)
        print("The temporary local credential is supplied privately to Streamlit.", flush=True)
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py", "--server.address", "127.0.0.1", "--server.port", "8501", "--browser.gatherUsageStats", "false"], env=env, check=False)
    except KeyboardInterrupt:
        pass
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1
    finally:
        api.send_signal(signal.SIGTERM)
        try:
            api.wait(timeout=5)
        except subprocess.TimeoutExpired:
            api.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
