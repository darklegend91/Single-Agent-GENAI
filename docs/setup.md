# Setup and operation

Python 3.12 or newer is required. Run commands from the repository root. The reference service uses one worker.

## Local synthetic demo

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m scripts.serve_demo --port 8000
```

Open http://127.0.0.1:8000. The page labels itself **Synthetic demo** and uses a deterministic Pydantic AI provider, a temporary SQLite database, and synthetic retrieval. It makes no paid model/embedding calls, reads no real customer CSV, and creates a random synthetic-only bearer token kept in page memory. API reference: http://127.0.0.1:8000/docs. This demonstrates the application flow; it does not establish live Milvus/provider quality.

```sh
python -m scripts.demo --mode synthetic
python -m scripts.evaluate --mode synthetic
python -m pytest -q
```

## Local Ollama, FastAPI, and Streamlit

Install Ollama and download the models once:

```sh
ollama pull qwen3:4b-instruct
ollama pull nomic-embed-text
```

Start the local app with the full customer dataset:

```sh
python -m scripts.run_local
```

Open **http://127.0.0.1:8501** for Streamlit. FastAPI is at http://127.0.0.1:8000 and its API reference at http://127.0.0.1:8000/docs. The runner starts both services, creates a short-lived local-only credential and passes it directly to Streamlit; the UI does not ask you to paste a token. FastAPI still performs all customer authorization. Press Ctrl+C to stop both services. The default uses Ollama at `127.0.0.1:11434`, `qwen3:4b-instruct`, `nomic-embed-text`, and local Milvus Lite at `output/milvus.db`; no customer text is sent to a cloud model provider.

The runner loads every source record into SQLite and embeds a compact set of 100 discovery summaries in local Milvus Lite. The shared chat can query any customer ID in the full source dataset: IDs missing from the vector index are retrieved directly from that customer’s SQLite row. The short-lived local credential is dataset-scoped and accepted only when the API binds to a loopback address; every requested ID is verified against the active source before retrieval. Exact field and monthly-revenue questions use the question-matched read tool and deterministic evidence renderer, avoiding an LLM generation call for direct lookups. Other questions are routed to only the relevant fixed tools and retain the Pydantic AI validation path. Retrieved claims must match authorized tool results exactly.

Swagger shows an **Authorize** button for protected routes. Click it and enter the bearer token configured for the API (without adding the word `Bearer`); then use **Try it out**. The local runner generates a private temporary token for Streamlit by default. If you also want to call the API from Swagger, choose a local token before starting it, for example `export KA_FRONTEND_TOKEN='your-long-random-local-token'`, then run `python -m scripts.run_local` and enter that same value in Swagger's Authorize dialog. Keep the token private and out of source control. `/health` remains public; `/chat`, customer records, and ingestion remain protected.

The source has roughly one million rows, so the first full SQLite load can take a few minutes and use several gigabytes of local disk. For a quick smoke test, choose a bounded sample explicitly:

```sh
python -m scripts.profile_data --csv dataset/customer.csv --output output/reports/data_quality.json
python -m scripts.run_local --limit 100
```

To run a persistent API configured for your own credentials, copy `.env.example` to `.env`, replace both sample tokens in `KA_CREDENTIAL_MAPPING`, then run `python -m scripts.ingest_data --limit 100`, `uvicorn app.main:app --factory --host 127.0.0.1 --port 8000 --workers 1`, and `streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501` in separate terminals. An ingestion administrator has no implicit customer-read permissions. Keep `.env` private. Full ingestion is selected only by omitting `--limit` from `scripts.ingest_data`.

The profiler accepts `--csv`, `--output`, `--chunk-size`, and repeatable `--timestamp-format FIELD=FORMAT`. Ingestion accepts only `--limit`; source/destination paths are server configuration. Set `KA_TIMESTAMP_FORMATS` to a JSON object of source field names to explicit `strptime` patterns. Missing mappings remain unresolved. Category and cancellation code meanings are not invented.

The local runner limits Milvus discovery embeddings with `KA_VECTOR_INDEX_LIMIT` (default `100`). Any requested customer outside that compact set is still available through exact-ID source retrieval. Set this to a larger value only when you specifically need broader semantic vector coverage; it does not affect which customers can be queried.

`GET /health` reports service readiness without record contents. Missing auth configuration prevents startup. Missing provider/index dependencies result in unready health and controlled 503 responses, while authenticated admin ingestion can bootstrap the index. Successful ingestion publishes a coherent SQLite/vector manifest and refreshes subsequent requests. A request already running retains its original assistant snapshot. `POST /ingest` starts a background job; `GET /ingest/{job_id}` requires admin scope. Jobs are process-local and capped at 1,000 retained statuses.

Protected APIs:

- `POST /chat`: JSON `{"user_id":"U0000000001","question":"What is the monthly revenue?"}`.
- `GET /customers/{user_id}/summary` and `GET /customers/{user_id}/revenue`.
- `POST /ingest` and `GET /ingest/{job_id}`.

Use the `Authorization: Bearer <token>` header for direct API calls. Responses carry `X-Correlation-ID`. Unknown JSON fields, whitespace-only questions and invalid identifiers are rejected. Request bodies are limited to 16 KiB. Defaults: 60 authenticated requests/minute/principal, 120-second local-model deadline, 3 retrieved documents, 12,000 context characters and 4,000 context tokens. See `app/config.py` for the complete validated configuration.

## Container operation

```sh
docker compose --env-file .env.compose config --quiet
docker compose --env-file .env.compose up --build -d
# Bootstrap sample ingestion inside the application container:
docker compose --env-file .env.compose exec app python -m scripts.ingest_data --limit 100
```

Restart the app after CLI ingestion so its cached production service snapshot refreshes, or ingest using the authenticated API to refresh automatically. Application and Milvus ports bind to localhost; storage service ports are internal. Docker image pull/build and the Milvus/MinIO/etcd combination require live verification on the deployment host. The API health check remains unhealthy until its required index/provider configuration exists; this does not prevent `exec` or the bootstrap endpoint.

For deployment beyond localhost, place the service behind TLS and an organizational identity gateway. Replace the static credential adapter as appropriate. Multiple workers/replicas require a shared rate limiter and durable job coordinator; this reference deliberately runs one worker. Keep SQLite/index volumes persistent and back up the active manifest together with its referenced artifacts. Shutdown drains an active ingestion worker rather than killing it during publication.

## Langfuse

Create a Langfuse project and configure `KA_LANGFUSE_PUBLIC_KEY`, `KA_LANGFUSE_SECRET_KEY`, and `KA_LANGFUSE_BASE_URL` in the backend environment or `.env`; keep the secret key out of source control. Restart FastAPI after setting them. The health endpoint should report telemetry as `configured`. In Streamlit's **Security & accuracy → Langfuse tracing**, select **Check Langfuse connection** to verify the project credentials, then ask a question and review its stage trace in the Langfuse project. The app exports bounded operational metadata, timing, and provider token counts only; it does not export customer questions, prompts, answers, or raw records, and automatic model instrumentation is disabled. If credentials are omitted, Langfuse remains disabled. The pinned Langfuse 5 SDK requires a compatible Langfuse backend. See [the official SDK reference](https://python.reference.langfuse.com/langfuse) and [instrumentation guidance](https://langfuse.com/docs/observability/sdk/instrumentation). Live trace receipt still needs verification against your configured project.
