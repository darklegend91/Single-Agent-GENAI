# User guide

This guide covers the local Customer Knowledge Assistant: a Streamlit interface backed by FastAPI, Ollama, and a local Milvus Lite index. It answers questions from ingested customer records that your identity is allowed to read.

## Start the app

Use Python 3.12 or newer. From the repository root, create an environment and install the dependencies:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install Ollama separately, start its service (or open the Ollama desktop app), and download the configured chat and embedding models:

```sh
ollama pull qwen3:4b-instruct
ollama pull nomic-embed-text
```

Start the local stack:

```sh
python -m scripts.run_local
```

The default loads every customer in the CSV into the local SQLite source store and embeds a compact set of discovery summaries in Milvus. With roughly one million rows, the first source load can take a few minutes and use several gigabytes of local disk. When ready, open [the Streamlit app](http://127.0.0.1:8501/). FastAPI runs at [http://127.0.0.1:8000](http://127.0.0.1:8000), its interactive API documentation is at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs), and its readiness endpoint is [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health).

The runner supplies a temporary credential directly to the local Streamlit process. You do not need to create, copy, or enter a bearer token in the page. Press **Ctrl+C** in the terminal running the stack to stop Streamlit and FastAPI.

## Use the three tabs

Enter a customer ID in the sidebar’s **Customer ID** field. To ask about another customer in the shared chat, include that ID in your question; it overrides the sidebar selection. Ask about one customer at a time. The backend checks the ID against all active source records. IDs outside the compact Milvus index are fetched directly from their matching SQLite source row. **Check backend** checks service readiness, and **FastAPI documentation** opens the API reference.

- **Ask assistant**: enter a question in the chat box. The answer shows its status, and expandable sections show source evidence and limitations.
- **Customer details**: select **Load customer summary** to inspect the authorized record grouped by data domain, such as subscription information.
- **Revenue**: select **Load revenue** to see available monthly settled revenue and its total. Values are in paise. The completeness label indicates whether the total covers the full 12-month period.

Try questions such as:

- “Summarize subscription and payments.”
- “What is the monthly revenue?”
- “What is the subscription status?”
- “How many payment attempts are recorded?”
- “Show open support tickets.”
- “What usage aggregates are available?”

Ask for a particular fact or domain when possible. The assistant uses verified, question-matched tools for direct field lookups and monthly revenue; broader questions use Ollama to select from the available read-only tools. Direct lookups can avoid an LLM generation call. Other questions require local model inference and may take longer, especially on the first request while Ollama loads the model.

## Understand answers and data limits

An **answered** status means the response was grounded in the available authorized records. **Insufficient evidence** means the indexed data did not support a reliable answer. A **refused** status means a safety or access rule prevented the request. Check **Source evidence** and **Limitations** before relying on an answer.

Missing or unresolved fields are not evidence of zero or “no.” The app does not infer undocumented category or cancellation code meanings. Revenue is calculated from settled monthly values in the source; it is labelled in paise, and a total over an incomplete period covers only the months available.

For a quick test instead of indexing all customers, stop the app and rerun with an explicit limit, for example:

```sh
python -m scripts.run_local --limit 100
```

With `--limit`, only IDs present in that limited index can be queried. To restore full-dataset access, restart without `--limit`. For a persistent setup with explicit customer permissions, follow [setup and operation](setup.md) and configure the credential mapping and frontend token outside the UI. Keep those credentials private.

## Troubleshooting

- **Ollama connection error**: start the Ollama service, then run `ollama list` and confirm both `qwen3:4b-instruct` and `nomic-embed-text` are installed.
- **A model is missing**: pull it using the commands above, stop the stack, and start it again.
- **Backend is not ready**: check the FastAPI `/health` endpoint and terminal output. Verify Ollama is running and the index finished building.
- **Customer not found**: confirm the ID is present in the CSV and restart without `--limit` if you launched a limited test index.
- **Port 8000 or 8501 is already in use**: stop the other process using that port before launching this runner.
- **Answers lack information**: check whether the requested field exists in the source data. A full index cannot fill in fields absent from the dataset.

## Local data and privacy

The default runner uses local Ollama inference and a Milvus Lite database at `output/milvus.db`; it does not send customer text to a cloud model provider. The generated index and reports are local artifacts. Keep source customer data, tokens, and generated records private, and do not commit them to the repository.

For API routes, custom credentials, ingestion, evaluation, container deployment, and observability configuration, see [setup and operation](setup.md).
