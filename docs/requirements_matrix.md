# Requirements and evidence matrix

This audit follows section 14 (deliverables, PDF page 26) and section 16 (definition of done, page 27) of the supplied *Single-Agent GenAI Knowledge Assistant* brief. **Implemented** means code/artifact exists; **measured-synthetic** means local tests/evaluation exercised synthetic data; **unverified-live** means external service/data behavior still requires explicit real configuration. No synthetic measurement is represented as live-model quality.

## Section 14 — required deliverables

| # | Required deliverable | Evidence | Status / remaining evidence |
|---|---|---|---|
| 1 | Complete modular Python source | `app/`, `scripts/`, `pyproject.toml`, pinned `requirements.txt` | Implemented; compileall and automated suite measured locally |
| 2 | Dataset validation, documents and Milvus ingestion | `app/data/quality.py`, `app/data/ingestion.py`, `app/rag/documents.py`, `app/rag/store.py`, `scripts/profile_data.py`, `scripts/ingest_data.py`, `tests/test_ingestion.py` | Implemented; failure/atomic publication measured-synthetic; paid million-row embedding and real Milvus unverified-live |
| 3 | One bounded Pydantic AI agent and typed tools | `app/agent/orchestrator.py`, `app/agent/tools.py`, `tests/test_agent.py` | Measured-synthetic using actual FunctionModel tool loop; configured cloud model unverified-live |
| 4 | LangChain retrieval and evidence context | `app/rag/retriever.py`, `app/rag/context.py`, `tests/test_retrieval.py`, `tests/test_context.py` | Filter, scope, budgets and schema/config checks measured-synthetic; live retrieval unverified-live |
| 5 | System/task prompt library and comparison results | `app/prompts/v1/`, `docs/prompt_comparison.md`, `scripts/evaluate.py` | Library implemented; synthetic regression measured; model-backed baseline comparison unavailable until separately run |
| 6 | Safety/input/injection/policy/output modules | `app/guardrails/`, `app/security.py`, `tests/test_guardrails.py`, `tests/test_output_validation.py`, `tests/test_security.py` | Measured-synthetic; general protection against all possible attacks not claimed |
| 7 | Documented FastAPI and controlled errors | `app/main.py`, `app/api/`, `tests/test_api.py`, `/docs`, `docs/setup.md` | Measured-synthetic: scopes/statuses, limits, timeout, body bound, job lifecycle/cancellation |
| 8 | Langfuse traces, evaluation and monitoring report | `app/observability/telemetry.py`, `tests/test_observability.py`, `docs/observability_report.md`, generated evaluation artifacts | Safe recording/adapter tests measured-synthetic; opt-in live export skipped without credentials; real trace screenshots/receipt unverified-live |
| 9 | Functional/retrieval/security/schema suites | `tests/`, including `tests/test_required_scenarios.py` | Local suite measured-synthetic; live tests explicitly opt-in/skipped |
| 10 | README, diagram, setup, assumptions, limits | Appended `readme.md` links; `docs/setup.md`, `docs/architecture.md`, `docs/data_dictionary.md`, security/observability reports | Implemented; external Docker image pull/service deployment unverified-live |
| 11 | Walkthrough: supported, unsupported, security query | `scripts/demo.py --mode synthetic`, browser at `/`, `tests/test_demo.py` | Runnable and measured-synthetic; cloud walkthrough unverified-live |

## Section 16 — definition of done

| Criterion | Evidence | Assessment |
|---|---|---|
| Answers supported questions from shared data | Configured ingestion/repository/agent; `tests/test_agent.py`, synthetic golden questions | Synthetic supported answers measured; original real dataset + paid provider answering unverified-live |
| Retrieves relevant evidence through Milvus | `VectorStore`, `Retriever`, version/config validation and retrieval tests | Integration implemented; real Milvus relevance/Precision@K/Recall@K unverified-live |
| Uses deterministic exact calculations | Integer monthly paise, `AnalyticsService.revenue`, `tests/test_analytics.py` | Measured-synthetic including values beyond binary floating-point exact range |
| Schema-valid evidence-backed responses | `CustomerAnswer`, strict `AgentResult`, output validator and renderer | Measured-synthetic; evidence-free answered status rejected |
| Handles missing data without invention | Null preservation, incomplete-period flags, unknown-code limitations | Measured-synthetic missing-field/unknown-code/timeline cases |
| Customer access control independent of LLM | Bearer scope, repository/tool auth, filtered retrieval and metadata recheck | Measured-synthetic unauthorized requests/tools/records |
| Layered prompt-injection mitigation | Policy, bounded tools, untrusted context, deterministic claims | Direct/indirect synthetic cases measured; real-model adversarial generalization unverified-live |
| Validates before FastAPI returns output | Assistant output checks and API `response_model=CustomerAnswer` | Measured-synthetic malformed/unsupported/inconsistent claims |
| Observable traces and measurable Langfuse evaluation | Safe stage events, generation usage, validation scores, evaluation summaries | Local metrics measured; real Langfuse trace/evaluation receipt unverified-live |
| Install, test and run using docs | `requirements.txt`, Docker/Compose, setup/CLI help, browser demo | Existing pinned environment, compileall, tests/CLI/demo checked; clean Linux container build/image pulls unverified-live |

## Verification and actual remediation

Initial API/telemetry test collection failed because their modules did not exist, then passed after implementation. The ingestion cancellation regression initially failed because cancelling the async wrapper freed its active slot before the underlying worker thread stopped; the coordinator now shields/drains that worker and the regression passes. The Langfuse grouping test initially failed because observations lacked a shared trace context; observations now share a deterministic trace ID derived from the safe correlation identifier and validation emits a boolean score.

Compose configuration validation initially reported the intentionally absent `.env` file. Structural validation passed with `docker compose --env-file .env.compose.example config --no-env-resolution --quiet`; this does not test application secrets, image availability or service boot. The application `.env` and Compose interpolation credentials are separate to retain strict application settings validation.

Current exact suite counts and generated evaluation metrics are recorded in the delivery report and final response, since parallel integration can add tests after this matrix is written. Full original-dataset embedding, a live model comparison, live Milvus quality/latency, Langfuse screenshots/receipt and production deployment remain unfinished external-service verification, not claimed completion.
