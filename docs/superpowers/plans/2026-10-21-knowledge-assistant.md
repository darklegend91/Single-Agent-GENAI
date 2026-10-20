# Knowledge Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the supplied PDF's secure customer analytics assistant, with executable tests and honest evaluation evidence.

**Architecture:** Stream CSV preparation into versioned SQLite records and Milvus customer summaries. FastAPI authorizes requests before LangChain retrieval and a single Pydantic AI agent; deterministic tools and claim rendering establish the final facts. Langfuse receives redacted stage telemetry.

**Tech Stack:** Python 3.11+, pandas, SQLite, FastAPI, Pydantic 2, LangChain, Milvus, Pydantic AI, Langfuse, pytest, Docker.

**Spec:** `docs/superpowers/specs/2026-10-21-knowledge-assistant-design.md`

## Global Constraints

- Retain integer paise as authoritative; format rupees using Decimal.
- Missing values remain missing.
- Every source lookup and typed tool checks authorization.
- No arbitrary SQL, shell, code, or filesystem tool is available.
- Use LangChain for a bounded retrieval pipeline and prompt templates, not a second agent loop.
- Do not commit dataset or secrets.
- Preserve the existing notebook, scratch script, and README changes.
- Acceptance thresholds for the synthetic regression suite: all 22 scenarios pass; zero unauthorized disclosures and unsupported accepted claims; 100% financial calculation and schema correctness; all labeled expected customer retrievals pass.
- Live integration tests are opt-in and explicitly report skipped dependencies.

## Review Focus

1. CSV values resembling dates or integers must survive ingestion without floating-point precision loss or guessed timestamp semantics (Task 2).
2. A partially failed index build must leave existing requests using one consistent active dataset/index pair (Tasks 2–3).
3. A provider returning a different customer's document or asking tools for another customer must not expose that record (Tasks 3–5).
4. Schema-valid answers containing correct citations but invented factual prose must not reach the API (Task 4).
5. Errors, cancellation, and unavailable tracing must neither expose customer content nor leave ingestion permanently locked (Tasks 5–6).

## Task 1: Contracts, configuration, and authorization

**Files:** Create `pyproject.toml`, `app/__init__.py`, `app/config.py`, `app/errors.py`, `app/schemas.py`, `app/security.py`, `tests/conftest.py`, `tests/test_contracts.py`, `tests/test_security.py`; update `requirements.txt` and `.gitignore` while preserving existing entries.

**Interfaces:** `Settings` validates configuration and limits. Frozen `Principal(subject: str, customer_ids: frozenset[str], ingest_admin: bool)` describes a server-side identity. `authenticate(token: str, settings: Settings) -> Principal` and `authorize(principal: Principal, user_id: str) -> None` enforce scope. `ChatRequest`, `Evidence`, `CustomerAnswer`, `Claim`, and `AgentResult` use extra-forbid Pydantic models; UserID format is `U` followed by 10 decimal digits, matching the dataset and PDF examples. Claims reference a field or approved metric ID and an exact typed value; they contain no free-form factual prose. Question limit defaults to 2,000 characters.

- [ ] Write tests: malformed IDs and oversized/empty questions fail validation; answered responses without evidence fail; unknown body fields fail; unknown tokens and unauthorized customers fail; a client cannot supply permissions; missing credentials have no permissive fallback.
- [ ] Run `python -m pytest tests/test_contracts.py tests/test_security.py -q`; expect import failures before implementation.
- [ ] Implement models, credential mapping with secret-safe repr and constant-time comparison, typed controlled errors, and bounded settings. Verify current official APIs before choosing and locking provider integration dependencies.
- [ ] Re-run the two test files; require all passing. Commit only Task 1 files using explicit paths.

## Task 2: Data profiling, source repository, and deterministic tools

**Files:** Create `app/data/__init__.py`, `app/data/dictionary.py`, `app/data/quality.py`, `app/data/repository.py`, `app/data/ingestion.py`, `app/services/__init__.py`, `app/services/analytics.py`, `scripts/__init__.py`, `scripts/profile_data.py`, `tests/test_data_quality.py`, `tests/test_analytics.py`, and synthetic `tests/fixtures/customers.csv`.

**Interfaces:** `profile_csv(path: Path, chunk_size: int) -> QualityReport`; `SourceRepository.lookup(principal: Principal, user_id: str) -> CustomerRecord`; `AnalyticsService.summary(principal, user_id) -> CustomerSummary`; `AnalyticsService.revenue(principal, user_id) -> RevenueSummary`. `StagedDataset` owns version, SQLite path, and publication state. Final publication is through a single atomically replaced active manifest containing both SQLite path and Milvus collection; no independent publication pointers exist.

- [ ] Write fixture tests for exact paise above `2**53`, duplicate IDs, nullable fields, malformed financial values, explicit timestamp formats, unknown 14-digit dates, invalid months, distinct payment/risk counts, incomplete revenue periods, unknown cancellation codes, and unauthorized repository/tool calls. Assert missing revenue never becomes a complete zero total.
- [ ] Run `python -m pytest tests/test_data_quality.py tests/test_analytics.py -q`; verify expected failures.
- [ ] Implement chunked pandas reads with strings, strict integer conversion, aggregate quality counts, explicit dictionary semantics, parameterized SQLite lookup, indexed primary keys, and fixed domain summaries. Reject invalid required keys/duplicate rows. Preserve unresolved timestamp source values with warnings. Build SQLite in staging, not in the active database.
- [ ] Run the tests; require all passing. Run `python -m scripts.profile_data --csv dataset/customer.csv --output output/reports/data_quality.json`; inspect aggregate-only output and record measured limitations. Do not ingest or embed the million rows at this step.
- [ ] Commit Task 2 code, synthetic fixtures, and sanitized aggregate report using explicit paths.

## Task 3: Milvus ingestion, LangChain retrieval, and bounded context

**Files:** Create `app/rag/__init__.py`, `app/rag/documents.py`, `app/rag/embeddings.py`, `app/rag/store.py`, `app/rag/retriever.py`, `app/rag/context.py`, `scripts/ingest_data.py`, `tests/test_ingestion.py`, `tests/test_retrieval.py`, and `tests/test_context.py`.

**Interfaces:** `build_document(record: CustomerRecord) -> Document`; `VectorStore.stage(documents, version, embedding_signature) -> str`; `Retriever.retrieve(principal, user_id, question) -> list[RetrievedEvidence]`; `build_context(question, principal, user_id, documents, results, settings) -> ContextBundle`; `ingest(settings, limit: int | None = None) -> IngestionResult`. Context and vector interfaces are dependency-injectable for tests. Defaults: top-K 3, maximum 3 documents, 12,000 context characters; deployment config selects the relevance threshold for the chosen distance metric.

- [ ] Write tests proving filtering is passed to vector search before top-K, foreign metadata is rejected afterward, irrelevant results yield no context, embedding version mismatch fails, deterministic document IDs prevent duplicate ingestion, missing domains stay missing, malicious retrieved instructions cannot change permissions, and context respects both document and character limits.
- [ ] Add ingestion failure tests at SQLite completion, vector-write failure, and manifest publication; assert the old active manifest remains unchanged. Add success/repeat tests checking one coherent dataset/index pair and bounded batching. Explicit sample limits must be recorded in ingestion results.
- [ ] Run the three Task 3 test files; verify failures. Implement document generation with LangChain Document, one embedding adapter/configuration, Milvus server-filtered retrieval, batching, staging/atomic manifest publication, and context allowlisting. Clean abandoned staging artifacts without deleting the active version.
- [ ] Re-run Task 3 tests and earlier tests; require all passing. Add opt-in real-Milvus tests, report skip reasons if Milvus/provider credentials are absent, and never label fake retrieval as live Milvus verification.
- [ ] Commit Task 3 files using explicit paths.

## Task 4: Guardrails, structured claims, and factual rendering

**Files:** Create `app/guardrails/__init__.py`, `app/guardrails/input.py`, `app/guardrails/policy.py`, `app/guardrails/output.py`, `tests/test_guardrails.py`, `tests/test_output_validation.py`.

**Interfaces:** `check_input(question: str) -> PolicyDecision`; `validate_result(result: AgentResult, context: ContextBundle, tool_evidence: EvidenceLedger) -> CustomerAnswer`. `EvidenceLedger` contains only authorized retrieved fields and deterministic metric results. Rendering uses field labels/value formatting and fixed limitation templates; arbitrary model-provided prose is not rendered as factual output.

- [ ] Write tests for direct/indirect injection, internal prompt/secret disclosure, personal inference, unsupported fraud labels, undocumented cancellation meanings, unavailable event timelines/ticket descriptions, nonexistent field citations, wrong units, changed amounts, mismatched UserIDs, and fabricated computed metrics.
- [ ] Add a schema-valid result with legitimate citations plus malicious extra prose; assert extra prose is rejected or cannot be represented. Assert invoiced/settled values and pending/confirmed risk remain distinct, empty source values cannot support claims, and correct financial citations with incorrect values fail.
- [ ] Run Task 4 tests and confirm failures. Implement deterministic policy checks, exact claim verification, fixed factual rendering, and controlled refusals/insufficient-evidence results. Document that injection patterns alone are not a security guarantee.
- [ ] Re-run Task 4 and prior tests; require all passing. Commit Task 4 files.

## Task 5: One Pydantic AI agent and approved typed tools

**Files:** Create `app/agent/__init__.py`, `app/agent/tools.py`, `app/agent/orchestrator.py`, `app/prompts/v1/system.txt`, `app/prompts/v1/customer.txt`, `app/prompts/v1/payment.txt`, `app/prompts/v1/revenue.txt`, `app/prompts/baseline/system.txt`, and `tests/test_agent.py`.

**Interfaces:** frozen `AgentDependencies` carries principal, requested UserID, context, repository/services, and evidence ledger. `Assistant.answer(principal: Principal, request: ChatRequest) -> CustomerAnswer` is async. Approved typed tools cover customer record, subscription, payment, revenue, support, risk, engagement, referrals, feedback, and rewards; each independently calls authorization and restricts its user_id to the request scope.

- [ ] Write agent tests using a Pydantic AI test/function model for typed tool invocation and output. Cover another-customer tool requests, missing customer/fields, unsupported questions, tool failures, malformed outputs, invalid facts, one bounded repair, model failure, and timeout. Assert no competing LangChain agent loop or unrestricted tool is registered.
- [ ] Run `python -m pytest tests/test_agent.py -q`; verify failures. Verify installed Pydantic AI usage-limit and output-validator APIs against official documentation before implementing adapters.
- [ ] Implement retrieval-first orchestration, task prompts, immutable scope, authorized typed tools, validated output, maximum 4 model requests, maximum 8 tool calls, one output repair, and 30-second request deadline by default. No relevant context yields an abstention without generation. Cancellation must propagate to provider operations that support it.
- [ ] Run agent and regression tests; require all passing. Add opt-in live-model test with explicit environment gating. Commit Task 5 files.

## Task 6: FastAPI, ingestion jobs, and Langfuse

**Files:** Create `app/main.py`, `app/api/__init__.py`, `app/api/dependencies.py`, `app/api/routes.py`, `app/api/jobs.py`, `app/observability/__init__.py`, `app/observability/telemetry.py`, `tests/test_api.py`, and `tests/test_observability.py`.

**Interfaces:** `create_app(settings: Settings | None = None, services: AppServices | None = None) -> FastAPI`. `Telemetry.stage(name, correlation_id, safe_metadata)` records bounded allowlisted metadata. An injectable recording adapter supports assertions; the runtime adapter exports Langfuse observations without raw inputs/outputs. `IngestionJobs.start(principal) -> JobStatus` allows one active job; `get(principal, job_id) -> JobStatus` requires admin.

- [ ] Write API tests for all five required endpoints plus ingestion-job status, 401/403/404/422/429/503/504 behavior, correlation headers, unknown JSON fields, concurrent ingestion rejection, secret-free errors, and health readiness without customer data. Rate limit defaults to 60 authenticated requests/minute per principal; use an injected clock in tests.
- [ ] Write telemetry tests for the nine request stages, retrieval/validation failures, latency and token usage, explicitly unknown costs, pseudonymous IDs, no raw query/record/credential leakage, and degraded export. Test ingestion worker exceptions/cancellation releasing its lock and preserving the old active version.
- [ ] Run Task 6 tests; verify failures. Implement app lifespan wiring, bounded middleware, protected routes, worker-thread ingestion, request/job cleanup, controlled errors, and Langfuse spans/generations/scores using verified official APIs. Do not expose a successful dependency-ready status when required services are unavailable.
- [ ] Re-run API, observability, and all regression tests. Add opt-in Langfuse export verification; report absent credentials as skipped, not successful. Commit Task 6 files.

## Task 7: Golden evaluation and all 22 required scenarios

**Files:** Create `evaluation/golden_questions.jsonl`, `evaluation/security_cases.jsonl`, `evaluation/acceptance.json`, `scripts/evaluate.py`, `scripts/demo.py`, `tests/test_evaluation.py`, and `tests/test_required_scenarios.py`.

**Interfaces:** `evaluate(cases_path: Path, assistant, mode: Literal['synthetic','live']) -> EvaluationReport`. The report contains actual per-case outcomes, support references, retrieval labels/ranks, correctness, schema status, timings, token/cost availability, and explicit skipped/unmeasured metrics. The baseline uses the same cases and is labeled clearly; only accepted, validated output counts as grounded system output.

- [ ] Write 22 individually named tests matching PDF section 13, not one aggregate test. Reuse earlier fixtures/adapters for expected answers and failure injection. Link each scenario number to its test name in the report generator.
- [ ] Add metric tests with known ranks and times; assert exact Precision@K/Recall@K, correct abstention/unsupported-claim/disclosure denominators, median/p95 values, and no fabricated cost or baseline comparison when a run is absent.
- [ ] Run Task 7 tests; verify failures. Implement golden/security case loading, scenario mapping, evaluation CLI, and supported/unsupported/injection demo. Include every functional domain in golden cases and explicitly distinguish fake-model behavior tests from live answer-quality evaluation.
- [ ] Run `python -m pytest -q`, then `python -m scripts.evaluate --mode synthetic --output output/reports/evaluation.json` and `python -m scripts.demo --mode synthetic`; inspect measured results against acceptance.json. Run live evaluation only if services and credentials are available.
- [ ] Commit Task 7 files and sanitized reports.

## Task 8: Reproducible delivery and requirements audit

**Files:** Create `Dockerfile`, `compose.yaml`, `.dockerignore`, `.env.example`, `docs/setup.md`, `docs/architecture.md`, `docs/data_dictionary.md`, `docs/security_report.md`, `docs/observability_report.md`, `docs/prompt_comparison.md`, and `docs/requirements_matrix.md`; append run/setup links to existing `readme.md` without replacing user content.

**Interfaces:** `python -m scripts.profile_data`, `python -m scripts.ingest_data`, `python -m scripts.evaluate`, `python -m scripts.demo`, and `uvicorn app.main:app --host 127.0.0.1 --port 8000` form the documented operator interface. Docker/compose supports Milvus and the application with explicit credential configuration and health checks.

- [ ] Document actual CLI flags and environment settings from implemented parsers, bearer scope setup, sample ingestion before full embedding, server deployment, explicit timestamp mapping, unresolved dictionary meanings, one-worker limiter/job limitations, and Langfuse credential setup. Describe embeddings as customer data transmission when using a remote provider so deployment choice is informed.
- [ ] Audit every PDF section 14 deliverable and section 16 completion criterion into requirements_matrix.md, pointing to code/tests/reports with implemented, measured, unverified-live, or unavailable evidence status. Record actual remediation from failed tests; never invent attack results, trace screenshots, prompt wins, or live cost measurements.
- [ ] Verify `python -m pytest -q`, `python -m compileall -q app scripts`, each CLI `--help`, evaluation/demo commands, OpenAPI generation, and compose configuration validation if Docker is installed. Inspect dependency installation and clean-start instructions in a disposable environment when feasible. Broaden checks only for a new failure/change.
- [ ] Complete an independent whole-change review if the approved execution method includes one; fix findings and rerun affected checks. Check `git diff --check` and explicit staged file paths for customer data or credentials before any commit.
- [ ] Update plan checkboxes from actual results; record remaining external-service verification limits. Provide run instructions, test counts, and precise unfinished live requirements in the delivery response.

## Plan self-review

All spec sections have owners: data/dictionary (Task 2), index/retrieval/context (Task 3), security/claims (Tasks 1, 4), single-agent/prompts (Task 5), API/tracing/jobs (Task 6), required scenarios/metrics/demo (Task 7), packaging/reports/audit (Task 8). The five review-focus conditions have concrete owning tests. Shared interfaces use the same names, and atomic publication is one manifest operation. Default limits are explicit; cloud verification remains separate from local synthetic acceptance. Product implementation starts after the user reviews this plan and selects execution method.
