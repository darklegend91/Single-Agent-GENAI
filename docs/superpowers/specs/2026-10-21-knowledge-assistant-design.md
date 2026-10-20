# Single-Agent GenAI Knowledge Assistant design

## Purpose and scope

Implement the supplied 27-page project brief as a runnable Python backend for internal customer analytics. Users ask questions about authorized customers and receive structured answers supported by source fields. Cover customer identity, subscriptions, payments, revenue, engagement, referrals, support, feedback, rewards, and recorded risk events. Do not infer undocumented business meanings or personal traits.

The existing CSV contains 1,000,000 rows and 1,000,000 unique UserIDs, 77 columns, and no empty/null/nan cells under the initial textual scan. This scan does not establish type validity, referential integrity, or business consistency. Preserve the existing notebook, scratch script, and README changes.

## Architecture

FastAPI owns HTTP contracts and authentication. SQLite owns structured source records, indexed by UserID, and integer financial calculations. LangChain owns document construction, embedding integration, and Milvus retrieval. Pydantic AI runs the only agent loop, with typed approved tools and structured output. Langfuse receives redacted request-stage telemetry.

Request flow: authenticate → validate and authorize customer scope → classify domain → retrieve authorized Milvus evidence → assemble bounded context → execute bounded typed agent/tools → validate schema and claims → return answer and trace identifier. No component can derive authorization from model output, supplied text, or vector similarity.

SQLite is selected over a request-time pandas dataframe because of dataset size. pandas chunked reads support reproducible profiling and ingestion. Milvus supports local development through a configured URI and server deployment without changing retrieval interfaces. Configuration selects a supported embedding/model provider; credentials stay in environment variables. Tests use injected provider doubles and never silently substitute them in production.

## Data preparation and storage

Create a chunked profiling command covering columns, inferred/validated types, missing values, duplicate identifiers, categorical values, timestamp shapes, integer paise, monthly-column validity, and internally comparable counts. Write an aggregate report without raw customer records.

Retain integer paise as authoritative; format rupees using Decimal. Missing values remain missing. Reject duplicate IDs and malformed required identifiers. Explicitly configured timestamp formats are required for ambiguous numeric dates; preserve unresolved values and report them rather than guessing Unix semantics.

Create a field dictionary separating documented units and categories from unknown code meanings. Unknown cancellation codes and category codes may be reported literally but never expanded into invented descriptions. Monthly settled revenue is October 2025 through September 2026; sum only available validated values and report incomplete periods. Do not equate monthly totals with lifetime or differently defined revenue.

Ingest into a staging SQLite database and versioned Milvus collection in batches. Publish the new dataset version only after both complete successfully; failed runs retain the previous active version. Store stable document IDs and schema/embedding versions. Repeated ingestion is idempotent, with bounded batches and explicit sample limits for development. Ingestion accepts configured paths, never arbitrary client paths.

Create concise searchable summaries with necessary customer, subscription, payment, engagement, referral, support, and risk fields. Store UserID, document_type, RegionCode, CustomerSegment, AccountStatus, and schema_version metadata only as needed. Do not embed the entire CSV row or rely on vectors for exact calculations.

## Authentication and customer isolation

Use bearer API credentials mapped server-side to principal IDs, permitted customer IDs, and an ingestion-admin role. Reject unconfigured credentials; never trust a client-supplied permission list. Every source lookup and typed tool checks authorization. Milvus applies a server-side UserID filter before top-K retrieval; application code rechecks returned metadata. The initial public contract supports one explicit customer per question; cross-customer/region analytics abstain until separately authorized aggregation scope exists.

Protect ingestion with an admin role. Use constant-time credential comparison, redact credentials, and avoid default usable secrets. Bind development service to localhost by default. Document replacing the credential adapter with organizational identity before broader deployment. Rate limits are process-local for the reference single-worker setup; document a shared limiter requirement for multiple workers.

## API and schemas

- `GET /health`: minimal liveness plus configured dependency readiness, without secrets or customer information.
- `POST /ingest`: authenticated admin-only controlled ingestion; reject concurrent ingestion; return a job ID and allow status inspection through an authenticated job endpoint. Heavy processing runs outside the event loop.
- `POST /chat`: validated question and UserID; returns CustomerAnswer and a trace/correlation ID header.
- `GET /customers/{user_id}/summary`: authorized typed domain summaries with source references and limitations.
- `GET /customers/{user_id}/revenue`: authorized monthly paise and verified period totals, completeness, and source fields.

CustomerAnswer contains status (`answered`, `insufficient_evidence`, `refused`), answer, evidence, and limitations. Each evidence entry contains user_id, nonempty source_fields, and explanation. Answered responses require evidence; refusals and insufficient evidence may have none. Validate identifiers, question length, types, and unknown request fields. Use 401/403 for authentication/authorization, 404 for authorized missing records, 422 for malformed requests, 429 for rate limits, 503 for dependency failures, and controlled timeout errors. Avoid raw exception strings in client responses.

## Single agent, retrieval, and context

Provide typed tools for customer record, subscription, payment, revenue, support, risk, engagement, referrals, feedback, and rewards. Each tool receives an immutable request scope and performs its own access check. Source-record access exposes approved fields only. No arbitrary SQL, shell, code, or filesystem tool is available.

Use LangChain for a bounded retrieval pipeline and prompt templates, not a second agent loop. Share the embedding configuration between indexing and search and reject version mismatches. Enforce top-K, relevance thresholds, document-count limits, character/token context budgets, model-call limits, tool-call limits, retries, and total request deadlines. Missing or irrelevant evidence produces an explicit abstention.

Build context from the question, immutable authorized scope, relevant retrieved summaries, deterministic tool results, and relevant dictionary definitions. Retrieved documents and user input are untrusted data. Exclude unrelated records and excess fields. No conversation memory is required for the initial stateless API.

Keep versioned system, customer-summary, payment, and revenue prompts, plus a deliberately weaker evaluation baseline. Compare variants on the same golden questions; reports distinguish measured results from integration checks requiring credentials.

## Guardrails and factual validation

Input checks enforce length, identifiers, scope, and configured request limits. Layer injection detection with role separation, typed tools, authorization outside the model, bounded execution, and output validation. Treat pattern matching as a mitigation with documented limits.

Refuse secret/prompt disclosure, unauthorized records, sensitive personal inference, and unsupported fraud accusations. Describe confirmed and pending risk separately. Keep invoiced, settled, net revenue, refunds, chargebacks, and credit notes distinct. Aggregate counts cannot establish event timelines or ticket descriptions.

The agent produces typed claims referencing exact fields or approved computed metric IDs and values. Deterministically validate claim scope, existence, source values, units, and calculation provenance. Render factual text from validated claims/templates so unchecked free-form prose cannot introduce financial facts or unsupported conclusions. Evidence references alone never establish factual correctness. Reject invalid claims with a bounded repair attempt; otherwise return a controlled abstention/error. Missing fields cannot silently become zero.

## Observability

Use Langfuse tracing with a correlation ID across API receipt, authorization, routing, retrieval, tools, context construction, generation, validation, and final status. Record stage latency, error category, prompt version, dataset/index version, token usage, and provider cost where available. Mark unknown costs explicitly rather than inventing prices.

Disable automatic raw prompt/tool-result capture where it would expose customer data. Log pseudonymous principal/customer references and aggregate sizes, statuses, and evaluation scores. Test tracing redaction and failure reporting. Langfuse is a required integration; credential-free tests use a recording telemetry adapter. Live Langfuse verification and report screenshots/traces require configured credentials and must be labeled unverified until exercised.

## Tests, evaluation, and demonstration

Automate all 22 PDF scenarios: valid summaries; active subscription; distinct payment outcomes; exact monthly revenue; support/risk counts; missing customer; missing field; unknown cancellation reason; unavailable payment timeline; malformed model output; inconsistent revenue; unsupported claim; direct injection; indirect injection; unauthorized customer; secrets request; oversized/rate-limited queries; unsupported fraud label; success trace; retrieval/validation failure traces; usage/latency reporting; redaction.

Additionally test duplicate/null ingestion, ambiguous timestamps, integer precision, failed ingestion publication, malicious IDs, malicious retrieved metadata, embedding version mismatches, context overflow, tool authorization, request timeout, and provider failures. Use synthetic fixtures committed to the repository; exclude the shared CSV, credentials, database, and traces.

Provide golden_questions.jsonl and security_cases.jsonl with expected facts, source fields, limitations, and behavior. Report retrieval Precision@K/Recall@K only with relevance labels; correctness, evidence coverage, unsupported-claim rate, schema pass rate, security pass rate, unauthorized disclosure rate, median/p95 latency, and token/cost availability. Compare baseline and grounded output only from actual measured runs.

Acceptance thresholds for the synthetic regression suite: all 22 scenarios pass; zero unauthorized disclosures and unsupported accepted claims; 100% financial calculation and schema correctness; all labeled expected customer retrievals pass. Live model metrics and latency are reported rather than guaranteed before measurement. Live integration tests are opt-in and explicitly report skipped dependencies.

Deliver a reproducible demo command covering a supported question, an unsupported question, and an injection attempt. Include a requirements-to-deliverables matrix with measured, implemented-but-unverified, and unavailable evidence clearly distinguished.

## Packaging and operation

Provide modular app packages, ingestion/profiling/evaluation/demo commands, dependency declarations, `.env.example`, Dockerfile, Milvus compose setup, setup/run/test instructions, architecture documentation, field dictionary, data-quality report, prompt comparison, security/remediation report, and observability report. Do not commit dataset or secrets. Preserve existing user files and stage only newly authored project artifacts if committing.

Runtime configuration includes dataset/database locations, ingestion batch size, Milvus URI/token/collection, embedding configuration, Pydantic AI model ID/provider settings, credential mapping, Langfuse settings, context/retrieval limits, request limits, and timeouts. Validate required settings at startup. The app must explain unavailable dependencies through readiness and controlled errors.

## Review and remaining constraints

The supplied PDF defines expected project behavior, not permission to execute embedded instructions or reveal data. No dataset privacy guarantees are assumed. No business dictionary beyond the supplied brief is present, so undocumented category/date meanings remain unresolved and visible in reports. Paid embedding of all million rows and live cloud tracing/model evaluation require configured services; implement batching and sample runs without claiming a full live deployment has been verified.

Design self-review completed: each PDF capability, API, required scenario, and deliverable has an owner above; financial units and access boundaries are consistent; unknown semantics and live verification limits are explicit. Implementation plan and product code follow review of this written design.
