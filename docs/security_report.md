# Security and remediation report

Authentication is a server-configured bearer mapping with constant-time credential comparison. Scope is explicit per customer; admin ingestion access does not grant record access. API authorization, tool authorization, filtered retrieval and returned-document checks are independent of model instructions. Parameters cannot supply SQL, filesystem paths, shell commands or authorization lists.

Layered prompt-injection mitigations include untrusted-data delimiters, policy checks, one bounded typed agent, independent source authorization, context budgets, model/tool limits and deterministic claim validation/rendering. Pattern detection is a mitigation, not a proof against every future injection. Unsupported personal inference, secret/prompt extraction and fraud accusations are refused. Aggregate-only evidence cannot justify event details.

API request validation excludes unknown JSON fields and suppresses raw validation inputs. Provider failures return controlled errors without raw exception strings. Tokens and record content are excluded from manual telemetry; pseudonyms use per-process random HMAC salt. Request body size, rate and timeout limits bound common abuse paths. The UI uses text nodes for answers/evidence and keeps credentials in page memory.

Observed remediation: a cancellation regression test demonstrated that cancelling an ingestion task could release its job lock while the Python worker thread continued. The job now shields and drains the worker before releasing the active slot. Error tests verify provider exception text and invalid payload contents do not enter client responses. See `tests/test_api.py`, `tests/test_security.py`, `tests/test_guardrails.py` and `tests/test_rag.py` for executable cases where present.

Synthetic tests are local evidence only. No penetration test of a live deployment, cloud identity integration, paid-model adversarial campaign or real customer-data disclosure scan is claimed. Single-worker process-local rate limits and jobs require shared infrastructure before horizontal deployment. Deploy beyond localhost only with appropriately configured TLS/identity controls.
