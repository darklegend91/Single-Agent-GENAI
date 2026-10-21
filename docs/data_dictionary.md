# Data dictionary and assumptions

The executable dictionary is `app/data/dictionary.py`: all 77 source fields are grouped into customer, subscription, payments, revenue, engagement, referrals, support, feedback, risk and rewards domains. Each field records kind/unit and semantic limitations. `UserID` is a `U` followed by exactly ten digits; duplicate identifiers are rejected during ingestion.

Financial source fields remain integer paise. Monthly settled revenue covers October 2025 through September 2026, and sums available validated monthly fields only. A partial period is explicitly incomplete; no available monthly fields produces null, not zero. Decimal formatting converts paise to rupees without binary floating-point loss. Lifetime, invoiced, settled, net, refunded, charged-back and credit-note amounts retain their distinct source meanings.

Unmapped category/cancellation/region codes are literal codes; no independent business dictionary was supplied. Ambiguous numeric timestamp encodings are preserved and flagged unresolved until an explicit source-field format is configured. Counts are aggregates: payment counts cannot establish transaction timelines, support counts cannot establish ticket descriptions, and confirmed/pending risk counts do not establish a fraud accusation. Missing values remain null and carry limitations.

Source provenance records dataset/schema version, CSV checksum, row number and source field. Approved calculated metrics additionally record the exact source fields and calculation rule. See the aggregate quality report produced by `python -m scripts.profile_data`; profiling is separate from full-source embedding and never prints customer records.
