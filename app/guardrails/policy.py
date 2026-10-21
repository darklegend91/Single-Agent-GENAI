from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    status: Literal['refused', 'insufficient_evidence'] = 'refused'
    reason: str = 'restricted_request'

LIMITATIONS = {
    'missing_field': 'One or more required source values are blank or absent. Missing values are not treated as zero.',
    'incomplete_period': 'Some monthly source values are missing, so the total covers available months only; it is not a full-period total.',
    'unknown_code': 'Source codes are reported literally; their business meanings are undocumented.',
    'unresolved_timestamp': 'A source timestamp has no configured encoding and is reported as recorded.',
    'aggregate_only': 'The source contains summary counts or totals, not individual events, so it cannot provide transaction, ticket, or activity timelines.',
    'no_relevant_evidence': 'The selected customer’s indexed record has no fields relevant to this question. Check the selected customer ID and Customer details, or ask about a recorded field or aggregate.',
    'unsupported_question': 'The dataset does not record the requested detail or support the requested conclusion.',
    'unverified_data_consistency': 'Related source aggregates have not been established as internally consistent.',
    'restricted_request': 'The request was blocked by a safety or access policy; customer data cannot be shared across customer scopes.',
    'customer_scope_mismatch': 'The customer ID in the question differs from the selected customer. Select that customer in the sidebar if it is in your authorized sample.',
}
