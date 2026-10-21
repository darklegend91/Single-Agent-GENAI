"""Resolve the per-question customer target without expanding server permissions."""
import re


CUSTOMER_ID_PATTERN = re.compile(r"\bU[0-9]{10}\b")


def resolve_customer_id(question, default_user_id, allowed_customers=()):
    """Use an explicit single customer ID, otherwise the selected default."""
    mentioned = list(dict.fromkeys(CUSTOMER_ID_PATTERN.findall(question)))
    if len(mentioned) > 1:
        raise ValueError("Ask about one customer at a time. Include a single customer ID in your question.")
    user_id = mentioned[0] if mentioned else default_user_id
    if allowed_customers and user_id not in allowed_customers:
        raise ValueError("That customer is not included in the indexed sample. Re-run without --limit to index the full dataset.")
    return user_id
