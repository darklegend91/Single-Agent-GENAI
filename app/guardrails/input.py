"""Heuristics reduce obvious misuse, but are not an authorization boundary.

Prompt injection is ultimately contained by independently authorized tools,
prose-free claims, exact source verification, and fixed output rendering.
"""
import re
from app.guardrails.policy import PolicyDecision

INJECTION = re.compile(r'ignore\s+(?:(?:all|the|previous|prior|above)\s+)*(?:instructions|rules)|system\s*prompt|developer\s*(?:prompt|message)|api[ _-]*keys?|\bsecrets?\b|\bpasswords?\b|override\s+(?:security|instructions)|pretend\s+(?:the\s+)?(?:csv|source|data)', re.I)
CROSS_CUSTOMER = re.compile(r'\b(?:all|every|other)\s+(?:customer|user)(?:s| record)?\b', re.I)
INFERENCE = re.compile(r'\b(?:religion|caste|ethnicity|sexual orientation|fraudster|fraudulent|criminal)\b|\b(?:is|committed|prove)\b.{0,35}\bfraud\b', re.I)
OUTSIDE_DOMAIN = re.compile(r'\b(?:weather|stock price|write (?:a |an )?(?:poem|essay|program)|capital of|sports score)\b', re.I)
UNAVAILABLE = re.compile(r'\b(?:why|meaning|explain)\b.{0,60}\bcancel|\bcancel.{0,40}\b(?:why|meaning)|ticket\s+(?:description|details|contents)|event\s+timeline|\bevery\s+(?:payment|event|ticket)\b', re.I)

def check_input(question: str) -> PolicyDecision:
    if INJECTION.search(question) or INFERENCE.search(question) or CROSS_CUSTOMER.search(question):
        return PolicyDecision(False)
    if UNAVAILABLE.search(question) or OUTSIDE_DOMAIN.search(question):
        return PolicyDecision(False, 'insufficient_evidence', 'unsupported_question')
    return PolicyDecision(True)
