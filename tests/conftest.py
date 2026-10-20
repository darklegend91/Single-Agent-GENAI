"""Synthetic fixtures only; never load the user's customer dataset."""

import pytest


@pytest.fixture
def credential_mapping():
    return {
        "synthetic-customer-token": {
            "subject": "analyst",
            "customer_ids": ["U0000000001"],
            "ingest_admin": False,
        },
        "synthetic-admin-token": {
            "subject": "operator",
            "customer_ids": [],
            "ingest_admin": True,
        },
    }
