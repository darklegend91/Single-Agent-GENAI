import pytest

from frontend.customer_scope import resolve_customer_id


CUSTOMERS = ["U0000000001", "U0000000010"]


def test_question_customer_id_overrides_sidebar_default():
    assert resolve_customer_id("Summarize payments for U0000000010", CUSTOMERS[0], CUSTOMERS) == CUSTOMERS[1]


def test_question_without_id_uses_sidebar_default():
    assert resolve_customer_id("Summarize subscription and payments", CUSTOMERS[1], CUSTOMERS) == CUSTOMERS[1]


def test_question_cannot_target_multiple_customers():
    with pytest.raises(ValueError, match="one customer at a time"):
        resolve_customer_id("Compare U0000000001 and U0000000010", CUSTOMERS[0], CUSTOMERS)


def test_local_sample_rejects_customer_outside_authorized_scope():
    with pytest.raises(ValueError, match="not included in the indexed sample"):
        resolve_customer_id("Tell me about U0000000002", CUSTOMERS[0], CUSTOMERS)


def test_full_dataset_mode_accepts_any_well_formed_customer_id():
    assert resolve_customer_id("Tell me about U0000000045", CUSTOMERS[0]) == "U0000000045"
