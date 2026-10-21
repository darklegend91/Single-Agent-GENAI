"""The explicit local demo works without provider credentials or customer CSV."""

from fastapi.testclient import TestClient

from scripts.demo_support import create_demo_runtime


def test_local_demo_has_chat_and_customer_isolation():
    with create_demo_runtime() as runtime:
        from app.main import create_app
        with TestClient(create_app(runtime.settings, runtime.services)) as client:
            assert client.get("/").status_code == 200
            assert client.get("/health").json()["mode"] == "demo"
            headers = {"Authorization": f"Bearer {runtime.token}"}
            answer = client.post("/chat", headers=headers, json={"user_id": "U0000000001", "question": "Summarize subscription and payments."})
            assert answer.status_code == 200
            assert answer.json()["status"] == "answered"
            assert answer.json()["evidence"]
            denied = client.post("/chat", headers=headers, json={"user_id": "U0000000002", "question": "Summarize payments."})
            assert denied.status_code == 403


def test_local_demo_revenue_is_exact():
    with create_demo_runtime() as runtime:
        summary = runtime.services.analytics.revenue(runtime.principal, "U0000000001")
        assert summary.complete is True
        assert summary.total_paise == 120000
