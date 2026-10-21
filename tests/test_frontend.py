import httpx
import pytest

from frontend.client import BackendClient, BackendError


def test_frontend_posts_only_question_and_scope():
    captured = []
    def handle(request):
        captured.append(request)
        return httpx.Response(200, json={"status": "answered", "answer": "Source value.", "evidence": [], "limitations": []})
    client = BackendClient("http://127.0.0.1:8000", "local-token", transport=httpx.MockTransport(handle))
    assert client.chat("U0000000001", "Show payments.")["status"] == "answered"
    assert captured[0].url.path == "/chat"
    assert captured[0].headers["authorization"] == "Bearer local-token"
    assert b'"user_id":"U0000000001"' in captured[0].content


def test_frontend_errors_do_not_echo_provider_payload():
    client = BackendClient("http://127.0.0.1:8000", "token", transport=httpx.MockTransport(lambda request: httpx.Response(500, text="RAW SECRET")))
    with pytest.raises(BackendError) as error:
        client.chat("U0000000001", "Show payments.")
    assert "RAW SECRET" not in str(error.value)


def test_frontend_langfuse_status_uses_authenticated_route():
    captured = []
    def handle(request):
        captured.append(request)
        return httpx.Response(200, json={"status": "disabled", "message": "keys are missing"})
    client = BackendClient("http://127.0.0.1:8000", "local-token", transport=httpx.MockTransport(handle))
    assert client.langfuse_status()["status"] == "disabled"
    assert captured[0].url.path == "/observability/langfuse"
    assert captured[0].headers["authorization"] == "Bearer local-token"


def test_streamlit_page_runs_without_backend(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("KA_API_URL", "http://127.0.0.1:1")
    from pathlib import Path
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "streamlit_app.py").run(timeout=10)
    assert not app.exception
    assert app.title[0].value == "Customer Knowledge Assistant"
    assert app.chat_input
    assert all("Bearer token" not in item.label for item in app.text_input)


def test_streamlit_always_displays_selected_customer_id(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from pathlib import Path
    monkeypatch.setenv("KA_API_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("KA_FRONTEND_CUSTOMERS", '["U0000000010"]')
    monkeypatch.setenv("KA_FRONTEND_CUSTOMER", "U0000000010")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "streamlit_app.py").run(timeout=10)

    assert not app.exception
    assert any("Selected customer: U0000000010" in item.value for item in app.caption)
