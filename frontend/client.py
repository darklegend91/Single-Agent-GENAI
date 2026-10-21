"""Small authenticated FastAPI client shared by the Streamlit views."""
import httpx


class BackendError(Exception):
    pass


class BackendClient:
    def __init__(self, base_url, token, *, transport=None, timeout=120):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.transport = transport
        self.timeout = timeout

    def request(self, method, path, **kwargs):
        try:
            with httpx.Client(base_url=self.base_url, timeout=self.timeout, transport=self.transport, trust_env=False) as client:
                response = client.request(method, path, headers={"Authorization": f"Bearer {self.token}"}, **kwargs)
        except httpx.HTTPError:
            raise BackendError("Cannot reach FastAPI. Check that the backend is running.") from None
        if response.is_error:
            messages = {401: "Enter a valid bearer token.", 403: "This token cannot access that customer.", 404: "Customer or resource not found.", 422: "Check the customer ID and question.", 429: "Request limit reached. Try again shortly.", 503: "A backend dependency is unavailable. Check Ollama and the knowledge index.", 504: "The local model exceeded the request deadline. Try a shorter question."}
            try:
                payload = response.json()
                detail = payload.get("error", {}).get("message")
            except (ValueError, AttributeError):
                detail = None
            raise BackendError(detail or messages.get(response.status_code, "The backend could not complete this request."))
        try:
            return response.json()
        except ValueError:
            raise BackendError("The backend returned an invalid response.") from None

    def chat(self, user_id, question):
        return self.request("POST", "/chat", json={"user_id": user_id, "question": question})

    def summary(self, user_id):
        return self.request("GET", f"/customers/{user_id}/summary")

    def revenue(self, user_id):
        return self.request("GET", f"/customers/{user_id}/revenue")

    def health(self):
        return self.request("GET", "/health")

    def langfuse_status(self):
        return self.request("GET", "/observability/langfuse")
