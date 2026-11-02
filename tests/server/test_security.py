from fastapi import FastAPI
from fastapi.testclient import TestClient

from oprel.server.security import LocalGuardMiddleware
from oprel.utils.auth import get_api_token


def _client():
    app = FastAPI()
    app.add_middleware(LocalGuardMiddleware)

    @app.get("/models")
    def models():
        return {"ok": True}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return TestClient(app, base_url="http://127.0.0.1:11435")


def test_no_token_rejected():
    assert _client().get("/models").status_code == 401


def test_valid_token_allowed():
    r = _client().get("/models", headers={"Authorization": f"Bearer {get_api_token()}"})
    assert r.status_code == 200


def test_wrong_token_rejected():
    assert _client().get("/models", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_foreign_origin_blocked_even_with_token():
    r = _client().get(
        "/models",
        headers={"Origin": "https://evil.example", "Authorization": f"Bearer {get_api_token()}"},
    )
    assert r.status_code == 403


def test_cross_site_fetch_blocked():
    assert _client().get("/models", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403


def test_dns_rebinding_host_blocked():
    c = TestClient(_client().app, base_url="http://evil.example")
    assert c.get("/models", headers={"Authorization": f"Bearer {get_api_token()}"}).status_code == 403


def test_local_ui_origin_allowed():
    assert _client().get("/models", headers={"Origin": "http://localhost:3000"}).status_code == 200
    assert _client().get("/models", headers={"Sec-Fetch-Site": "same-origin"}).status_code == 200


def test_health_open():
    assert _client().get("/health").status_code == 200
