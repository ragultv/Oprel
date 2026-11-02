import pytest


@pytest.fixture
def tool_env(tmp_path, monkeypatch):
    """
    Isolated environment for integration/tool tests:
    in-memory secret store, temp SQLite DB, temp daemon token file.
    Never touches the real OS keychain or ~/.cache/oprel.
    """
    monkeypatch.setenv("OPREL_SECRET_BACKEND", "memory")
    # Per-install bearer token used by LocalGuardMiddleware; fixed so ~/.oprel isn't touched.
    monkeypatch.setenv("OPREL_API_TOKEN", "test-install-token-0123456789")
    monkeypatch.delenv("OPREL_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("OPREL_ALLOW_REMOTE_INTEGRATIONS", raising=False)

    from oprel.server import db
    from oprel.server.security import local_auth, secrets

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()

    secrets.reset_secret_store_for_tests()
    monkeypatch.setattr(local_auth, "_token_file", lambda: tmp_path / "daemon.token")
    monkeypatch.setattr(local_auth, "_token", None)
    local_auth.set_bind_host("127.0.0.1")

    yield tmp_path

    secrets.reset_secret_store_for_tests()


@pytest.fixture
def client(tool_env):
    from fastapi.testclient import TestClient

    from oprel.server.app import app

    # No `with`: skip the lifespan (backend cleanup, idle monitor) for unit tests.
    # Behave like the bundled web UI: same-origin browser requests pass LocalGuardMiddleware.
    return TestClient(app, base_url="http://127.0.0.1:11435", headers={"Sec-Fetch-Site": "same-origin"})


@pytest.fixture
def token(client):
    return client.get("/auth/session").json()["token"]
