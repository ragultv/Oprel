"""
Unit tests for Hugging Face authentication, DB persistence, and downloader integration.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from oprel.core.config import Config
from oprel.downloader.hub import get_active_hf_token
from oprel.server import db
from oprel.server.app import app


@pytest.fixture
def temp_db(tmp_path: Path):
    """Fixture ensuring a clean temporary database for tests."""
    original_db = db.DB_PATH
    test_db = tmp_path / "test_chat.db"
    db.DB_PATH = test_db
    db.init_db()
    yield test_db
    db.DB_PATH = original_db


def test_hf_auth_db_crud(temp_db):
    """Verify storing, retrieving, and clearing Hugging Face auth in SQLite."""
    # Initially not connected
    assert db.get_hf_auth() is None

    # Save credentials
    saved = db.save_hf_auth(
        token="hf_test_secret_token_12345",
        username="oprel_tester",
        display_name="Oprel Tester",
        avatar_url="https://huggingface.co/avatar.png",
        email="test@oprel.ai",
    )
    assert saved["username"] == "oprel_tester"
    assert saved["token"] == "hf_test_secret_token_12345"

    # Fetch stored
    fetched = db.get_hf_auth()
    assert fetched is not None
    assert fetched["username"] == "oprel_tester"
    assert fetched["display_name"] == "Oprel Tester"
    assert fetched["token"] == "hf_test_secret_token_12345"

    # Delete credentials
    assert db.delete_hf_auth() is True
    assert db.get_hf_auth() is None


def test_get_active_hf_token_resolution(temp_db, monkeypatch):
    """Verify resolution priority: ENV -> DB -> CONFIG -> None."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    db.delete_hf_auth()

    # 1. When empty -> None
    assert get_active_hf_token() is None

    # 2. When stored in DB -> returns DB token
    db.save_hf_auth("hf_db_token_111", "db_user")
    assert get_active_hf_token() == "hf_db_token_111"

    # 3. When ENV is set -> ENV takes precedence over DB
    monkeypatch.setenv("HF_TOKEN", "hf_env_token_222")
    assert get_active_hf_token() == "hf_env_token_222"


def test_auth_routes_status_flow(temp_db, monkeypatch):
    """Verify GET /v1/auth/hf/status when disconnected vs connected."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    client = TestClient(app)

    # Disconnected
    db.delete_hf_auth()
    res = client.get("/v1/auth/hf/status")
    assert res.status_code == 200
    data = res.json()
    assert data["connected"] is False
    assert data["username"] is None

    # Connected
    db.save_hf_auth(
        token="hf_secret_1234567890",
        username="john_doe",
        display_name="John Doe",
        avatar_url="https://huggingface.co/avatar.jpg",
    )
    res = client.get("/v1/auth/hf/status")
    assert res.status_code == 200
    data = res.json()
    assert data["connected"] is True
    assert data["username"] == "john_doe"
    assert data["masked_token"].startswith("hf_s...")
    assert data["masked_token"].endswith("7890")


def test_connect_hf_token_success(temp_db, monkeypatch):
    """Verify POST /v1/auth/hf/token with valid token."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    client = TestClient(app)

    mock_hf_response = MagicMock()
    mock_hf_response.status_code = 200
    mock_hf_response.json.return_value = {
        "name": "hugging_fan",
        "fullname": "Hugging Fanatic",
        "avatarUrl": "https://huggingface.co/avatar.png",
        "email": "fan@hf.co",
    }

    with patch("requests.get", return_value=mock_hf_response):
        res = client.post("/v1/auth/hf/token", json={"token": "hf_valid_token_abc"})
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["connected"] is True
        assert data["username"] == "hugging_fan"

        # Verify DB persisted
        stored = db.get_hf_auth()
        assert stored is not None
        assert stored["token"] == "hf_valid_token_abc"
        assert stored["username"] == "hugging_fan"


def test_connect_hf_token_invalid_rejected(temp_db, monkeypatch):
    """Verify POST /v1/auth/hf/token with invalid token returns 400."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    client = TestClient(app)

    mock_hf_response = MagicMock()
    mock_hf_response.status_code = 401

    with patch("requests.get", return_value=mock_hf_response):
        res = client.post("/v1/auth/hf/token", json={"token": "hf_invalid_token"})
        assert res.status_code == 400
        assert "Invalid Hugging Face token" in res.json()["detail"]


def test_disconnect_hf_token(temp_db, monkeypatch):
    """Verify DELETE /v1/auth/hf/token clears DB and environment."""
    monkeypatch.setenv("HF_TOKEN", "hf_existing")
    db.save_hf_auth("hf_existing", "alice")

    client = TestClient(app)
    res = client.delete("/v1/auth/hf/token")
    assert res.status_code == 200
    assert res.json()["connected"] is False

    assert db.get_hf_auth() is None
    assert "HF_TOKEN" not in os.environ
