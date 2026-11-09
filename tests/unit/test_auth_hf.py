"""Unit tests for live Hugging Face token verification and expired status handling."""

import pytest
from unittest.mock import patch, MagicMock
from oprel.server.routes.auth import get_hf_status


@pytest.mark.asyncio
async def test_hf_status_missing():
    with patch("oprel.server.db.get_hf_auth", return_value=None), \
         patch.dict("os.environ", {}, clear=True):
        status = await get_hf_status()
        assert status["connected"] is False
        assert status["status"] == "missing"


@pytest.mark.asyncio
async def test_hf_status_valid():
    mock_auth = {
        "token": "hf_valid_token_12345",
        "username": "testuser",
        "display_name": "Test User",
        "avatar_url": "https://example.com/avatar.png",
        "email": "test@example.com",
        "connected_at": "2026-01-01T00:00:00",
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "name": "testuser",
        "fullname": "Test User",
        "avatarUrl": "https://example.com/avatar.png",
        "email": "test@example.com",
    }

    with patch("oprel.server.db.get_hf_auth", return_value=mock_auth), \
         patch("requests.get", return_value=mock_resp):
        status = await get_hf_status()
        assert status["connected"] is True
        assert status["status"] == "valid"
        assert status["username"] == "testuser"


@pytest.mark.asyncio
async def test_hf_status_expired():
    mock_auth = {
        "token": "hf_expired_token_12345",
        "username": "expireduser",
        "display_name": "Expired User",
        "avatar_url": None,
        "email": None,
        "connected_at": "2025-01-01T00:00:00",
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 401

    with patch("oprel.server.db.get_hf_auth", return_value=mock_auth), \
         patch("requests.get", return_value=mock_resp):
        status = await get_hf_status()
        assert status["connected"] is False
        assert status["status"] == "expired"
        assert "expired" in status["message"].lower()


@pytest.mark.asyncio
async def test_hf_status_offline_fallback():
    mock_auth = {
        "token": "hf_offline_token_12345",
        "username": "offlineuser",
        "display_name": "Offline User",
        "avatar_url": None,
        "email": None,
        "connected_at": "2026-01-01T00:00:00",
    }

    with patch("oprel.server.db.get_hf_auth", return_value=mock_auth), \
         patch("requests.get", side_effect=Exception("Connection timed out")):
        status = await get_hf_status()
        # Offline mode should fall back gracefully to valid so offline inference works
        assert status["connected"] is True
        assert status["status"] == "valid"

