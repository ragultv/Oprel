"""
Pytest configuration and shared fixtures
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock
import subprocess
import time

from oprel.core.config import Config


@pytest.fixture
def temp_cache(tmp_path):
    """Temporary cache directory for testing"""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def temp_binary_dir(tmp_path):
    """Temporary binary directory for testing"""
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    return binary_dir


@pytest.fixture
def test_config(temp_cache, temp_binary_dir):
    """Test configuration with temporary directories"""
    return Config(
        cache_dir=temp_cache,
        binary_dir=temp_binary_dir,
        default_max_memory_mb=4096,
        log_level="DEBUG",
    )


@pytest.fixture
def mock_model_file(temp_cache):
    """Create a mock model file for testing"""
    model_dir = temp_cache / "models--test--model" / "snapshots" / "abc123"
    model_dir.mkdir(parents=True)
    model_file = model_dir / "test-model.Q4_K_M.gguf"
    model_file.write_bytes(b"0" * (1024 * 1024 * 100))  # 100MB fake model
    return model_file


@pytest.fixture
def mock_subprocess():
    """Mock subprocess.Popen for testing"""
    mock = Mock()
    mock.poll.return_value = None  # Process is running
    mock.pid = 12345
    mock.returncode = None
    mock.stdout = Mock()
    mock.stderr = Mock()
    mock.stdout.read = Mock(return_value="")
    mock.stderr.read = Mock(return_value="")
    return mock


@pytest.fixture
def mock_requests_get():
    """Mock requests.get for testing HTTP client"""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"status": "ok"}
    mock_response.text = '{"status": "ok"}'
    return mock_response


@pytest.fixture
def mock_requests_post():
    """Mock requests.post for testing HTTP client"""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "success": True,
        "message": "Model loaded",
        "text": "Test response"
    }
    return mock_response


@pytest.fixture
def mock_gpu_info():
    """Mock GPU information"""
    return {
        'gpu_type': 'nvidia',
        'gpu_name': 'NVIDIA GeForce GTX 1650',
        'vram_total_gb': 4.0,
        'vram_available_gb': 3.5,
        'cuda_version': '12.0',
    }


@pytest.fixture
def mock_hardware_info():
    """Mock complete hardware information"""
    return {
        'os': 'Windows',
        'arch': 'x86_64',
        'cpu_count': 8,
        'cpu_threads': 16,
        'ram_total_gb': 16.0,
        'ram_available_gb': 8.0,
        'gpu_type': 'nvidia',
        'gpu_name': 'NVIDIA GeForce GTX 1650',
        'vram_total_gb': 4.0,
    }


@pytest.fixture
def sample_chat_messages():
    """Sample chat messages for testing"""
    return [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Hello!'},
        {'role': 'assistant', 'content': 'Hi there! How can I help you?'},
        {'role': 'user', 'content': 'What is Python?'},
    ]

    mock.pid = 12345
    mock.poll.return_value = None  # Process is running
    mock.returncode = 0
    return mock


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration between tests"""
    import logging
    
    # Remove all handlers from oprel loggers
    for name in list(logging.Logger.manager.loggerDict.keys()):
        if name.startswith('oprel'):
            logger = logging.getLogger(name)
            logger.handlers.clear()
            logger.setLevel(logging.NOTSET)
    
    yield
    
    # Cleanup after test
    for name in list(logging.Logger.manager.loggerDict.keys()):
        if name.startswith('oprel'):
            logger = logging.getLogger(name)
            logger.handlers.clear()


@pytest.fixture
def mock_hardware_info():
    """Mock hardware information"""
    return {
        "os": "Linux",
        "arch": "x86_64",
        "cpu_count": 8,
        "cpu_threads": 16,
        "ram_total_gb": 32.0,
        "ram_available_gb": 16.0,
        "gpu_type": "cuda",
        "gpu_name": "NVIDIA RTX 3090",
        "vram_total_gb": 24.0,
    }


@pytest.fixture
def mock_cpu_only_hardware():
    """Mock hardware info for CPU-only system"""
    return {
        "os": "Darwin",
        "arch": "x86_64",
        "cpu_count": 4,
        "cpu_threads": 8,
        "ram_total_gb": 16.0,
        "ram_available_gb": 8.0,
    }

@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch):
    """Every test gets an in-memory keyring so the real OS credential store is never read or changed."""
    from types import SimpleNamespace

    from oprel.utils import secrets_store

    vault = {}
    fake = SimpleNamespace(
        set_password=lambda service, name, value: vault.__setitem__((service, name), value),
        get_password=lambda service, name: vault.get((service, name)),
        delete_password=lambda service, name: vault.pop((service, name), None),
    )
    monkeypatch.setattr(secrets_store, "_keyring", lambda: fake)
    yield vault
