"""
Basic unit tests for Oprel SDK
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from oprel.core.config import Config
from oprel.core.exceptions import OprelError, ModelNotFoundError, MemoryError
from oprel.telemetry.hardware import get_hardware_info
from oprel.telemetry.recommender import recommend_quantization
from oprel.downloader.cache import get_cache_path, sanitize_filename
from oprel.utils.paths import expand_path, ensure_dir, get_file_size_mb
from oprel.utils.platform import get_platform, is_64bit


class TestConfig:
    """Test configuration management"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = Config()
        
        # default_max_memory_mb should be dynamically calculated (minimum 8GB)
        assert config.default_max_memory_mb >= 8192
        assert config.use_unix_socket is True
        assert config.log_level == "INFO"
        assert config.binary_version == "b3901"
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = Config(
            default_max_memory_mb=16384,
            log_level="DEBUG",
        )
        
        assert config.default_max_memory_mb == 16384
        assert config.log_level == "DEBUG"
    
    def test_ensure_dirs(self, tmp_path):
        """Test directory creation"""
        config = Config(
            cache_dir=tmp_path / "cache",
            binary_dir=tmp_path / "bin",
        )
        
        config.ensure_dirs()
        
        assert config.cache_dir.exists()
        assert config.binary_dir.exists()


class TestHardwareDetection:
    """Test hardware detection"""
    
    def test_get_hardware_info(self):
        """Test basic hardware info retrieval"""
        info = get_hardware_info()
        
        assert "os" in info
        assert "arch" in info
        assert "cpu_count" in info
        assert "ram_total_gb" in info
        assert info["ram_total_gb"] > 0
    
    def test_recommend_quantization(self):
        """Test quantization recommendation"""
        quant = recommend_quantization(model_size_b=7)
        
        # Should return one of the standard quantization levels
        valid_quants = ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q8_0"]
        assert quant in valid_quants


class TestCacheManagement:
    """Test cache utilities"""
    
    def test_get_cache_path(self):
        """Test cache path retrieval"""
        cache_path = get_cache_path()
        
        assert cache_path.is_absolute()
        assert "oprel" in str(cache_path)
        assert cache_path.exists()
    
    def test_sanitize_filename(self):
        """Test filename sanitization"""
        from oprel.downloader.cache import sanitize_filename
        
        assert sanitize_filename("my/model:v1.0") == "my_model_v1.0"
        assert sanitize_filename("test<>file") == "test__file"
        assert sanitize_filename("  .dots.  ") == "dots"


class TestPathUtilities:
    """Test path utilities"""
    
    def test_expand_path(self):
        """Test path expansion"""
        path = expand_path("~/test")
        assert path.is_absolute()
        assert "test" in str(path)
    
    def test_ensure_dir(self, tmp_path):
        """Test directory creation"""
        test_dir = tmp_path / "new" / "nested" / "dir"
        
        result = ensure_dir(test_dir, create=True)
        assert result.exists()
        assert result.is_dir()
    
    def test_ensure_dir_raises(self, tmp_path):
        """Test ensure_dir raises when create=False"""
        test_dir = tmp_path / "nonexistent"
        
        with pytest.raises(FileNotFoundError):
            ensure_dir(test_dir, create=False)
    
    def test_get_file_size_mb(self, tmp_path):
        """Test file size calculation"""
        test_file = tmp_path / "test.bin"
        test_file.write_bytes(b"0" * (1024 * 1024 * 5))  # 5MB
        
        size = get_file_size_mb(test_file)
        assert size == pytest.approx(5.0, abs=0.1)


class TestPlatformDetection:
    """Test platform utilities"""
    
    def test_get_platform(self):
        """Test platform detection"""
        platform = get_platform()
        assert platform in ["Darwin", "Linux", "Windows"]
    
    def test_is_64bit(self):
        """Test 64-bit detection"""
        assert is_64bit() is True  # Modern systems are 64-bit


class TestExceptions:
    """Test custom exceptions"""
    
    def test_oprel_error(self):
        """Test base exception"""
        with pytest.raises(OprelError):
            raise OprelError("Test error")
    
    def test_model_not_found(self):
        """Test ModelNotFoundError"""
        with pytest.raises(ModelNotFoundError):
            raise ModelNotFoundError("Model not found")
    
    def test_memory_error(self):
        """Test custom MemoryError"""
        with pytest.raises(MemoryError):
            raise MemoryError("Out of memory")


class TestDownloader:
    """Test model downloader (mocked)"""
    
    @patch("oprel.downloader.hub.hf_hub_download")
    @patch("oprel.downloader.hub.list_repo_files")
    def test_download_model(self, mock_list_files, mock_download, tmp_path):
        """Test model download with mocked HF Hub"""
        from oprel.downloader.hub import download_model
        
        # Mock available files
        mock_list_files.return_value = [
            "llama-2-7b.Q4_K_M.gguf",
            "llama-2-7b.Q8_0.gguf",
        ]
        
        # Mock download
        model_path = tmp_path / "llama-2-7b.Q4_K_M.gguf"
        model_path.touch()
        mock_download.return_value = str(model_path)
        
        # Test download
        result = download_model(
            "TheBloke/Llama-2-7B-GGUF",
            quantization="Q4_K_M",
            cache_dir=tmp_path,
        )
        
        assert result.exists()
        assert result.name == "llama-2-7b.Q4_K_M.gguf"
    
    @patch("oprel.downloader.hub.list_repo_files")
    def test_list_available_quantizations(self, mock_list_files):
        """Test listing quantizations"""
        from oprel.downloader.hub import list_available_quantizations
        
        mock_list_files.return_value = [
            "model.Q4_K_M.gguf",
            "model.Q5_K_M.gguf",
            "model.Q8_0.gguf",
            "README.md",
        ]
        
        quants = list_available_quantizations("test/model")
        
        assert "Q4_K_M" in quants
        assert "Q5_K_M" in quants
        assert "Q8_0" in quants
        assert len(quants) == 3


@pytest.fixture
def mock_config(tmp_path):
    """Fixture for test configuration"""
    return Config(
        cache_dir=tmp_path / "cache",
        binary_dir=tmp_path / "bin",
        default_max_memory_mb=4096,
    )