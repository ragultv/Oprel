"""
Quick test for embedding support
"""
import sys

def test_embed_api_import():
    """Test that embed function can be imported"""
    try:
        from oprel import embed
        print("✓ embed function imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import embed: {e}")
        return False


def test_client_embed_method():
    """Test that Client has embed method"""
    try:
        from oprel import Client
        client = Client()
        assert hasattr(client, 'embed'), "Client missing embed() method"
        print("✓ Client.embed() method exists")
        return True
    except Exception as e:
        print(f"✗ Client.embed() check failed: {e}")
        return False


def test_embedding_model_type():
    """Test that embedding models are marked as supported"""
    try:
        from oprel.models.model_types import detect_model_type, is_supported_model_type
        
        # Test detection
        model_type = detect_model_type("nomic-embed-text")
        assert model_type == "embeddings", f"Expected 'embeddings', got '{model_type}'"
        
        # Test supported
        is_supported = is_supported_model_type("embeddings")
        assert is_supported, "Embeddings should be supported"
        
        print("✓ Embedding model type is supported")
        return True
    except Exception as e:
        print(f"✗ Model type check failed: {e}")
        return False


def test_backend_embedding_mode():
    """Test that backend can detect embedding models"""
    try:
        from pathlib import Path
        from oprel.runtime.backends.llama_cpp import LlamaCppBackend
        from oprel.core.config import Config
        
        # Simulate an embedding model path
        class MockPath:
            def __init__(self, name):
                self._name = name
            
            @property
            def name(self):
                return self._name
            
            def __str__(self):
                return f"/fake/path/{self._name}"
        
        # Create a mock backend
        config = Config()
        backend = LlamaCppBackend(
            model_path=MockPath("nomic-embed-text-v1.5.Q4_K_M.gguf"),
            binary_path=Path("/fake/llama-server"),
            config=config
        )
        
        # Build command and check for --embedding flag
        cmd = backend.build_command(port=11434)
        cmd_str = " ".join(cmd)
        
        assert "--embedding" in cmd_str, "Backend should add --embedding flag for embedding models"
        print("✓ Backend detects embedding models and adds --embedding flag")
        return True
    except Exception as e:
        print(f"✗ Backend check failed: {e}")
        return False


def test_cli_command_exists():
    """Test that embed CLI command is registered"""
    try:
        import argparse
        from oprel.cli.main import main
        
        # Check if embed_command module exists
        from oprel.cli import embed_command
        assert hasattr(embed_command, 'cmd_embed'), "embed_command missing cmd_embed function"
        
        print("✓ CLI embed command registered")
        return True
    except Exception as e:
        print(f"✗ CLI command check failed: {e}")
        return False


def main():
    """Run all tests"""
    print("Testing Embedding Support Implementation\n" + "="*50)
    
    tests = [
        test_embed_api_import,
        test_client_embed_method,
        test_embedding_model_type,
        test_backend_embedding_mode,
        test_cli_command_exists,
    ]
    
    results = []
    for test in tests:
        print(f"\nRunning: {test.__name__}")
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            results.append(False)
    
    print("\n" + "="*50)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Embedding support is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Review implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
