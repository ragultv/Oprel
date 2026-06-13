"""
SSL/TLS Configuration Examples for oprel

This file demonstrates how to handle SSL certificate verification issues
that may occur when downloading binaries in corporate environments.
"""

import os
from pathlib import Path
from oprel import Model
from oprel.core.config import Config


# Example 1: Quick fix - Disable SSL verification (NOT RECOMMENDED for production)
# Useful for testing or corporate environments with proxy/firewall issues
def example_disable_ssl_verification():
    """Disable SSL verification using environment variable"""
    print("Example 1: Disable SSL verification via environment variable")
    
    # Set environment variable before importing oprel
    os.environ["OPREL_SSL_NO_VERIFY"] = "1"
    
    # Now models will download without SSL verification
    model = Model("qwencoder")
    response = model.chat("Hello!")
    print(f"Response: {response}")


# Example 2: Disable SSL via Config (programmatic approach)
def example_programmatic_ssl_disable():
    """Disable SSL verification via Config object"""
    print("\nExample 2: Disable SSL verification via Config")
    
    config = Config(ssl_verify=False)
    model = Model("qwencoder", config=config)
    
    response = model.chat("Hello!")
    print(f"Response: {response}")


# Example 3: Use custom CA certificate bundle (RECOMMENDED for corporate environments)
def example_custom_ca_certificate():
    """Use custom CA certificate for corporate proxy"""
    print("\nExample 3: Use custom CA certificate")
    
    # Path to your corporate CA certificate bundle
    # This file should be provided by your IT department
    ca_cert_path = Path("/path/to/corporate-ca-bundle.crt")
    
    # Only proceed if the certificate file exists
    if ca_cert_path.exists():
        config = Config(ssl_cert_file=ca_cert_path)
        model = Model("qwencoder", config=config)
        
        response = model.chat("Hello!")
        print(f"Response: {response}")
    else:
        print(f"Certificate file not found: {ca_cert_path}")
        print("Please obtain the CA certificate from your IT department")


# Example 4: Complete configuration with SSL settings
def example_complete_config():
    """Complete configuration including SSL and other settings"""
    print("\nExample 4: Complete configuration")
    
    config = Config(
        # SSL settings
        ssl_verify=True,  # Enable SSL verification (default)
        ssl_cert_file=None,  # No custom certificate
        
        # Cache settings
        cache_dir=Path.home() / ".cache" / "oprel" / "models",
        binary_dir=Path.home() / ".cache" / "oprel" / "bin",
        
        # Performance settings
        n_gpu_layers=-1,  # Auto-detect GPU layers
        ctx_size=4096,
        kv_cache_type="f16",  # or "q8_0" for 50% memory savings
    )
    
    model = Model("qwencoder", config=config)
    response = model.chat("Hello!")
    print(f"Response: {response}")


# Example 5: Handling SSL errors gracefully
def example_error_handling():
    """Handle SSL errors with fallback"""
    print("\nExample 5: Error handling with fallback")
    
    try:
        # Try with SSL verification first
        config = Config(ssl_verify=True)
        model = Model("qwencoder", config=config)
        response = model.chat("Hello!")
        print(f"Response: {response}")
        
    except Exception as e:
        if "SSL" in str(e) or "CERTIFICATE" in str(e):
            print(f"SSL Error occurred: {e}")
            print("Falling back to disabled SSL verification...")
            
            # Fallback to disabled SSL
            config = Config(ssl_verify=False)
            model = Model("qwencoder", config=config)
            response = model.chat("Hello!")
            print(f"Response (with SSL disabled): {response}")
        else:
            raise


# Main function to demonstrate all examples
if __name__ == "__main__":
    print("=" * 60)
    print("SSL/TLS Configuration Examples for oprel")
    print("=" * 60)
    
    # Uncomment the example you want to run:
    
    # example_disable_ssl_verification()
    # example_programmatic_ssl_disable()
    # example_custom_ca_certificate()
    # example_complete_config()
    # example_error_handling()
    
    print("\n" + "=" * 60)
    print("Choose an example to run by uncommenting it in the code")
    print("=" * 60)
    
    print("""
    For production use, prefer these options in order:
    1. Update system certificates (best)
    2. Use custom CA certificate (good for corporate)
    3. Disable SSL verification (only as last resort)
    
    For more information, see docs/troubleshooting.md
    """)
