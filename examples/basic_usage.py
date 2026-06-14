"""
Basic Usage Example for Oprel SDK.

This script demonstrates how to initialize the Oprel SDK client,
download a model if necessary, and run a simple generation query.
"""

from oprel.client import OprelClient

def main():
    # Initialize the client connecting to the local daemon
    client = OprelClient(base_url="http://localhost:8000")

    model_name = "llama3-8b"
    
    print(f"Ensuring model {model_name} is available...")
    client.pull(model_name)
    
    print("Generating response...")
    response = client.generate(
        model=model_name,
        prompt="Explain quantum computing in one sentence."
    )
    print("Response:", response)

if __name__ == "__main__":
    main()
