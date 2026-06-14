"""
Chat Templates Example for Oprel SDK.

This script shows how to conduct a multi-turn chat using the `chat` API,
which automatically handles applying the correct chat template (e.g., ChatML, Llama-3 format).
"""

from oprel.client import OprelClient

def main():
    client = OprelClient()
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, what is the capital of France?"}
    ]
    
    response = client.chat(model="llama3-8b", messages=messages)
    print("Assistant:", response['message']['content'])
    
    # Append the response and continue the conversation
    messages.append(response['message'])
    messages.append({"role": "user", "content": "What is its population?"})
    
    response2 = client.chat(model="llama3-8b", messages=messages)
    print("Assistant:", response2['message']['content'])

if __name__ == "__main__":
    main()
