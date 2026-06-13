"""
Example: Using Oprel Server with OpenAI SDK

This demonstrates how to use the Oprel server with the official OpenAI Python SDK.
The server is 100% compatible with OpenAI's API format.
"""

from openai import OpenAI

# Configure client to use local Oprel server
client = OpenAI(
    base_url="http://localhost:11435/v1",
    api_key="not-needed"  # Oprel doesn't require authentication
)

print("🚀 Oprel Server Example - OpenAI SDK\n")
print("Make sure you've started the server with: oprel serve\n")

# Example 1: Simple chat completion (non-streaming)
print("=" * 60)
print("Example 1: Simple Chat Completion (Non-Streaming)")
print("=" * 60)

response = client.chat.completions.create(
    model="qwen2.5-1.5b",  # Use a small model for quick testing
    messages=[
        {"role": "user", "content": "Write a haiku about programming"}
    ],
    max_tokens=100,
    temperature=0.7
)

print(f"\nResponse: {response.choices[0].message.content}")
print(f"Tokens used: {response.usage.total_tokens}")

# Example 2: Streaming chat completion
print("\n" + "=" * 60)
print("Example 2: Streaming Chat Completion")
print("=" * 60)

print("\nStreaming response: ", end="", flush=True)
stream = client.chat.completions.create(
    model="qwen2.5-1.5b",
    messages=[
        {"role": "user", "content": "Count from 1 to 5"}
    ],
    max_tokens=50,
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()

# Example 3: Multi-turn conversation
print("\n" + "=" * 60)
print("Example 3: Multi-Turn Conversation")
print("=" * 60)

conversation = [
    {"role": "user", "content": "What is 2+2?"},
]

response1 = client.chat.completions.create(
    model="qwen2.5-1.5b",
    messages=conversation,
    max_tokens=50
)

print(f"\nUser: {conversation[0]['content']}")
print(f"Assistant: {response1.choices[0].message.content}")

# Add to conversation
conversation.append({
    "role": "assistant",
    "content": response1.choices[0].message.content
})
conversation.append({
    "role": "user",
    "content": "What is that number multiplied by 3?"
})

response2 = client.chat.completions.create(
    model="qwen2.5-1.5b",
    messages=conversation,
    max_tokens=50
)

print(f"\nUser: {conversation[2]['content']}")
print(f"Assistant: {response2.choices[0].message.content}")

# Example 4: List available models
print("\n" + "=" * 60)
print("Example 4: List Available Models")
print("=" * 60)

models = client.models.list()
print(f"\nAvailable models: {len(models.data)}")
for model in models.data[:5]:  # Show first 5
    print(f"  - {model.id}")
print("  ...")

print("\n✅ All examples completed successfully!")
print("\nTips:")
print("  - The server keeps models loaded for 15 minutes")
print("  - Repeated requests to the same model are instant")
print("  - Switching models automatically unloads the old one")
print("  - Check health: curl http://localhost:11434/health")
