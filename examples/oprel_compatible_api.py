"""
Example: Oprel-compatible API usage

This example demonstrates how to use Oprel's compatible API.
"""

from oprel import chat, generate, list as list_models, Client, ChatResponse

# Example 1: Simple chat using module-level function
print("Example 1: Simple Chat")
print("-" * 50)

response: ChatResponse = chat(
    model='qwencoder',
    messages=[
        {'role': 'user', 'content': 'What is Python?'}
    ]
)

print(response.message.content)
# or access directly: print(response['message']['content'])

print("\n")

# Example 2: Streaming chat
print("Example 2: Streaming Chat")
print("-" * 50)

stream = chat(
    model='qwencoder',
    messages=[{'role': 'user', 'content': 'Write a haiku about coding'}],
    stream=True
)

for chunk in stream:
    print(chunk.message.content, end='', flush=True)

print("\n\n")

# Example 3: Using Client class
print("Example 3: Using Client Class")
print("-" * 50)

client = Client(host='http://localhost:11434')

response = client.chat(
    model='qwencoder',
    messages=[
        {'role': 'system', 'content': 'You are a Python expert.'},
        {'role': 'user', 'content': 'Explain decorators briefly.'}
    ]
)

print(response.message.content)

print("\n")

# Example 4: Generate (single prompt, no chat history)
print("Example 4: Simple Generation")
print("-" * 50)

response = generate(
    model='qwencoder',
    prompt='Why is the sky blue?'
)

print(response.response)

print("\n")

# Example 5: List available models
print("Example 5: List Models")
print("-" * 50)

models = list_models()

print(f"Found {len(models.models)} models:")
for model in models.models[:5]:  # Show first 5
    print(f"  - {model.name}")

print("\n")

# Example 6: Pydantic model validation (JSON mode)
print("Example 6: Structured Output with Pydantic")
print("-" * 50)

from pydantic import BaseModel

class Country(BaseModel):
    name: str
    capital: str
    languages: list[str]

response = chat(
    model='qwencoder',
    messages=[{'role': 'user', 'content': 'Tell me about France in JSON format'}],
    format=Country.model_json_schema(),
)

# Parse and validate response
try:
    country = Country.model_validate_json(response.message.content)
    print(f"Country: {country.name}")
    print(f"Capital: {country.capital}")
    print(f"Languages: {', '.join(country.languages)}")
except Exception as e:
    print(f"Note: Structured output requires model cooperation: {e}")

print("\n")

# Example 7: Custom options
print("Example 7: Custom Generation Options")
print("-" * 50)

response = client.generate(
    model='qwencoder',
    prompt='Write a one-sentence story.',
    options={
        'temperature': 0.9,
        'num_predict': 100,
    }
)

print(response.response)

print("\n")

# Example 8: Embeddings
print("Example 8: Embeddings")
print("-" * 50)

from oprel import embed

# Single text embedding
embedding = embed("Hello world", model="nomic-embed-text")
print(f"Single embedding dimensions: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")

# Batch embeddings
batch_embeddings = embed(
    ["Hello", "World", "Test"],
    model="nomic-embed-text"
)
print(f"\nBatch embeddings count: {len(batch_embeddings)}")
print(f"Each embedding dimensions: {len(batch_embeddings[0])}")

# Using client method
embedding = client.embed("Machine learning", model="nomic-embed-text")
print(f"\nClient method dimensions: {len(embedding)}")

print("\n")

# Example 9: Create custom model variant
print("Example 9: Create Custom Model")
print("-" * 50)

result = client.create(
    model='python-tutor',
    from_='qwencoder',
    system='You are an expert Python teacher. Explain concepts simply.'
)

print(result['message'])

print("\n")
print("=" * 50)
print("All examples completed!")
