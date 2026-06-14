# Cloud Providers

While Oprel SDK is built as a "local-first" AI runtime, it recognizes that local hardware may not always be powerful enough for complex tasks (like deep coding or massive context analysis). Therefore, Oprel includes a robust Cloud Provider proxy system.

## Supported Providers
- **OpenAI** (GPT-4o, GPT-3.5)
- **Anthropic** (Claude 3.5 Sonnet, Haiku)
- **Groq** (Llama 3 70B, Mixtral)
- **Google Gemini**
- **Ollama** (External remote instances)

## Configuration
You can configure providers in two ways:
1. **Via `.env` file:** Place a `.env` file in the root of your project or in your `~/.oprel` directory.
   ```env
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   GROQ_API_KEY=gsk_...
   ```
2. **Via the WebUI:** Oprel Studio has a "Settings" modal where you can paste your API keys directly into the UI. This saves them securely to the local SQLite database.

## Architecture

The proxy layer (`oprel/server/services/providers.py`) sits between the API endpoints and the inference engine. 

1. **Model Routing:** When a request is made for a model like `openai::gpt-4o`, Oprel detects the `::` prefix. 
2. **Key Resolution:** It fetches the appropriate API key from the database or environment variables.
3. **Format Translation:** It translates Oprel's internal message structure (or the incoming Ollama-compatible structure) into the specific schema required by the cloud provider.
4. **Streaming Relay:** It opens an asynchronous HTTP stream to the provider and relays the Server-Sent Events (SSE) back to the client exactly as if it were a local model.

This allows you to hot-swap between local models and cloud models in your applications without changing a single line of client code.
