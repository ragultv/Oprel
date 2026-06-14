# Skills (Function Calling)

Oprel supports advanced Function Calling workflows, natively matching the OpenAI Tool Calling specification. This allows the LLM to realize when it needs external information (like the current weather, or running a terminal command) and request the execution of a "Skill".

## API Usage

To define a skill, you pass a JSON schema describing the function signature to the generation endpoint.

```json
{
  "model": "llama3-8b",
  "messages": [{"role": "user", "content": "What's the weather like in Tokyo?"}],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "City name, e.g. Tokyo"}
          },
          "required": ["location"]
        }
      }
    }
  ]
}
```

If the model decides to use the tool, it will return a `tool_calls` block instead of plain text:
```json
{
  "message": {
    "role": "assistant",
    "tool_calls": [
      {
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\": \"Tokyo\"}"
        }
      }
    ]
  }
}
```

## Frontend Integration (Oprel Studio)

In the React WebUI (`oprel/webui-react/services/skills.ts`), Skills are executed dynamically. 
1. The UI passes the tool schemas to the daemon.
2. The daemon returns a `tool_calls` instruction.
3. The UI intercepts the call, executes the TypeScript function associated with `get_weather`.
4. The UI sends a new message back to the daemon containing the `tool_response` (the JSON weather data).
5. The daemon generates the final human-readable answer.

This decouples the dangerous execution of arbitrary code from the backend server, placing the responsibility on the frontend client.
