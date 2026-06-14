"""
Skills (Function Calling) Example for Oprel SDK.

Demonstrates providing tools/skills to the model.
"""

from oprel.client import OprelClient

def main():
    client = OprelClient()
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    messages = [{"role": "user", "content": "What's the weather like in Tokyo?"}]
    
    response = client.chat(model="llama3-8b", messages=messages, tools=tools)
    print("Response:", response)
    # The response should include a tool_calls block

if __name__ == "__main__":
    main()
