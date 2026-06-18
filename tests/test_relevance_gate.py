import pytest
from oprel.core.groups.relevance import parse_relevance_response

def test_parse_clean_json():
    output = '{"decision": "respond", "emoji": null, "reason": "I have a specific correction."}'
    result = parse_relevance_response(output)
    assert result.decision == "respond"
    assert result.emoji is None
    
def test_parse_markdown_json():
    output = '''```json
{
  "decision": "react",
  "emoji": "👍",
  "reason": "I agree with Claude."
}
```'''
    result = parse_relevance_response(output)
    assert result.decision == "react"
    assert result.emoji == "👍"
    
def test_parse_messy_json_with_preamble():
    output = '''Here is my decision based on the rules provided:
{"decision": "interrupt", "emoji": null, "reason": "GPT-4 stated the API is rate-limited, but it is not."}
This is why I must interrupt.'''
    result = parse_relevance_response(output)
    assert result.decision == "interrupt"
    
def test_parse_broken_json_fallback():
    # Invalid JSON (missing quotes around keys) but fallback logic should catch the keyword
    output = '{decision: "react", emoji: "🔥", reason: "Good point"}'
    result = parse_relevance_response(output)
    assert result.decision == "react"
    assert result.emoji == "👍" # Fallback assigns default react emoji
    
def test_parse_garbage_defaults_to_silent():
    output = "I'm not sure what to say, but I think the database should be Redis."
    result = parse_relevance_response(output)
    assert result.decision == "silent"
    assert "Failed to parse" in result.reason
