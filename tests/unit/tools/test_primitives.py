import base64
import logging

import pytest

from oprel.integrations.azure_devops.markdown_lite import html_to_text, markdown_to_html
from oprel.tools.validation import ToolArgumentError, validate_arguments
from oprel.utils.redaction import REDACTED, RedactingFilter, redact, register_secret

SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "integer", "minimum": 1},
        "title": {"type": "string", "maxLength": 10},
        "kind": {"type": "string", "enum": ["Task", "Bug"]},
        "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "urgent": {"type": "boolean"},
    },
    "required": ["id"],
    "additionalProperties": False,
}


class TestValidation:
    def test_coerces_sloppy_local_model_output(self):
        out = validate_arguments(SCHEMA, {"id": "#42", "kind": "task", "tags": "a, b", "urgent": "true"})
        assert out == {"id": 42, "kind": "Task", "tags": ["a", "b"], "urgent": True}

    def test_drops_unknown_and_null_keys(self):
        assert validate_arguments(SCHEMA, {"id": 1, "bogus": "x", "title": None}) == {"id": 1}

    @pytest.mark.parametrize(
        "args",
        [{}, {"id": 0}, {"id": "abc"}, {"id": 1, "title": "x" * 11}, {"id": 1, "kind": "Epic"},
         {"id": 1, "tags": ["a", "b", "c", "d"]}, {"id": True}],
    )
    def test_rejects_invalid(self, args):
        with pytest.raises(ToolArgumentError):
            validate_arguments(SCHEMA, args)


class TestRedaction:
    def test_registered_secret_and_its_encodings_are_removed(self):
        pat = "abcd1234efgh5678ijkl9012mnop3456"
        register_secret(pat)
        basic = base64.b64encode(f":{pat}".encode()).decode()
        text = redact(f"token={pat} header=Basic {basic} raw {basic}")
        assert pat not in text and basic not in text
        assert REDACTED in text

    def test_patterns_catch_unregistered_credentials(self):
        assert "s3cr3t" not in redact('{"api_key": "s3cr3t-value"}')
        assert "zzzz" not in redact("Authorization: Bearer zzzzzzzz")
        # but ordinary words survive
        assert redact("max_tokens: 512") == "max_tokens: 512"

    def test_logging_filter_scrubs_messages_and_tracebacks(self):
        secret = "filter-secret-0123456789"
        register_secret(secret)
        record = logging.LogRecord("t", logging.ERROR, __file__, 1, "value %s", (secret,), None)
        try:
            raise RuntimeError(f"boom {secret}")
        except RuntimeError:
            import sys

            record.exc_info = sys.exc_info()
        RedactingFilter().filter(record)
        assert secret not in record.getMessage()
        assert secret not in record.exc_text


class TestMarkdown:
    def test_escapes_html_from_model_output(self):
        html = markdown_to_html('<script>alert(1)</script> **bold** [x](javascript:alert(1))')
        assert "<script>" not in html
        assert "&lt;script&gt;" in html
        assert "<strong>bold</strong>" in html
        assert 'href="javascript' not in html

    def test_structure(self):
        html = markdown_to_html("## Scope\n- one\n- [x] two\n\n1. a\n2. b\n\n```\n<b>\n```\nSee [docs](https://x.y/?a=1&b=2)")
        assert "<h2>Scope</h2>" in html
        assert "<ul><li>one</li><li>&#9745; two</li></ul>" in html
        assert "<ol><li>a</li><li>b</li></ol>" in html
        assert "<pre><code>&lt;b&gt;</code></pre>" in html
        assert 'href="https://x.y/?a=1&amp;b=2"' in html

    def test_html_to_text(self):
        assert html_to_text("<p>Hello&nbsp;<b>world</b></p><p>bye</p>") == "Hello\xa0world\nbye"
