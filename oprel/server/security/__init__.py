"""
Security primitives for the Oprel daemon.

- guard:      app-wide LocalGuardMiddleware (Host/Origin checks, bearer token for non-browser clients)
- local_auth: extra token check for endpoints that act with stored credentials
              (integrations, tool approvals, tool-enabled chat)
- secrets:    OS-keychain backed credential storage (write-only from the API's view)
- redaction (oprel.utils.redaction): scrub secrets from logs, errors and tool output
"""

from oprel.server.security.guard import LOOPBACK_ORIGIN_REGEX, LocalGuardMiddleware

__all__ = ["LOOPBACK_ORIGIN_REGEX", "LocalGuardMiddleware"]
