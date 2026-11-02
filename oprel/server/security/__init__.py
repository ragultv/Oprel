"""
Security primitives for the Oprel daemon.

- redaction (oprel.utils.redaction): scrub secrets from logs, errors and tool output
- secrets:   OS-keychain backed credential storage (write-only from the API's view)
- local_auth: per-process API token, Host/Origin guards for privileged endpoints
"""
