"""
Third-party app integrations (Azure DevOps, and later GitHub, Jira, ...).

Each integration lives in its own subpackage and implements `Integration`
(see base.py). Registering it in `registry.BUILTIN_INTEGRATIONS` is all that is
needed for it to appear in the Integrations tab and expose tools to models.
This package must not import from oprel.server.
"""
