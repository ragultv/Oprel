"""
Async Azure DevOps Services REST client (work item tracking only).

Security notes:
- Only https://dev.azure.com/{organization} is ever contacted. The organization is
  validated against ADO's naming rules, so config cannot redirect requests (SSRF).
- The PAT only appears in the Authorization header built per request. Error
  messages are constructed from status codes / ADO's `message` field and passed
  through redact() before leaving this module.
"""

from __future__ import annotations

import base64
import re
from typing import Any
from urllib.parse import quote

import httpx

from oprel.tools.base import ToolError
from oprel.utils.redaction import redact

API_VERSION = "7.1"
COMMENTS_API_VERSION = "7.1-preview.4"
BASE_URL = "https://dev.azure.com"
DEFAULT_TIMEOUT = httpx.Timeout(20.0, connect=10.0)

ORG_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,48}[A-Za-z0-9])?$")
# ADO forbids these in project names; rejecting them also keeps the URL path intact.
_PROJECT_FORBIDDEN = set('\\/:*?"<>|;#$%{}+=[],') | {chr(c) for c in range(32)}

SUMMARY_FIELDS = [
    "System.Id",
    "System.WorkItemType",
    "System.Title",
    "System.State",
    "System.AssignedTo",
    "System.Tags",
    "System.ChangedDate",
]


class AzureDevOpsError(ToolError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(redact(message))
        self.status_code = status_code


def normalize_organization(value: str) -> str:
    """Accept 'contoso', 'https://dev.azure.com/contoso[/...]' or 'https://contoso.visualstudio.com'."""
    v = (value or "").strip().rstrip("/")
    m = re.match(r"^(?:https?://)?dev\.azure\.com/([^/?#]+)", v, re.I)
    if m:
        v = m.group(1)
    else:
        m = re.match(r"^(?:https?://)?([^./?#]+)\.visualstudio\.com", v, re.I)
        if m:
            v = m.group(1)
    if not ORG_RE.match(v):
        raise ValueError("Organization must be the name from https://dev.azure.com/<organization>")
    return v


def validate_project(value: str) -> str:
    v = (value or "").strip()
    if not v or len(v) > 64 or any(ch in _PROJECT_FORBIDDEN for ch in v) or v.startswith((".", "_")):
        raise ValueError("Project name is not a valid Azure DevOps project name")
    return v


def web_url(organization: str, project: str, work_item_id: int) -> str:
    return f"{BASE_URL}/{organization}/{quote(project, safe='')}/_workitems/edit/{int(work_item_id)}"


class AzureDevOpsClient:
    def __init__(
        self,
        organization: str,
        project: str,
        pat: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        if not pat:
            raise AzureDevOpsError("No Personal Access Token is configured for Azure DevOps")
        self.organization = normalize_organization(organization)
        self.project = validate_project(project)
        self._pat = pat
        self._transport = transport

    # -- plumbing -----------------------------------------------------------

    @property
    def _org_url(self) -> str:
        return f"{BASE_URL}/{self.organization}"

    @property
    def _project_url(self) -> str:
        return f"{self._org_url}/{quote(self.project, safe='')}"

    def _headers(self, content_type: str = "application/json") -> dict[str, str]:
        token = base64.b64encode(f":{self._pat}".encode()).decode()
        return {
            "Authorization": f"Basic {token}",
            "Accept": "application/json",
            "Content-Type": content_type,
            # Stops ADO from answering an expired PAT with a 203 + HTML sign-in page.
            "X-TFS-FedAuthRedirect": "Suppress",
        }

    async def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: Any = None,
        content_type: str = "application/json",
    ) -> Any:
        query = {"api-version": API_VERSION, **(params or {})}
        try:
            async with httpx.AsyncClient(
                timeout=DEFAULT_TIMEOUT, transport=self._transport, follow_redirects=False
            ) as client:
                resp = await client.request(
                    method, url, params=query, json=json_body, headers=self._headers(content_type)
                )
        except httpx.TimeoutException:
            raise AzureDevOpsError("Azure DevOps did not respond in time. Try again shortly.")
        except httpx.HTTPError as exc:
            raise AzureDevOpsError(f"Could not reach Azure DevOps ({type(exc).__name__})")
        return self._handle(resp)

    @staticmethod
    def _handle(resp: httpx.Response) -> Any:
        ct = resp.headers.get("content-type", "")
        is_json = "json" in ct
        if resp.status_code in (200, 201) and is_json:
            return resp.json()
        if resp.status_code == 204:
            return {}

        ado_message = ""
        if is_json:
            try:
                ado_message = str(resp.json().get("message", ""))[:500]
            except ValueError:
                pass

        if resp.status_code in (203, 302) or resp.status_code == 401 or (resp.status_code == 200 and not is_json):
            raise AzureDevOpsError(
                "Azure DevOps rejected the Personal Access Token (invalid or expired). "
                "Reconnect the integration with a new token.",
                401,
            )
        if resp.status_code == 403:
            raise AzureDevOpsError(
                "The Personal Access Token lacks permission for this action. "
                "It needs the 'Work Items (Read & Write)' scope. " + ado_message,
                403,
            )
        if resp.status_code == 404:
            raise AzureDevOpsError(ado_message or "Not found in Azure DevOps.", 404)
        if resp.status_code == 429:
            raise AzureDevOpsError("Azure DevOps rate limit reached. Wait a moment and retry.", 429)
        raise AzureDevOpsError(
            ado_message or f"Azure DevOps request failed (HTTP {resp.status_code})", resp.status_code
        )

    # -- API ----------------------------------------------------------------

    async def list_work_item_types(self) -> list[str]:
        data = await self._request("GET", f"{self._project_url}/_apis/wit/workitemtypes")
        names = [t.get("name") for t in data.get("value", []) if not t.get("isDisabled")]
        return sorted(n for n in names if n)

    async def get_authenticated_user(self) -> str | None:
        try:
            data = await self._request("GET", f"{self._org_url}/_apis/connectionData", params={"api-version": "7.1-preview"})
        except AzureDevOpsError:
            return None
        user = data.get("authenticatedUser") or {}
        return user.get("providerDisplayName") or user.get("customDisplayName")

    async def create_work_item(self, work_item_type: str, operations: list[dict[str, Any]]) -> dict[str, Any]:
        url = f"{self._project_url}/_apis/wit/workitems/${quote(work_item_type, safe='')}"
        return await self._request("POST", url, json_body=operations, content_type="application/json-patch+json")

    async def update_work_item(self, work_item_id: int, operations: list[dict[str, Any]]) -> dict[str, Any]:
        url = f"{self._project_url}/_apis/wit/workitems/{int(work_item_id)}"
        return await self._request("PATCH", url, json_body=operations, content_type="application/json-patch+json")

    async def get_work_item(self, work_item_id: int) -> dict[str, Any]:
        url = f"{self._project_url}/_apis/wit/workitems/{int(work_item_id)}"
        return await self._request("GET", url, params={"$expand": "relations"})

    async def query_ids(self, wiql: str, top: int) -> list[int]:
        data = await self._request(
            "POST", f"{self._project_url}/_apis/wit/wiql", params={"$top": int(top)}, json_body={"query": wiql}
        )
        return [int(w["id"]) for w in data.get("workItems", [])][:top]

    async def get_work_items_batch(self, ids: list[int], fields: list[str] | None = None) -> list[dict[str, Any]]:
        if not ids:
            return []
        data = await self._request(
            "POST",
            f"{self._project_url}/_apis/wit/workitemsbatch",
            json_body={"ids": ids[:200], "fields": fields or SUMMARY_FIELDS},
        )
        return data.get("value", [])

    async def add_comment(self, work_item_id: int, html_text: str) -> dict[str, Any]:
        url = f"{self._project_url}/_apis/wit/workItems/{int(work_item_id)}/comments"
        return await self._request(
            "POST", url, params={"api-version": COMMENTS_API_VERSION}, json_body={"text": html_text}
        )
