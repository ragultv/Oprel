from __future__ import annotations

from typing import Any

from oprel.integrations.azure_devops.client import BASE_URL, AzureDevOpsClient, web_url
from oprel.integrations.azure_devops.markdown_lite import html_to_text, markdown_to_html
from oprel.tools.base import Tool, ToolContext, ToolError

_TYPE_ALIASES = {
    "story": "User Story",
    "userstory": "User Story",
    "pbi": "Product Backlog Item",
    "backlog item": "Product Backlog Item",
    "defect": "Bug",
}

_PARENT_REL = "System.LinkTypes.Hierarchy-Reverse"


def _client(ctx: ToolContext) -> AzureDevOpsClient:
    return AzureDevOpsClient(
        organization=ctx.config.get("organization", ""),
        project=ctx.config.get("project", ""),
        pat=ctx.secrets.get("pat", ""),
    )


def _resolve_type(ctx: ToolContext, requested: str) -> str:
    known: list[str] = ctx.config.get("work_item_types") or []
    wanted = _TYPE_ALIASES.get(requested.strip().lower().replace("_", " "), requested.strip())
    if not known:
        return wanted
    for name in known:
        if name.lower() == wanted.lower():
            return name
    raise ToolError(f"Unknown work item type '{requested}'. This project supports: {', '.join(known)}")


def _set(path: str, value: Any) -> dict[str, Any]:
    return {"op": "add", "path": f"/fields/{path}", "value": value}


def _identity_name(value: Any) -> str | None:
    if isinstance(value, dict):
        return value.get("displayName") or value.get("uniqueName")
    return value or None


def _parent_id(item: dict[str, Any]) -> int | None:
    for rel in item.get("relations") or []:
        if rel.get("rel") == _PARENT_REL:
            try:
                return int(str(rel.get("url", "")).rstrip("/").rsplit("/", 1)[-1])
            except ValueError:
                return None
    return None


def _summary(ctx: ToolContext, item: dict[str, Any]) -> dict[str, Any]:
    fields = item.get("fields", {})
    wid = int(item.get("id") or fields.get("System.Id"))
    out: dict[str, Any] = {
        "id": wid,
        "type": fields.get("System.WorkItemType"),
        "title": fields.get("System.Title"),
        "state": fields.get("System.State"),
        "url": web_url(ctx.config["organization"], ctx.config["project"], wid),
    }
    assigned = _identity_name(fields.get("System.AssignedTo"))
    if assigned:
        out["assigned_to"] = assigned
    if fields.get("System.Tags"):
        out["tags"] = [t.strip() for t in fields["System.Tags"].split(";") if t.strip()]
    parent = _parent_id(item)
    if parent:
        out["parent_id"] = parent
    return out


def _common_field_ops(args: dict[str, Any]) -> list[dict[str, Any]]:
    ops: list[dict[str, Any]] = []
    if "title" in args:
        ops.append(_set("System.Title", args["title"].strip()))
    if "description" in args:
        ops.append(_set("System.Description", markdown_to_html(args["description"])))
    if "acceptance_criteria" in args:
        ops.append(_set("Microsoft.VSTS.Common.AcceptanceCriteria", markdown_to_html(args["acceptance_criteria"])))
    if "assigned_to" in args:
        ops.append(_set("System.AssignedTo", args["assigned_to"].strip()))
    if "area_path" in args:
        ops.append(_set("System.AreaPath", args["area_path"].strip()))
    if "iteration_path" in args:
        ops.append(_set("System.IterationPath", args["iteration_path"].strip()))
    if "priority" in args:
        ops.append(_set("Microsoft.VSTS.Common.Priority", int(args["priority"])))
    if "tags" in args:
        ops.append(_set("System.Tags", "; ".join(t.strip() for t in args["tags"] if t.strip())))
    return ops


def _parent_op(ctx: ToolContext, parent_id: int) -> dict[str, Any]:
    return {
        "op": "add",
        "path": "/relations/-",
        "value": {
            "rel": _PARENT_REL,
            "url": f"{BASE_URL}/{ctx.config['organization']}/_apis/wit/workItems/{int(parent_id)}",
        },
    }


# -- handlers ---------------------------------------------------------------

async def create_work_item(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    wtype = _resolve_type(ctx, args["work_item_type"])
    ops = _common_field_ops(args)
    if args.get("parent_id"):
        ops.append(_parent_op(ctx, args["parent_id"]))
    item = await _client(ctx).create_work_item(wtype, ops)
    return {"created": _summary(ctx, item)}


async def update_work_item(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    ops = _common_field_ops(args)
    if "state" in args:
        ops.append(_set("System.State", args["state"].strip()))
    if args.get("parent_id"):
        ops.append(_parent_op(ctx, args["parent_id"]))
    if not ops:
        raise ToolError("Nothing to update: provide at least one field to change.")
    item = await _client(ctx).update_work_item(args["id"], ops)
    return {"updated": _summary(ctx, item)}


async def get_work_item(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    item = await _client(ctx).get_work_item(args["id"])
    out = _summary(ctx, item)
    fields = item.get("fields", {})
    out["description"] = html_to_text(fields.get("System.Description"))
    criteria = html_to_text(fields.get("Microsoft.VSTS.Common.AcceptanceCriteria"))
    if criteria:
        out["acceptance_criteria"] = criteria
    for key, label in (
        ("System.AreaPath", "area_path"),
        ("System.IterationPath", "iteration_path"),
        ("Microsoft.VSTS.Common.Priority", "priority"),
    ):
        if fields.get(key) is not None:
            out[label] = fields[key]
    return {"work_item": out}


def _wiql_literal(value: str) -> str:
    return "'" + value.replace("'", "''")[:200] + "'"


async def search_work_items(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    clauses = ["[System.TeamProject] = @project"]
    if args.get("text"):
        clauses.append(f"[System.Title] CONTAINS {_wiql_literal(args['text'])}")
    if args.get("work_item_type"):
        clauses.append(f"[System.WorkItemType] = {_wiql_literal(_resolve_type(ctx, args['work_item_type']))}")
    if args.get("state"):
        clauses.append(f"[System.State] = {_wiql_literal(args['state'])}")
    if args.get("assigned_to_me"):
        clauses.append("[System.AssignedTo] = @me")
    wiql = (
        "SELECT [System.Id] FROM WorkItems WHERE "
        + " AND ".join(clauses)
        + " ORDER BY [System.ChangedDate] DESC"
    )
    top = int(args.get("top") or 10)
    client = _client(ctx)
    ids = await client.query_ids(wiql, top)
    items = await client.get_work_items_batch(ids)
    return {"count": len(items), "work_items": [_summary(ctx, i) for i in items]}


async def add_comment(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    result = await _client(ctx).add_comment(args["id"], markdown_to_html(args["text"]))
    return {
        "commented": {
            "work_item_id": args["id"],
            "comment_id": result.get("id"),
            "url": web_url(ctx.config["organization"], ctx.config["project"], args["id"]),
        }
    }


# -- schemas ----------------------------------------------------------------

_ID = {"type": "integer", "minimum": 1, "description": "Work item ID number, e.g. 1234"}
_TITLE = {"type": "string", "minLength": 1, "maxLength": 255, "description": "Short, specific title"}
_DESCRIPTION = {
    "type": "string",
    "maxLength": 20000,
    "description": "Detailed description in Markdown (headings, bullet lists, code blocks allowed)",
}
_OPTIONAL_FIELDS: dict[str, Any] = {
    "acceptance_criteria": {"type": "string", "maxLength": 10000, "description": "Acceptance criteria in Markdown"},
    "assigned_to": {"type": "string", "description": "Email or display name of the assignee"},
    "priority": {"type": "integer", "minimum": 1, "maximum": 4, "description": "1 (highest) to 4 (lowest)"},
    "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 20, "description": "Tag names"},
    "area_path": {"type": "string", "description": "Area path, e.g. 'Project\\\\Team'"},
    "iteration_path": {"type": "string", "description": "Iteration/sprint path, e.g. 'Project\\\\Sprint 5'"},
    "parent_id": {"type": "integer", "minimum": 1, "description": "ID of the parent work item to link under"},
}


def build_tools() -> list[Tool]:
    return [
        Tool(
            name="ado_create_work_item",
            title="Create Azure DevOps work item",
            kind="write",
            description=(
                "Create a new work item (Task, Bug, User Story, etc.) in the connected Azure DevOps project. "
                "Write a clear title and a well-structured Markdown description."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "work_item_type": {
                        "type": "string",
                        "description": "Work item type, e.g. 'Task', 'Bug', 'User Story', 'Feature', 'Epic'",
                    },
                    "title": _TITLE,
                    "description": _DESCRIPTION,
                    **_OPTIONAL_FIELDS,
                },
                "required": ["work_item_type", "title"],
                "additionalProperties": False,
            },
            handler=create_work_item,
        ),
        Tool(
            name="ado_update_work_item",
            title="Update Azure DevOps work item",
            kind="write",
            destructive=True,
            description="Change fields of an existing Azure DevOps work item. Only send the fields that change.",
            parameters={
                "type": "object",
                "properties": {
                    "id": _ID,
                    "title": _TITLE,
                    "description": {**_DESCRIPTION, "description": "Replaces the whole description (Markdown)"},
                    "state": {"type": "string", "description": "New state, e.g. 'New', 'Active', 'Resolved', 'Closed'"},
                    **_OPTIONAL_FIELDS,
                },
                "required": ["id"],
                "additionalProperties": False,
            },
            handler=update_work_item,
        ),
        Tool(
            name="ado_get_work_item",
            title="Read Azure DevOps work item",
            kind="read",
            description="Fetch one Azure DevOps work item by ID, including its description.",
            parameters={
                "type": "object",
                "properties": {"id": _ID},
                "required": ["id"],
                "additionalProperties": False,
            },
            handler=get_work_item,
        ),
        Tool(
            name="ado_search_work_items",
            title="Search Azure DevOps work items",
            kind="read",
            description="Find work items in the connected project by title text, type, state or assignee. Newest first.",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "maxLength": 200, "description": "Words that appear in the title"},
                    "work_item_type": {"type": "string", "description": "Filter by type, e.g. 'Bug'"},
                    "state": {"type": "string", "description": "Filter by state, e.g. 'Active'"},
                    "assigned_to_me": {"type": "boolean", "description": "Only items assigned to the token owner"},
                    "top": {"type": "integer", "minimum": 1, "maximum": 50, "description": "Max results (default 10)"},
                },
                "required": [],
                "additionalProperties": False,
            },
            handler=search_work_items,
        ),
        Tool(
            name="ado_add_comment",
            title="Comment on Azure DevOps work item",
            kind="write",
            description="Add a discussion comment (Markdown) to an existing Azure DevOps work item.",
            parameters={
                "type": "object",
                "properties": {
                    "id": _ID,
                    "text": {"type": "string", "minLength": 1, "maxLength": 10000, "description": "Comment in Markdown"},
                },
                "required": ["id", "text"],
                "additionalProperties": False,
            },
            handler=add_comment,
        ),
    ]
