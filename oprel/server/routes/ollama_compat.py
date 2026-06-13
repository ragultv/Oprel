from __future__ import annotations

import time as time_module

from fastapi import APIRouter

from oprel.server.schemas.openai import OpenAIChatRequest, OpenAIChatMessage, OpenAICompletionRequest
from oprel.server.routes.openai_compat import _handle_chat_completions, v1_completions, v1_list_models
from oprel.server.services.images import generate_image

router = APIRouter()


@router.post("/api/chat")
async def api_chat(request: dict):
    openai_req = OpenAIChatRequest(
        model=request.get("model", ""),
        messages=[OpenAIChatMessage(**msg) for msg in request.get("messages", [])],
        stream=request.get("stream", False),
        temperature=request.get("options", {}).get("temperature", 0.7),
        max_tokens=request.get("options", {}).get("num_predict", 512),
    )
    return await _handle_chat_completions(openai_req, "")


@router.post("/api/generate")
async def api_generate(request: dict):
    openai_req = OpenAICompletionRequest(
        model=request.get("model", ""),
        prompt=request.get("prompt", ""),
        stream=request.get("stream", False),
        temperature=request.get("options", {}).get("temperature", 0.7),
        max_tokens=request.get("options", {}).get("num_predict", 512),
    )
    return await v1_completions(openai_req)


@router.post("/api/images/generate")
async def api_generate_image(request: dict):
    return await generate_image(
        prompt=request.get("prompt", ""),
        model=request.get("model"),
        response_format=request.get("response_format"),
        size=request.get("size"),
        negative_prompt=request.get("negative_prompt"),
        steps=request.get("steps"),
        cfg_scale=request.get("cfg_scale"),
        seed=request.get("seed"),
        sampler=request.get("sampler"),
    )


@router.get("/api/tags")
async def api_tags():
    models_response = await v1_list_models()
    return {
        "models": [
            {
                "name": model["id"],
                "modified_at": time_module.strftime("%Y-%m-%dT%H:%M:%SZ", time_module.gmtime(model["created"])),
                "size": 0,
                "digest": "",
            }
            for model in models_response["data"]
        ]
    }
