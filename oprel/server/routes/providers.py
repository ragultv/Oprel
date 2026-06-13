from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from oprel.server.schemas.providers import ProviderUpsertRequest, ProviderChatRequest
from oprel.server.services import providers as provider_service
from oprel.server.services.generation import StreamResult

router = APIRouter()


@router.get("/providers")
async def list_providers_route():
    return provider_service.list_providers()


@router.get("/providers/{provider_id}")
async def get_provider_route(provider_id: str):
    p = provider_service.get_provider(provider_id)
    if not p:
        raise HTTPException(status_code=404, detail="Provider not found")
    return p


@router.post("/providers/{provider_id}")
async def upsert_provider_route(provider_id: str, body: ProviderUpsertRequest):
    data = body.dict()
    data["id"] = provider_id
    return provider_service.upsert_provider(data)


@router.delete("/providers/{provider_id}")
async def delete_provider_route(provider_id: str):
    return provider_service.delete_provider(provider_id)


@router.get("/providers/{provider_id}/models")
async def fetch_provider_models_proxy(provider_id: str):
    try:
        return await provider_service.fetch_provider_models(provider_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Provider error: {str(exc)}")


@router.post("/providers/{provider_id}/chat")
async def provider_chat_proxy(provider_id: str, body: ProviderChatRequest):
    try:
        result = await provider_service.provider_chat_proxy(provider_id, body)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if isinstance(result, StreamResult):
        response = StreamingResponse(result.iterator, media_type="text/event-stream")
        response.headers["X-Conversation-ID"] = result.conversation_id
        return response

    return result
