from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from oprel.server.schemas.generation import GenerateRequest, GenerateResponse, EmbedRequest
from oprel.server.services.generation import (
    GenerateParams,
    EmbeddingParams,
    generate_text,
    get_embeddings,
    StreamResult,
)

router = APIRouter()


@router.post("/generate")
async def generate(request: GenerateRequest):
    params = GenerateParams(
        model_id=request.model_id,
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repeat_penalty=request.repeat_penalty,
        stream=request.stream,
        images=request.images,
        conversation_id=request.conversation_id,
        system_prompt=request.system_prompt,
        reset_conversation=request.reset_conversation,
        thinking=request.thinking,
        rag=request.rag,
    )

    try:
        result = await generate_text(params)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(exc)}")

    if isinstance(result, StreamResult):
        return StreamingResponse(
            result.iterator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Conversation-ID": result.conversation_id,
            },
        )

    return GenerateResponse(
        text=result.text,
        model_id=result.model_id,
        conversation_id=result.conversation_id,
        message_count=result.message_count,
    )


@router.post("/embedding")
@router.post("/v1/embeddings")
async def embeddings(request: EmbedRequest):
    try:
        result = await get_embeddings(EmbeddingParams(model=request.model, input=request.input))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(exc)}")

    if result.embedding is not None:
        return {"embedding": result.embedding}
    return {"embeddings": result.embeddings}
