from __future__ import annotations

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse

from oprel.server.schemas.common import IngestRequest, DocumentInfo
from oprel.server.services import knowledge as knowledge_service

router = APIRouter()


@router.post("/knowledge/ingest")
async def ingest_knowledge(request: IngestRequest):
    try:
        return knowledge_service.ingest_text_or_file(request.text, request.file_path, request.metadata)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/knowledge/search")
async def search_knowledge(q: str, top_k: int = 5):
    return await knowledge_service.search_knowledge(q, top_k=top_k)


@router.get("/index/search")
async def search_index(q: str, top_k: int = 5):
    return await knowledge_service.search_knowledge(q, top_k=top_k)


@router.post("/knowledge/reset")
async def reset_knowledge():
    try:
        return knowledge_service.reset_knowledge()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/index/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        return await knowledge_service.upload_document(file.filename, file.file)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/chat/upload")
async def chat_upload_document(file: UploadFile = File(...), model_id: str | None = None, reply_reserve: int | None = None):
    try:
        return await knowledge_service.chat_extract_file(file.filename, file.file, model_id=model_id, reply_reserve=reply_reserve)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/index/documents", response_model=list[DocumentInfo])
async def list_documents():
    return knowledge_service.list_documents()


@router.delete("/index/documents/{filename}")
@router.delete("/knowledge/documents/{filename}")
async def delete_indexed_document(filename: str):
    return await knowledge_service.delete_document(filename)


@router.delete("/index/documents")
@router.delete("/knowledge/documents")
async def delete_all_indexed_documents():
    return await knowledge_service.delete_all_documents()


import mimetypes

@router.get("/index/documents/{filename}/preview")
async def preview_document(filename: str):
    res = knowledge_service.get_document_content(filename)
    if not res.get("success"):
        raise HTTPException(status_code=404, detail=res.get("message"))
    return res


@router.get("/index/documents/{filename}/raw")
async def get_raw_document(filename: str):
    file_path = knowledge_service.get_document_raw_path(filename)
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="Raw document file not found")
    
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if not mime_type:
        mime_type = "application/pdf" if filename.lower().endswith(".pdf") else "application/octet-stream"

    return FileResponse(
        file_path,
        media_type=mime_type,
        content_disposition_type="inline",
        filename=filename,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )
