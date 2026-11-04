import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.modules.knowledge.ingester import document_ingester
from src.modules.knowledge.store import vector_store

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


class IngestRequest(BaseModel):
    doc_id: str = Field(pattern=r"^[A-Za-z0-9_.\-]+$", max_length=128, examples=["my-document"])
    text: str = Field(min_length=1)
    metadata: dict = Field(default_factory=dict)


@router.post("/ingest")
async def ingest(req: IngestRequest):
    try:
        count = await document_ingester.ingest_document(req.doc_id, req.text, req.metadata)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Embedding service error: {e}")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"doc_id": req.doc_id, "chunks_indexed": count}


@router.get("/stats")
async def stats():
    table = vector_store.get_or_create_table()
    return {
        "db": vector_store.uri,
        "rows": table.count_rows(),
        "sample_ids": [r["id"] for r in table.head(5).to_pylist()],
    }