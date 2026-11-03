from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

from src.modules.knowledge.ingest import document_ingester
from src.modules.knowledge.retriever import knowledge_retriever
from src.modules.knowledge.loader import document_loader

router = APIRouter(prefix="/knowledge", tags=["Knowledge Base (RAG)"])

class IngestRequest(BaseModel):
    doc_id: str = Field(..., description="Unique identifier for the document")
    text: str = Field(..., description="Raw text content to ingest and chunk")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="Optional metadata dictionary")

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query string")
    limit: Optional[int] = Field(default=3, description="Number of top results to return")

@router.post("/ingest", summary="Ingest raw text into local LanceDB")
async def ingest_endpoint(payload: IngestRequest):
    """Chunks raw text, generates local embeddings via Ollama, and stores them in LanceDB."""
    try:
        chunks_count = await document_ingester.ingest_document(
            doc_id=payload.doc_id,
            text=payload.text,
            metadata=payload.metadata
        )
        return {
            "status": "success",
            "doc_id": payload.doc_id,
            "chunks_ingested": chunks_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload", summary="Upload and ingest a document file (.txt, .md)")
async def upload_document_endpoint(
    file: UploadFile = File(..., description="The document file to upload"),
    doc_id: Optional[str] = Form(None, description="Optional custom document ID (defaults to filename)")
):
    """Accepts an uploaded file, extracts text, chunks it, and ingests it into LanceDB."""
    try:
        # Default doc_id to the original filename if not provided
        target_doc_id = doc_id or file.filename
        
        # Extract text using our document loader
        extracted_text = await document_loader.extract_text(file)
        
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="The uploaded file is empty or text could not be extracted.")

        # Ingest into vector store
        chunks_count = await document_ingester.ingest_document(
            doc_id=target_doc_id,
            text=extracted_text,
            metadata={"filename": file.filename, "content_type": file.content_type}
        )
        
        return {
            "status": "success",
            "filename": file.filename,
            "doc_id": target_doc_id,
            "chunks_ingested": chunks_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search", summary="Perform semantic vector search")
async def search_endpoint(payload: SearchRequest):
    """Searches the local knowledge base using vector similarity."""
    try:
        results = await knowledge_retriever.search(
            query=payload.query,
            limit=payload.limit
        )
        return {
            "status": "success",
            "query": payload.query,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))