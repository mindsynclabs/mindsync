import os
import json
import io
import lancedb
from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

router = APIRouter()

DB_PATH = os.path.expanduser("./lancedb_data")
TABLE_NAME = "mindsync_documents"

# Real local embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_db():
    os.makedirs(DB_PATH, exist_ok=True)
    return lancedb.connect(DB_PATH)

class IngestRequest(BaseModel):
    doc_id: Optional[str] = "doc_default"
    text: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

def generate_embedding(text: str) -> list[float]:
    return embedding_model.encode(text).tolist()

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Splits text into clean, smaller paragraphs so semantic search works accurately."""
    # Simple character splitter that preserves sentences/lines where possible
    lines = text.split("\n")
    chunks = []
    current_chunk = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(current_chunk) + len(line) < chunk_size:
            current_chunk += line + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = line + " "
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    # Fallback if text has no newlines
    if not chunks and text.strip():
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]
        
    return chunks

def save_chunks_to_db(doc_id: str, text_chunks: list[str], base_metadata: dict):
    """Helper to embed and insert individual chunks into LanceDB."""
    db = get_db()
    records = []
    
    for index, chunk in enumerate(text_chunks):
        # Generate embedding for the specific small chunk, not the whole document!
        vector = generate_embedding(chunk)
        
        # Keep chunk-specific metadata
        metadata = base_metadata.copy()
        metadata["chunk_index"] = index
        
        records.append({
            "doc_id": f"{doc_id}_chunk_{index}",
            "vector": vector,
            "text": chunk,
            "metadata": json.dumps(metadata)
        })

    existing_tables = list(db.list_tables())
    if TABLE_NAME in existing_tables:
        tbl = db.open_table(TABLE_NAME)
        tbl.add(records)
    else:
        db.create_table(TABLE_NAME, data=records)
        
    return len(records)

async def extract_text_from_file(file: UploadFile) -> str:
    filename = file.filename.lower()
    content = await file.read()
    await file.seek(0) 

    if filename.endswith(('.txt', '.md')):
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="Text file must be UTF-8 encoded.")
    elif filename.endswith('.pdf'):
        try:
            pdf_stream = io.BytesIO(content)
            reader = PdfReader(pdf_stream)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF file: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file extension.")

@router.post("/knowledge/ingest")
async def ingest_document(req: IngestRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text content cannot be empty.")

    # Split text into small chunks
    chunks = chunk_text(req.text)
    
    # Save split records
    chunks_count = save_chunks_to_db(
        doc_id=req.doc_id or "doc_text", 
        text_chunks=chunks, 
        base_metadata=req.metadata or {}
    )

    return {
        "status": "success",
        "message": f"Successfully split into {chunks_count} chunks and ingested.",
        "doc_id": req.doc_id
    }

@router.post("/knowledge/upload")
async def upload_document_endpoint(
    file: UploadFile = File(..., description="The document file to upload (.txt, .md, .pdf)"),
    doc_id: Optional[str] = Form(None, description="Optional custom document ID")
):
    try:
        target_doc_id = doc_id or file.filename
        extracted_text = await extract_text_from_file(file)
        
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")

        # Clean raw artifact strings like literal "string" text if accidentally sent
        cleaned_text = extracted_text.replace("\r", "").strip()

        # Split file text into manageable semantic pieces
        chunks = chunk_text(cleaned_text)

        # Save to DB
        meta = {"filename": file.filename, "content_type": file.content_type}
        chunks_count = save_chunks_to_db(doc_id=target_doc_id, text_chunks=chunks, base_metadata=meta)
        
        return {
            "status": "success", 
            "doc_id": target_doc_id,
            "chunks_created": chunks_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
