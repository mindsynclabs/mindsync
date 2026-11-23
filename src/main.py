import hashlib
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

import ocr
import rag
import telegram_service
from typing import Any, Dict, List


# Telegram runs inside this app. It is off unless TELEGRAM_BOT_TOKEN is set.
telegram_service.configure(rag, ocr)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await telegram_service.startup()
    yield
    await telegram_service.shutdown()


app = FastAPI(title="RAG Prototype", lifespan=lifespan)
app.include_router(telegram_service.router)


# ---- Schemas -------------------------------------------------------------------
class TextIn(BaseModel):
    doc_id: str = "doc_default"
    text: str
    # Keep values flat (str/int/float/bool)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatIn(BaseModel):
    message: str
    session_id: str = "default"


# 1) Model phải nằm TRƯỚC route
class DocIn(BaseModel):
    doc_id: str = Field(..., min_length=1)
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---- Helpers -------------------------------------------------------------------
def extract_text(filename: str, content: bytes) -> str:
    name = filename.lower()
    if name.endswith((".txt", ".md")):
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(400, "Text file must be UTF-8 encoded.")
    if name.endswith(".pdf"):
        try:
            return ocr.extract_pdf_text(content)
        except Exception as e:
            raise HTTPException(400, f"Failed to parse PDF: {e}")
    raise HTTPException(400, "Unsupported file type. Use .pdf, .txt or .md")


# ---- Endpoints -----------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "llm": rag.OLLAMA_MODEL, "embed": rag.EMBED_MODEL}


@app.post("/documents/ingest")
async def insert_text(body: TextIn):
    if not body.text.strip():
        raise HTTPException(400, "Text cannot be empty.")
    chunks = await run_in_threadpool(
        rag.ingest_text, body.doc_id, body.text, body.metadata
    )
    return {"status": "success", "doc_id": body.doc_id, "chunks": chunks}



@app.post("/documents/ingest/batch")
def ingest(docs: List[DocIn]):   # body là một mảng
    if not docs:
        raise HTTPException(status_code=400, detail="docs is empty")
    try:
        result = rag.ingest_documents([d.model_dump() for d in docs])
    except ValueError as e:      # lỗi validate_docs (trùng doc_id, thiếu text...)
        raise HTTPException(status_code=400, detail=str(e))
    return {"ingested": result, "total_chunks": sum(result.values())}

@app.post("/documents/upload")
async def upload_file(
    file: UploadFile = File(...),
    doc_id: Optional[str] = Form(None),
):
    content = await file.read()
    text = await run_in_threadpool(extract_text, file.filename or "", content)
    text = text.replace("\r", "").strip()
    if not text:
        raise HTTPException(
            400, "No text found (scanned PDFs need OCR before they can be indexed)."
        )

    target_id = doc_id or file.filename
    meta = {
        "filename": file.filename,
        "file_hash": hashlib.sha256(content).hexdigest(),
    }
    chunks = await run_in_threadpool(rag.ingest_text, target_id, text, meta)
    return {"status": "success", "doc_id": target_id, "chunks": chunks}


@app.delete("/documents/{doc_id}")
async def delete_doc(doc_id: str):
    await run_in_threadpool(rag.delete_document, doc_id)
    return {"status": "deleted", "doc_id": doc_id}


@app.post("/chat")
async def chat(body: ChatIn):
    if not body.message.strip():
        raise HTTPException(400, "Message cannot be empty.")
    try:
        return await run_in_threadpool(rag.chat, body.session_id, body.message)
    except rag.EmptyIndexError:
        raise HTTPException(
            409,
            "No documents yet. Add some first via POST /documents/text "
            "or POST /documents/upload, then chat.",
        )
    except Exception as e:
        raise HTTPException(
            502, f"Chat failed (is Ollama running and the model pulled?): {e}"
        )


@app.delete("/chat/{session_id}")
def reset_chat(session_id: str):
    rag.reset_session(session_id)
    return {"status": "reset", "session_id": session_id}