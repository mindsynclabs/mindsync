import os
import json
import httpx
import lancedb
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from sentence_transformers import SentenceTransformer  # <-- Added

router = APIRouter()

DB_PATH = os.path.expanduser("./lancedb_data")
TABLE_NAME = "mindsync_documents"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:0.5b"  # Fallback: "qwen2.5:7b"

# Initialize the embedding model globally (matches your ingestion model)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # <-- Added

class ChatRequest(BaseModel):
    query: str


def get_table():
    if not os.path.exists(DB_PATH):
        return None
    db = lancedb.connect(DB_PATH)
    existing_tables = list(db.list_tables())
    if TABLE_NAME in existing_tables:
        return db.open_table(TABLE_NAME)
    return None


def generate_query_embedding(text: str) -> list[float]:
    """
    Generates a real 384-dimensional vector embedding using all-MiniLM-L6-v2
    so LanceDB can perform semantic vector calculations.
    """
    # Convert text to a numpy array, then convert it to a plain Python list of floats
    return embedding_model.encode(text).tolist()  # <-- Replaced [0.0] * 384


@router.post("/chat")
async def chat_with_mindsync(req: ChatRequest):

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    table = get_table()

    if table is None:
        return {
            "status": "warning",
            "prompt": req.query,
            "retrieved_context": "",
            "response": "No knowledge base found. Please ingest some documents first.",
        }

    # 1. Retrieve vector context from LanceDB
    try:
        # Generates a real vector from the question text
        query_vector = generate_query_embedding(req.query)
        
        # LanceDB now finds the closest matching vectors in the 384-dimensional space
        results = table.search(query_vector).limit(3).to_list()
        retrieved_context = "\n".join(
            [r.get("text", "") for r in results if r.get("text")]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LanceDB search failed: {str(e)}")

    if not retrieved_context.strip():
        retrieved_context = "No relevant knowledge found in the database."

    # 2. Build system prompt for context synthesis
    system_prompt = (
        "You are MindSync, a specialized local AI assistant. "
        "Answer the user's question using ONLY the provided context below. "
        "If the answer cannot be derived from the context, state that clearly.\n\n"
        f"Context:\n{retrieved_context}"
    )

    # 3. Request completion from local Ollama instance (e.g. qwen2.5:0.5b)
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            print(f"Sending request to Ollama at {OLLAMA_URL} with model {MODEL_NAME}")
            ollama_res = await client.post(
                OLLAMA_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": f"{system_prompt}\n\nUser Question: {req.query}",
                    "stream": False,
                },
            )
            print(f"Ollama response: {ollama_res.json()}")
            ollama_res.raise_for_status()
            answer = ollama_res.json().get(
                "response", "No response generated from model."
            )
    except Exception as e:
        answer = f"Error reaching local Ollama service: {str(e)}"

    return {
        "status": "success",
        "prompt": req.query,
        "retrieved_context": retrieved_context,
        "response": answer,
    }
