"""Core RAG logic: LlamaIndex + LanceDB + Ollama (local LLM)."""
import json
import os
from typing import Any, Dict

import lancedb
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.lancedb import LanceDBVectorStore

# ---- Config (override with environment variables) --------------------------
DB_PATH = os.getenv("DB_PATH", "./lancedb_data")
TABLE_NAME = os.getenv("TABLE_NAME", "rag_documents")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# "context" = 1 LLM call per message (faster on CPU, weaker on follow-ups)
# "condense_plus_context" = rewrites follow-ups into standalone questions first
CHAT_MODE = os.getenv("CHAT_MODE", "condense_plus_context")
TOP_K = int(os.getenv("TOP_K", "3"))

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer using ONLY the provided context. "
    "If the context does not contain the answer, say you don't know. Be concise."
)

# ---- LlamaIndex global settings ----------------------------------------------
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
Settings.llm = Ollama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_URL,
    request_timeout=300.0,  # CPU inference can be slow
    context_window=4096,
    temperature=0.1,
)
# chunk_size / chunk_overlap are in tokens
Settings.node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=32)

# ---- Vector store + index ----------------------------------------------------
os.makedirs(DB_PATH, exist_ok=True)
vector_store = LanceDBVectorStore(uri=DB_PATH, table_name=TABLE_NAME)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)


# ---- Ingest ------------------------------------------------------------------
# LanceDB fixes the table schema on the first insert, so every document must
# carry the SAME metadata keys. Anything else is folded into a JSON "extra" field.
FIXED_META_KEYS = ("source", "filename", "file_hash")


def normalize_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
    out = {k: str(metadata.get(k, "")) for k in FIXED_META_KEYS}
    extra = {k: v for k, v in metadata.items() if k not in FIXED_META_KEYS}
    out["extra"] = json.dumps(extra, default=str) if extra else ""
    return out


def delete_document(doc_id: str) -> None:
    try:
        index.delete_ref_doc(doc_id)
    except Exception:
        pass  # table may not exist yet, or doc_id not found


def ingest_text(doc_id: str, text: str, metadata: Dict[str, Any]) -> int:
    """Chunk, embed and store text. Re-ingesting a doc_id replaces its old chunks."""
    delete_document(doc_id)
    metadata = normalize_metadata(metadata)

    doc = Document(text=text, doc_id=doc_id, metadata=metadata)
    # Keep metadata out of the embedded / LLM text so it doesn't skew results.
    doc.excluded_embed_metadata_keys = list(metadata.keys())
    doc.excluded_llm_metadata_keys = list(metadata.keys())

    nodes = Settings.node_parser.get_nodes_from_documents([doc])
    index.insert_nodes(nodes)
    return len(nodes)


# ---- Chat (one engine + memory per session) -----------------------------------
_sessions: Dict[str, Any] = {}


class EmptyIndexError(Exception):
    """Raised when chatting before any document has been ingested."""


def has_documents() -> bool:
    # LanceDB creates the table lazily on the first insert
    try:
        lancedb.connect(DB_PATH).open_table(TABLE_NAME)
        return True
    except Exception:
        return False


def _get_engine(session_id: str):
    if session_id not in _sessions:
        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        _sessions[session_id] = index.as_chat_engine(
            chat_mode=CHAT_MODE,
            memory=memory,
            system_prompt=SYSTEM_PROMPT,
            similarity_top_k=TOP_K,
        )
    return _sessions[session_id]


def chat(session_id: str, message: str) -> Dict[str, Any]:
    if not has_documents():
        raise EmptyIndexError("No documents ingested yet.")
    response = _get_engine(session_id).chat(message)
    sources = [
        {
            "doc_id": n.node.ref_doc_id,
            "score": n.score,
            "snippet": n.node.get_content()[:200],
            "metadata": n.node.metadata,
        }
        for n in response.source_nodes
    ]
    return {"answer": str(response), "sources": sources}


def reset_session(session_id: str) -> None:
    _sessions.pop(session_id, None)