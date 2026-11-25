"""Core RAG logic: LlamaIndex + LanceDB + Ollama (local LLM) + re-ranking.

Pipeline, one function (or small group) per step
------------------------------------------------
INGEST  ingest_text()
          normalize_text / normalize_metadata   clean the input
          make_document -> chunk_document       cut into chunks (nodes)
          embed_nodes                           text -> vectors
          store_nodes                           write to LanceDB

CHAT    chat()
          documents_are_relevant()              routing (retrieve + score)
            retrieve -> rerank                  vector search, then cross-encoder
          build_rag_engine / build_general_engine   engines (prompt assembly)
          answer_with_documents / answer_general
          print_trace / PrintNodes              terminal debug output

Public API used by the web app is unchanged:
  ingest_text, delete_document, has_documents, chat, reset_session, EmptyIndexError
"""
import json
import logging
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional

import lancedb
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.lancedb import LanceDBVectorStore

# =============================================================================
# 1. CONFIG (override with environment variables)
# =============================================================================
DB_PATH = os.getenv("DB_PATH", "./lancedb_data")
TABLE_NAME = os.getenv("TABLE_NAME", "rag_documents")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# None = auto-detect (GPU if PyTorch can see one, otherwise CPU). Or "cpu" / "cuda" / "mps".
EMBED_DEVICE = os.getenv("EMBED_DEVICE") or None
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# "context" = 1 LLM call per message (faster on CPU, weaker on follow-ups)
# "condense_plus_context" = rewrites follow-ups into standalone questions first
CHAT_MODE = os.getenv("CHAT_MODE", "condense_plus_context")
TOP_K = int(os.getenv("TOP_K", "3"))

# "hybrid" = use the documents when they are relevant, otherwise answer from the
#            LLM's own general knowledge (and say the documents don't cover it)
# "strict" = answer ONLY from the documents; otherwise say "I don't know"
ANSWER_MODE = os.getenv("ANSWER_MODE", "hybrid")
# Routing for hybrid mode, by cosine score (used when re-ranking is OFF):
# if the best retrieval score is below this value, skip the documents and ask the
# LLM directly. 0 = off (the prompt alone decides).
MIN_SCORE = float(os.getenv("MIN_SCORE", "0"))

# Chunking (sizes are in tokens)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "256"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "32"))
# Clean the text before chunking (Unicode NFC, line endings, spaces). Helps Vietnamese.
NORMALIZE_TEXT = os.getenv("NORMALIZE_TEXT", "1") == "1"
# Compute embeddings in our own step (embed_nodes) instead of letting LlamaIndex do it
# inside insert_nodes. Same result; set 0 if ingest ever looks slower than before.
EMBED_EXPLICIT = os.getenv("EMBED_EXPLICIT", "1") == "1"

# Re-ranking. Flow: vector search fetches RETRIEVE_K chunks -> a cross-encoder re-scores
# each (question, chunk) pair -> only the best RERANK_TOP_N chunks go to the LLM.
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "1") == "1"
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "10"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", str(TOP_K)))
# Hybrid routing by RERANK score (0 = off). The scale depends on the model/library
# version, so calibrate it: watch the "top rerank score" log lines for questions
# that ARE in the documents vs. ones that are NOT, then pick a value in between.
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "0"))

# Print the retrieved chunks, the route and the answer to the terminal. DEBUG_RAG=0 = off.
DEBUG_RAG = os.getenv("DEBUG_RAG", "1") == "1"

# LanceDB fixes the table schema on the first insert, so every document must
# carry the SAME metadata keys. Anything else is folded into a JSON "extra" field.
FIXED_META_KEYS = ("source", "filename", "file_hash")

# =============================================================================
# 2. PROMPTS
# =============================================================================
STRICT_PROMPT = (
    "You are a helpful assistant. Answer using ONLY the provided context. "
    "If the context does not contain the answer, say you don't know. Be concise."
)
# NOTE: no quoted refusal text in this prompt. Small models copy any sentence they are
# told to say ("Not found in your documents.") even when the context has the answer.
# The "Not found in your documents." prefix is added by the code instead (see chat()).
HYBRID_PROMPT = (
    "You are a helpful assistant. Answer the question using the context from the "
    "user's documents. The context may be short or hold several facts; read all of it "
    "and draw simple conclusions (for example, if the context says someone is male, "
    "then that person is not female). If the context is clearly unrelated to the "
    "question, ignore it and answer from your own general knowledge. Be concise."
)
GENERAL_PROMPT = "You are a helpful assistant. Be concise."
SYSTEM_PROMPT = STRICT_PROMPT if ANSWER_MODE == "strict" else HYBRID_PROMPT

# LlamaIndex wraps the retrieved chunks in its OWN built-in template, and the default one
# for "condense_plus_context" ends with: 'Answer "don't know" if not present in the document.'
# That is why the model said "Don't know." even though the answer was in the context.
# We pass our own template so the instruction around the chunks is the one we want.
CONTEXT_PROMPT_HYBRID = (
    "Here are the relevant documents for the context:\n\n{context_str}\n\n"
    "Instruction: Use the documents above to answer the user's question. The documents may be "
    "short or hold several facts; read all of them and draw simple conclusions (for example, if "
    "the documents say someone is male, then that person is not female). If the documents are "
    "clearly unrelated to the question, ignore them and answer from your own general knowledge. "
    "Be concise.\n"
)
CONTEXT_PROMPT_STRICT = (
    "Here are the relevant documents for the context:\n\n{context_str}\n\n"
    "Instruction: Answer the user's question using ONLY the documents above. Draw simple "
    "conclusions if needed. If the documents do not contain the answer, say you don't know. "
    "Be concise.\n"
)
CONTEXT_PROMPT = CONTEXT_PROMPT_STRICT if ANSWER_MODE == "strict" else CONTEXT_PROMPT_HYBRID

log = logging.getLogger("rag")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


# =============================================================================
# 3. SETUP: builders for the models and the index (each does one thing)
# =============================================================================
def build_embed_model() -> HuggingFaceEmbedding:
    return HuggingFaceEmbedding(model_name=EMBED_MODEL, device=EMBED_DEVICE)


def build_llm() -> Ollama:
    return Ollama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_URL,
        request_timeout=300.0,  # CPU inference can be slow
        context_window=4096,
        temperature=0.1,
    )


def build_node_parser() -> SentenceSplitter:
    return SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def build_index() -> VectorStoreIndex:
    os.makedirs(DB_PATH, exist_ok=True)
    store = LanceDBVectorStore(uri=DB_PATH, table_name=TABLE_NAME)
    return VectorStoreIndex.from_vector_store(vector_store=store)


def build_reranker() -> Optional[SentenceTransformerRerank]:
    """Loaded once; the model downloads on first run (~90 MB). None when disabled."""
    if not RERANK_ENABLED:
        log.info("Re-ranking OFF")
        return None
    log.info("Re-ranking ON: %s (retrieve %d -> keep %d)", RERANK_MODEL, RETRIEVE_K, RERANK_TOP_N)
    return SentenceTransformerRerank(model=RERANK_MODEL, top_n=RERANK_TOP_N, device=EMBED_DEVICE)


def print_config() -> None:
    print(
        f"[rag] CONFIG: ANSWER_MODE={ANSWER_MODE} CHAT_MODE={CHAT_MODE} OLLAMA_MODEL={OLLAMA_MODEL} "
        f"RERANK={'on' if reranker is not None else 'off'} RETRIEVE_K={RETRIEVE_K} "
        f"TOP_N={RERANK_TOP_N} RERANK_MIN_SCORE={RERANK_MIN_SCORE} MIN_SCORE={MIN_SCORE} "
        f"CHUNK={CHUNK_SIZE}/{CHUNK_OVERLAP} NORMALIZE_TEXT={NORMALIZE_TEXT}",
        flush=True,
    )
    print(f"[rag] SYSTEM_PROMPT = {SYSTEM_PROMPT!r}", flush=True)
    print(f"[rag] CONTEXT_PROMPT = {CONTEXT_PROMPT!r}", flush=True)


# LlamaIndex global settings + the objects the rest of the file uses
Settings.embed_model = build_embed_model()
Settings.llm = build_llm()
Settings.node_parser = build_node_parser()
index = build_index()
reranker = build_reranker()
print_config()


# =============================================================================
# 4. NORMALIZATION
# =============================================================================
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\u200b\ufeff]")


def normalize_text(text: str) -> str:
    """Clean raw text before chunking. Keeps paragraph breaks.

    - Unicode NFC: "e" + combining accent and the single accented character become the
      same string (important for Vietnamese, otherwise identical words embed differently)
    - Windows/Mac line endings -> "\\n"
    - drops control and zero-width characters
    - collapses runs of spaces/tabs, and 3+ blank lines into one blank line
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
    """Same keys for every document (LanceDB schema); everything else goes into "extra"."""
    out = {k: str(metadata.get(k, "")) for k in FIXED_META_KEYS}
    extra = {k: v for k, v in metadata.items() if k not in FIXED_META_KEYS}
    out["extra"] = json.dumps(extra, default=str) if extra else ""
    return out


# =============================================================================
# 5. CHUNKING
# =============================================================================
def make_document(doc_id: str, text: str, metadata: Dict[str, str]) -> Document:
    doc = Document(text=text, doc_id=doc_id, metadata=metadata)
    # Keep metadata out of the embedded / LLM text so it doesn't skew results.
    doc.excluded_embed_metadata_keys = list(metadata.keys())
    doc.excluded_llm_metadata_keys = list(metadata.keys())
    return doc


def chunk_document(doc: Document) -> list:
    """Cut a document into chunks (LlamaIndex "nodes")."""
    return Settings.node_parser.get_nodes_from_documents([doc])


# =============================================================================
# 6. EMBEDDING
# =============================================================================
def embed_nodes(nodes: list) -> list:
    """Turn each chunk's text into a vector and attach it to the node.

    LlamaIndex skips nodes that already carry an embedding when they are inserted,
    so this is the same work, just done in a step of its own.
    """
    texts = [n.get_content(metadata_mode=MetadataMode.EMBED) for n in nodes]
    vectors = Settings.embed_model.get_text_embedding_batch(texts)
    for node, vector in zip(nodes, vectors):
        node.embedding = vector
    return nodes


def embed_query(text: str) -> List[float]:
    """Vector for a question (handy for debugging scores outside the engines)."""
    return Settings.embed_model.get_query_embedding(text)

# =============================================================================
# 7. STORAGE + THE INGEST PIPELINE
# =============================================================================
def delete_document(doc_id: str) -> None:
    try:
        index.delete_ref_doc(doc_id)
    except Exception:
        pass  # table may not exist yet, or doc_id not found


def store_nodes(nodes: list) -> None:
    index.insert_nodes(nodes)


def validate_docs(docs: List[Dict[str, Any]]) -> None:
    """Check the whole array BEFORE touching the DB, so a bad item can't leave a half-done ingest."""
    if not isinstance(docs, list):
        raise ValueError("docs must be a list of {doc_id, text, metadata}")
    seen = set()
    for i, d in enumerate(docs):
        if not isinstance(d, dict) or not d.get("doc_id") or not isinstance(d.get("text"), str):
            raise ValueError(f"docs[{i}] needs a 'doc_id' and a string 'text'")
        if d["doc_id"] in seen:
            raise ValueError(f"duplicate doc_id {d['doc_id']!r} at docs[{i}]")
        seen.add(d["doc_id"])


def prepare_nodes(doc_id: str, text: str, metadata: Dict[str, Any]) -> list:
    """One document -> chunks (normalize -> document -> chunk). No DB access."""
    if NORMALIZE_TEXT:
        text = normalize_text(text)
    metadata = normalize_metadata(metadata)
    return chunk_document(make_document(doc_id, text, metadata))


def ingest_documents(docs: List[Dict[str, Any]]) -> Dict[str, int]:
    """Ingest an array of documents: [{"doc_id": str, "text": str, "metadata": {...}}, ...]

    Returns {doc_id: number_of_chunks}. Re-ingesting a doc_id replaces its chunks.
    """
    validate_docs(docs)

    # 1) chunk every document, then embed ALL chunks in one batch.
    #    Nothing has been deleted yet, so a failure here leaves the DB untouched.
    nodes_by_doc: Dict[str, list] = {
        d["doc_id"]: prepare_nodes(d["doc_id"], d["text"], d.get("metadata") or {})
        for d in docs
    }
    all_nodes = [n for nodes in nodes_by_doc.values() for n in nodes]
    if EMBED_EXPLICIT and all_nodes:
        embed_nodes(all_nodes)  # sets node.embedding in place

    # 2) replace the old chunks, then write everything with a single insert
    for doc_id in nodes_by_doc:
        delete_document(doc_id)
    if all_nodes:
        store_nodes(all_nodes)

    return {doc_id: len(nodes) for doc_id, nodes in nodes_by_doc.items()}


def ingest_text(doc_id: str, text: str, metadata: Dict[str, Any]) -> int:
    """Single-document wrapper (unchanged API for the web app)."""
    result = ingest_documents([{"doc_id": doc_id, "text": text, "metadata": metadata}])
    return result[doc_id]

    
# =============================================================================
# 8. RETRIEVAL + RE-RANKING
# =============================================================================
def retrieve(query: str, k: int) -> list:
    """Vector search: the k chunks closest to the query (cosine score)."""
    return index.as_retriever(similarity_top_k=k).retrieve(query)


def rerank(nodes: list, query: str) -> list:
    """Cross-encoder re-scoring; keeps the best RERANK_TOP_N. No-op when disabled."""
    if reranker is None:
        return nodes
    return reranker.postprocess_nodes(nodes, query_str=query)


def _top_score(nodes: list) -> Optional[float]:
    return float(nodes[0].score) if nodes and nodes[0].score is not None else None


# =============================================================================
# 9. ROUTING (hybrid mode): documents or the LLM's own knowledge?
# =============================================================================
def documents_are_relevant(message: str) -> bool:
    # Routing by re-rank score (sharper than cosine)
    if reranker is not None and RERANK_MIN_SCORE > 0:
        top = _top_score(rerank(retrieve(message, RETRIEVE_K), message))
        log.info("top rerank score %s (RERANK_MIN_SCORE %s) for %r", top, RERANK_MIN_SCORE, message)
        return top is not None and top >= RERANK_MIN_SCORE

    # Routing by cosine score
    if MIN_SCORE <= 0:
        return True  # routing off: always retrieve, let the prompt decide
    top = _top_score(retrieve(message, 1))
    log.info("top retrieval score %s (MIN_SCORE %s)", top, MIN_SCORE)
    return top is not None and top >= MIN_SCORE


# =============================================================================
# 10. DEBUG OUTPUT
# =============================================================================
class PrintNodes(BaseNodePostprocessor):
    """Prints the chunks passing through this step (as soon as the question is asked,
    before the LLM answers), then returns them unchanged."""

    label: str = ""

    @classmethod
    def class_name(cls) -> str:
        return "PrintNodes"

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        bar = "-" * 72
        lines = [f"\n{bar}"]
        if query_bundle is not None:
            lines.append(f"SEARCH QUERY: {query_bundle.query_str}")
        lines.append(f"{self.label}: {len(nodes)} chunk(s)")
        for i, n in enumerate(nodes, 1):
            score = "n/a" if n.score is None else f"{float(n.score):.3f}"
            text = " ".join(n.node.get_content().split())
            if len(text) > 300:
                text = text[:300] + "..."
            lines.append(f"  [{i}] score={score}  doc={n.node.ref_doc_id}")
            lines.append(f"      {text}")
        lines.append(bar)
        print("\n".join(lines), flush=True)
        return nodes


def print_trace(session_id: str, message: str, route: str, nodes: list, answer: str) -> None:
    """Terminal trace: question -> route -> context -> answer."""
    if not DEBUG_RAG:
        return
    bar = "=" * 72
    lines = [
        f"\n{bar}",
        f"SESSION : {session_id}",
        f"CONFIG  : mode={ANSWER_MODE} chat_mode={CHAT_MODE} model={OLLAMA_MODEL}",
        f"QUESTION: {message}",
        f"ROUTE   : {route}",
    ]
    if nodes:
        lines.append(f"CONTEXT : {len(nodes)} chunk(s) sent to the LLM (listed above)")
    else:
        lines.append("CONTEXT : (none)")
    lines.append(f"ANSWER  : {answer}")
    lines.append(bar)
    print("\n".join(lines), flush=True)


# =============================================================================
# 11. ENGINES AND SESSIONS (prompt assembly lives here)
# =============================================================================
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


def build_postprocessors() -> list:
    """Steps applied to the retrieved chunks, in order: [print] -> rerank -> [print]."""
    steps: List[Any] = []
    if DEBUG_RAG:
        steps.append(PrintNodes(
            label="RETRIEVED by vector search" if reranker is not None
            else "RETRIEVED -> sent to the LLM"))
    if reranker is not None:
        steps.append(reranker)
        if DEBUG_RAG:
            steps.append(PrintNodes(label="KEPT after re-ranking -> sent to the LLM"))
    return steps


def build_rag_engine(memory: ChatMemoryBuffer):
    """Engine that answers from the documents."""
    kwargs: Dict[str, Any] = {}
    steps = build_postprocessors()
    if steps:
        kwargs["node_postprocessors"] = steps
    # replace LlamaIndex's built-in wrapper around the chunks (see CONTEXT_PROMPT above)
    if CHAT_MODE == "condense_plus_context":
        kwargs["context_prompt"] = CONTEXT_PROMPT
    elif CHAT_MODE == "context":
        kwargs["context_template"] = CONTEXT_PROMPT
    return index.as_chat_engine(
        chat_mode=CHAT_MODE,
        memory=memory,
        system_prompt=SYSTEM_PROMPT,
        # with re-ranking: fetch wide first, the reranker keeps the best few
        similarity_top_k=RETRIEVE_K if reranker is not None else TOP_K,
        **kwargs,
    )


def build_general_engine(memory: ChatMemoryBuffer):
    """Plain LLM engine: no documents, the model's own knowledge."""
    return SimpleChatEngine.from_defaults(
        llm=Settings.llm, memory=memory, system_prompt=GENERAL_PROMPT
    )


def get_session(session_id: str) -> Dict[str, Any]:
    """Per session: one shared memory, a document-grounded engine and a plain LLM engine."""
    if session_id not in _sessions:
        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        _sessions[session_id] = {
            "rag": build_rag_engine(memory),
            "general": build_general_engine(memory),
        }
    return _sessions[session_id]


def reset_session(session_id: str) -> None:
    _sessions.pop(session_id, None)


# =============================================================================
# 12. CHAT
# =============================================================================
def format_sources(nodes: list) -> List[Dict[str, Any]]:
    return [
        {
            "doc_id": n.node.ref_doc_id,
            # re-rank score when RERANK_ENABLED=1, otherwise cosine.
            # The reranker returns numpy.float32, which FastAPI cannot serialize -> cast to float.
            "score": float(n.score) if n.score is not None else None,
            "snippet": n.node.get_content()[:200],
            "metadata": n.node.metadata,
        }
        for n in nodes
    ]


def answer_general(session_id: str, session: Dict[str, Any], message: str, docs_exist: bool) -> Dict[str, Any]:
    """No documents, or nothing relevant: the LLM answers from its own knowledge."""
    answer = str(session["general"].chat(message))
    if docs_exist:  # documents exist but were judged not relevant: say so (in code, not in the prompt)
        answer = "Not found in your documents. " + answer
    reason = "documents judged not relevant" if docs_exist else "no documents ingested"
    print_trace(session_id, message, f"general LLM, no documents ({reason})", [], answer)
    return {"answer": answer, "sources": [], "grounded": False}


def answer_with_documents(session_id: str, session: Dict[str, Any], message: str) -> Dict[str, Any]:
    """Retrieve + rerank + generate, all inside the RAG engine."""
    response = session["rag"].chat(message)
    answer = str(response)
    print_trace(session_id, message, "documents (RAG)", response.source_nodes, answer)
    return {"answer": answer, "sources": format_sources(response.source_nodes), "grounded": True}


def chat(session_id: str, message: str) -> Dict[str, Any]:
    docs_exist = has_documents()
    if not docs_exist and ANSWER_MODE == "strict":
        raise EmptyIndexError("No documents ingested yet.")

    session = get_session(session_id)
    use_docs = docs_exist and (ANSWER_MODE == "strict" or documents_are_relevant(message))

    if use_docs:
        return answer_with_documents(session_id, session, message)
    return answer_general(session_id, session, message, docs_exist)


# Backward-compatible names (the previous version used leading underscores)
_get_session = get_session
_documents_are_relevant = documents_are_relevant
_print_trace = print_trace