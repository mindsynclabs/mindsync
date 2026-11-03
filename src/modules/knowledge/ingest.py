import os
import httpx
from typing import List, Dict, Any
from src.modules.knowledge.store import vector_store, DocumentChunk

# Ollama embedding endpoint configuration (local-first)
OLLAMA_HOST = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

class DocumentIngester:
    """Handles text splitting, embedding generation, and insertion into LanceDB."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """Simple sliding-window text chunker."""
        words = text.split()
        chunks = []
        
        if len(words) <= self.chunk_size:
            return [text]
            
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            if i + self.chunk_size >= len(words):
                break
                
        return chunks

    async def get_embedding(self, text: str) -> List[float]:
        """Generate vector embedding using local Ollama instance."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OLLAMA_HOST}/api/embeddings",
                json={
                    "model": EMBEDDING_MODEL,
                    "prompt": text
                }
            )
            response.raise_for_status()
            data = response.json()
            return data.get("embedding", [])

    async def ingest_document(self, doc_id: str, text: str, metadata: Dict[str, Any] = None) -> int:
        """Chunk a document, generate embeddings, and store them in LanceDB."""
        if metadata is None:
            metadata = {}

        chunks = self.split_text(text)
        table = vector_store.get_or_create_table()
        
        records = []
        for idx, chunk in enumerate(chunks):
            vector = await self.get_embedding(chunk)
            if not vector:
                continue
                
            records.append({
                "id": f"{doc_id}-chunk-{idx}",
                "text": chunk,
                "vector": vector,
                "metadata": str(metadata)
            })
            
        if records:
            table.add(records)
            
        return len(records)

# Initialize a shared ingester instance
document_ingester = DocumentIngester()