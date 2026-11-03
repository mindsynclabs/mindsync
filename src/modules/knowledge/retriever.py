import os
import httpx
from typing import List, Dict, Any
from src.modules.knowledge.store import vector_store

# Ollama configuration for query embedding generation
OLLAMA_HOST = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

class KnowledgeRetriever:
    """Handles semantic similarity search against the local LanceDB vector store."""

    async def _get_query_embedding(self, query: str) -> List[float]:
        """Generate embedding vector for the search query using local Ollama."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{OLLAMA_HOST}/api/embeddings",
                json={
                    "model": EMBEDDING_MODEL,
                    "prompt": query
                }
            )
            response.raise_for_status()
            data = response.json()
            return data.get("embedding", [])

    async def search(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Perform vector similarity search and return top matching document chunks."""
        # 1. Generate embedding for the incoming search query
        query_vector = await self._get_query_embedding(query)
        if not query_vector:
            return []

        # 2. Open the vector table and run nearest neighbor search
        table = vector_store.get_or_create_table()
        
        # LanceDB vector search query execution
        results = (
            table.search(query_vector)
            .limit(limit)
            .to_list()
        )

        formatted_results = []
        for row in results:
            formatted_results.append({
                "id": row.get("id"),
                "text": row.get("text"),
                "metadata": row.get("metadata"),
                "_distance": row.get("_distance") # Distance metric (lower means higher similarity)
            })

        return formatted_results

# Initialize a shared retriever instance
knowledge_retriever = KnowledgeRetriever()