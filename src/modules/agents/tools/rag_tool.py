from pydantic_ai import RunContext
from src.modules.knowledge.retriever import knowledge_retriever

async def search_knowledge_base(ctx: RunContext, query: str) -> str:
    """Search the local vector database for relevant information based on the user query."""
    results = await knowledge_retriever.search(query=query, limit=3)
    
    if not results:
        return "No relevant information found in the local knowledge base."
    
    context_parts = []
    for r in results:
        context_parts.append(f"- [ID: {r['id']}] {r['text']}")
        
    return "\n".join(context_parts)