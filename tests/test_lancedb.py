import asyncio
from src.modules.knowledge.retriever import knowledge_retriever

async def test_search():
    query = "Core Architectural Pillars"
    print(f"🔍 Testing direct LanceDB search for: '{query}'...")
    
    # Time the vector search
    import time
    start_time = time.time()
    
    results = await knowledge_retriever.search(query=query, limit=3)
    
    elapsed = time.time() - start_time
    print(f"⏱️ LanceDB search completed in {elapsed:.4f} seconds.")
    print(f"📦 Found {len(results)} results:")
    
    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} (ID: {r.get('id')}) ---")
        print(r.get('text'))

if __name__ == "__main__":
    asyncio.run(test_search())