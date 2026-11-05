import os
from fastapi import FastAPI

# Import the routers from your endpoints package
from src.api.endpoints.chat import router as chat_router
from src.api.endpoints.knowledge import router as knowledge_router
# from src.api.endpoints.agent import router as agent_router

app = FastAPI(
    title="MindSync Backend",
    version="1.0.0",
    description="Local RAG & AI Agent System"
)

# Register endpoints
# Leave prefix empty if chat.py already defines @router.post("/chat")
app.include_router(chat_router, tags=["Chat & RAG"])
app.include_router(knowledge_router, tags=["Knowledge Base"])
# app.include_router(agent_router, tags=["Agent"])

@app.get("/")
async def root():
    return {"message": "MindSync API is running"}