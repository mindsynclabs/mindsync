from fastapi import FastAPI
from src.api.endpoints.knowledge import router as knowledge_router
from src.api.endpoints.agent import router as agent_router

app = FastAPI(
    title="MindSync Engine",
    description="Enterprise Open-Source AI Platform (Local-First)",
    version="0.1.0",
)

# Register all domain routers
app.include_router(knowledge_router)
app.include_router(agent_router)

@app.get("/", summary="Health Check")
async def root():
    return {
        "message": "MindSync Engine is running smoothly!",
        "architecture": "Modular Monolith (Local-First)"
    }