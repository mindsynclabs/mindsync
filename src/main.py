from fastapi import FastAPI

app = FastAPI(
    title="MindSync Engine",
    description="Enterprise Open-Source AI Platform (Local-First)",
    version="0.1.0",
)

@app.get("/")
async def root():
    return {"message": "MindSync Engine is running smoothly!"}