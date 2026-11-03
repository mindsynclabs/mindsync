from fastapi import APIRouter, HTTPException
from src.modules.agents.schemas import AgentPromptRequest, AgentResponse
from src.modules.agents.core_agents.router_agent import router_agent
from src.modules.agents.deps import AgentDependencies

router = APIRouter(prefix="/agent", tags=["AI Agent (PydanticAI)"])

@router.post("/chat", response_model=AgentResponse, summary="Interact with the MindSync Multi-Agent router")
async def chat_endpoint(payload: AgentPromptRequest):
    """Entry point for agent queries. Router dynamically delegates tasks to specialized sub-agents."""
    try:
        deps = AgentDependencies()
        result = await router_agent.run(payload.prompt, deps=deps)
        
        return AgentResponse(
            status="success",
            prompt=payload.prompt,
            response=result.output  # Updated to V2 .output property
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))