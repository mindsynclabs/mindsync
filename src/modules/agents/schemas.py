from pydantic import BaseModel, Field

class AgentPromptRequest(BaseModel):
    prompt: str = Field(..., description="User prompt or question for the agent")

class AgentResponse(BaseModel):
    status: str = "success"
    prompt: str
    response: str