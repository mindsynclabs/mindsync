import os
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from src.modules.agents.deps import AgentDependencies
from src.modules.agents.core_agents.knowledge_agent import knowledge_agent

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
# Bạn có thể thiết lập tham số runtime qua OllamaProvider hoặc biến môi trường Ollama
os.environ["OLLAMA_NUM_PARALLEL"] = "2"

provider = OllamaProvider(base_url=OLLAMA_BASE_URL)
model = OllamaModel(model_name=LLM_MODEL, provider=provider)

router_agent = Agent(
    model=model,
    deps_type=AgentDependencies,
    system_prompt=(
        "You are MindSync Router. ALWAYS use the `delegate_to_knowledge_agent` tool "
        "if the query asks about documentation, features, or architectural concepts. "
        "Do not ask for clarification—search first."
    ),
)


@router_agent.tool
async def delegate_to_knowledge_agent(ctx: RunContext, prompt: str) -> str:
    """Delegates query execution to the specialized Knowledge Agent."""
    print("🤖 [Router Agent] Decided to delegate query to -> Knowledge Agent")
    result = await knowledge_agent.run(prompt, deps=ctx.deps)
    print("🤖 [Knowledge Agent] Finished synthesis. Returning to Router...")
    return result.output  # Updated to V2 .output property
