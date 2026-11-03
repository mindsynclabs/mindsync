import os
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from src.modules.agents.deps import AgentDependencies
from src.modules.agents.tools.rag_tool import search_knowledge_base

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")

# Configure provider with explicit base_url
provider = OllamaProvider(base_url=OLLAMA_BASE_URL)
model = OllamaModel(model_name=LLM_MODEL, provider=provider)

knowledge_agent = Agent(
    model=model,
    deps_type=AgentDependencies,
    system_prompt=(
        "You are MindSync's Knowledge Agent. Your primary objective is to answer user queries "
        "by retrieving and synthesizing accurate information from the local vector database. "
        "Always rely on the search tool provided when internal context or documentation is needed."
    )
)

knowledge_agent.tool(search_knowledge_base)