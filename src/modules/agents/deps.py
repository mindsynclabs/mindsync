from dataclasses import dataclass

@dataclass
class AgentDependencies:
    """Container for runtime dependencies passed to PydanticAI agents."""
    user_id: str = "default_user"
    session_id: str = "default_session"