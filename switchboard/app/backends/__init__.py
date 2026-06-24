from switchboard.app.backends.base import AgentAdapter
from switchboard.app.backends.cli_agents import ClaudeCodeCliAdapter, CodexCliAdapter
from switchboard.app.backends.ollama_backend import OllamaAdapter
from switchboard.app.backends.registry import BackendRegistry

__all__ = [
    "AgentAdapter",
    "BackendRegistry",
    "ClaudeCodeCliAdapter",
    "CodexCliAdapter",
    "OllamaAdapter",
]
