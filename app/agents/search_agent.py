from typing import Any
from langsmith import traceable

from app.core.config import Settings
from app.llm.gateway import infer
from app.core.logging_config import setup_logging

logger = setup_logging(__name__)

_HISTORY_TURNS = 4

class SearchAgent:
    """
        Finds key facts. Recieves session history so
        it understands what the user has asked before.
    """

    def __init__(self, 
                 config: Settings):
        self.config = config

    @traceable(
            run_type="tool",
            name="agent:search"
    )
    async def run(self,
                  topic : str,
                  session_history : list[dict[str, Any]]) -> str:
        logger.info(
            "SearchAgent: researching %r (%d history turns)",
            topic[:200],
            len(session_history)
        )
        history_ctx = ""
        if session_history:
            recent = session_history[-_HISTORY_TURNS:]
            history_ctx="""\n\n Previous conversation context (use this to understand
                         what the user already knows and what angle they care about):\n """
            history_ctx+="\n".join(
                f"{str(m.get('role','user')).upper()}: {m.get('content','')}"
                for m in recent
            )

        return await infer(
            self.config,
            "research_summarize",
            f"You are a research specialist. Find and list 5 key facts, recent developments,"
            f"and important details about: {topic}. Be through and specific."
            f"{history_ctx}"
        )