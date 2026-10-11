from langsmith import traceable

from app.core.logging_config import setup_logging
from app.core.config import Settings
from app.llm.gateway import infer

logger = setup_logging(__name__)

class SummarizerAgent:
    """
        Condenses raw search results into structured bullet points.
    """
    def __init__(self, config: Settings):
        self.config = config

    @traceable(
        run_type="tool", 
        name="agent:summarizer"
    )
    async def run(self, 
                  search_results: list[str]) -> str:
        logger.info(
            "SummarizerAgent:  condensing %d search result(s)", 
            len(search_results)
        )
        combined = "\n\n".join(search_results)
        return await infer(
            self.config,
            "research_summarize",
            f"Summarize these research findings into clear, structured bullet points:\n\n{combined}",
        )
