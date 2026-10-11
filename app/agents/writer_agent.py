from langsmith import traceable

from app.core.config import Settings
from app.core.logging_config import setup_logging
from app.llm.gateway import infer

logger = setup_logging(__name__)

_LTM_CONTEXT_CHARS = 200

class WriterAgent:
    """
        Produces the final structured report. If a related prev report exists in
        LTM, it uses it as reference so the new report builds on the existing knowledge
        rather than starting from scratch.
    """

    def __init__(self,
                 config: Settings):
        self.config = config

    async def run(self,
                  topic : str,
                  summaries: list[str],
                  ltm_context: str) -> str:
        logger.info(
            "WriterAgent: drafting report (%d summaries, LTM context: %s)",
            len(summaries),
            "yes" if ltm_context else "no"
        )
        combined="\n\n".join(summaries)

        ltm_section = ""
        if ltm_context:
            ltm_section =(
                f"\n\nPREVIOUS RESEARCH ON A RELATED TOPIC (use this as reference -"
                f"build on it, correct outdated information, and highlight what has changed):\n"
                f"{ltm_context[:_LTM_CONTEXT_CHARS]}"
            )
        return await infer(
            self.config,
            "report_write",
            f"Write a comprehensive, well-structured research report on: '{topic}'\n\n"
            f"Current research findings:\n{combined}"
            f"{ltm_section}\n\n"
            f"Include: Executive Summary, Key Findings, Analysis, and Conclusion.", 
        )




   









           