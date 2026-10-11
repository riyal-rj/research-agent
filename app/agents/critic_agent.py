import re

from langsmith import traceable

from app.core.config import Settings
from app.llm.gateway import infer
from app.core.logging_config import setup_logging

logger = setup_logging(__name__)

_VERDICT = re.compile(
    r"^[\W_]*(?:(?:verdict|answer|result|decision)\s*[:\--]\s*)?[\W_]*(YES|NO)\b[\W_]*(.*)",
    re.IGNORECASE | re.DOTALL,
)                            


def parse_verdict(reply : str) -> tuple[bool | None, str]:
    """
        Return (passed, reason); passed is None when the reply has no leading YES/NO.
    """
    match = _VERDICT.match(reply.strip())
    if match is None:
        return None, reply.strip()
    return match.group(1).upper() == "YES", match.group(2).strip()

class CriticAgent:
    """
        Verifies the factual consistency and logical coherence of the report.
    """

    def __init__(self,
                 config: Settings):
        self.config = config

    @traceable(
        run_type="tool",
        name="agent:critic"
    )
    async def run(self,
                  report : str) -> bool:
        logger.info(
            "CriticAgent : verifying report (%d chars)",
            len(report)
        )
        check = await infer(
            self.config,
            "research_summarize",
            f"Review this report for factual consistency and logical coherence."
            f"Reply with YES if it passes or NO with a brief reason:\n\n"
            f"{report[:self.config.agent_report_truncate]}"
        )
        passed, reason = parse_verdict(check)
        if passed is None:
            logger.warning(
                "CriticAgent: no YES/NO verdict, treating as rejected: %.200s",
                reason
            )
            return False
        if passed:
            logger.info(
                "CriticAgent: report passed"
            )
        else:
            logger.info(
                "CriticAgent: report rejected: %.300s",
                reason or "(no reason given)"
            )
        return passed