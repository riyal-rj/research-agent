from typing import Any

from langsmith import traceable
from langgraph.graph import END

from app.agents.critic_agent import CriticAgent
from app.agents.summarizer_agent import SummarizerAgent
from app.agents.writer_agent import WriterAgent
from app.agents.search_agent import SearchAgent

from app.agents.agent_state import ResearchState

from app.core.config import Settings
from app.core.logging_config import setup_logging

logger = setup_logging(__name__)

class OrchestratorAgent:
    """
        Cordinates all the subagent. Passes session history to SearchAgent and LTM Context
        to Writer Agent so the pipeline is genuinely context aware. If the critic rejects
        the reports it loops back for another pass (up to agent_max_iterations times).

        State keys have no reducers, so every node replaces the value it returns:
        a retry pass starts from fresh search results and summaries.
    """

    def __init__(self,
                 config: Settings):
        self.config = config
        self.search_agent = SearchAgent(config)
        self.summarizer_agent = SummarizerAgent(config)
        self.writer_agent = WriterAgent(config)
        self.critic_agent = CriticAgent(config)


    @traceable(
        run_type="chain", 
        name="orchestrator:search"
    )
    async def search_node(self,
                          state: ResearchState) -> dict[str, Any]:
        result = await self.search_agent.run(state["topic"],
                                             state.get("session_history",[]))
        return {"search_results": [result]}

    @traceable(
        run_type="chain", 
        name="orchestrator:summarize"
    )
    async def summarize_node(self,
                             state: ResearchState) -> dict[str, Any]:
        result = await self.summarizer_agent.run(state["search_results"])
        return {"summaries": [result]}


    @traceable(
        run_type="chain", 
        name="orchestrator:write"
    )
    async def write_node(self,
                         state: ResearchState) -> dict[str, Any]:
        report = await self.writer_agent.run(
            state["topic"],
            state["summaries"],
            state.get("ltm_context","")
        )
        return {
            "report": [report],
            "iterations": state.get("iterations",0)+1
        }

    @traceable(
        run_type="chain", 
        name="orchestrator:verify"
    )
    async def verify_node(self,
                          state: ResearchState) -> dict[str, Any]:
        verified = await self.critic_agent.run(state["report"])
        return {
            "verified": [verified]
        }


    def route(self,
              state: ResearchState) -> str:
        iterations = state.get("iterations",0)
        if state["verified"]:
            return END
        if iterations < self.config.agent_max_iterations:
            logger.info(
                "Critic rejected report, retrying (pass %d of %d)",
                iterations+1, self.config.agent_max_iterations
            )
            return "search"
        logger.warning(
            "Critic rejected report after %d pass(es); returning it unverified",
            iterations
        )
        return END
    

