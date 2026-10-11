from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.agent_state import ResearchState
from app.agents.orchestrator_agent import OrchestratorAgent
from app.core.config import Settings, settings_inst

_STEPS_PER_PASS = 4

def build_graph(config: Settings | None = None) -> CompiledStateGraph:
    if config is None:
        config = settings_inst

    orchestrator = OrchestratorAgent(config)
    workflow = StateGraph(ResearchState)

    workflow.add_node("search", orchestrator.search_node)
    workflow.add_node("summarize", orchestrator.summarize_node)
    workflow.add_node("write", orchestrator.write_node)
    workflow.add_node("verify", orchestrator.verify_node)

    workflow.add_edge(START, "search",)
    workflow.add_edge("search", "summarize")
    workflow.add_edge("summarize", "write")
    workflow.add_edge("write", "verify")
    workflow.add_conditional_edges(
        "verify",
        orchestrator.route,
        {"search":"search", END : END}
    )
    return workflow.compile().with_config(
        recursion_limit=_STEPS_PER_PASS*config.agent_max_iterations
    )