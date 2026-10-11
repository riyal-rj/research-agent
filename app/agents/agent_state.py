from typing import TypedDict

class ResearchState(TypedDict):
    topic : str
    session_id : str
    session_history : list[dict]
    ltm_context : str
    search_results : list[str]
    summaries : list[str]
    report : str
    verified : bool
    error: str
    iterations: int

def inittail_state(topic : str,
                   session_id : str,
                   session_history : list[dict],
                   ltm_context : str) -> ResearchState:
    return ResearchState(
        topic = topic,
        session_id = session_id,
        session_history = session_history,
        ltm_context = ltm_context,
        search_results = [],
        summaries = [],
        report = "",
        verified = False,
        error = "",
        iterations = 0
    )