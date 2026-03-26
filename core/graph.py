# core/graph.py
# ─────────────────────────────────────────────────────────────────────────────
# Defines the LangGraph StateGraph that orchestrates all three agents.
#
# The graph works like a pipeline with decision points:
#
#   [START]
#     │
#     ▼
#   [retrieve_literature]  ← Literature Retrieval Agent
#     │
#     ▼ (check for errors)
#   [analyze_gaps]         ← Analysis Agent (uses local Mistral model)
#     │
#     ▼ (check for errors)
#   [plan_experiment]      ← Planning Agent (uses cloud GPT-4o model)
#     │
#     ▼
#   [END]
#
# Between each node, a conditional edge checks if the previous agent
# encountered an error. If it did, the graph routes to END early with
# a meaningful error message rather than crashing silently.
# ─────────────────────────────────────────────────────────────────────────────

from langgraph.graph import StateGraph, END
from core.state import AgentState


def should_continue_after_retrieval(state: AgentState) -> str:
    """
    Conditional edge function called after the retrieval agent completes.
    Returns the name of the next node to execute, or END if there was an error.

    LangGraph calls this function with the current state and uses the
    returned string to decide which node to route to next.
    """
    if state.get("error_message"):
        # If the retrieval agent set an error, stop the pipeline gracefully
        return END
    if not state.get("fetched_papers"):
        # If no papers were found, there's nothing to analyze
        return END
    # Everything looks good — proceed to the Analysis Agent
    return "analyze_gaps"


def should_continue_after_analysis(state: AgentState) -> str:
    """
    Conditional edge function called after the Analysis Agent completes.
    Routes to Planning Agent if analysis succeeded, or END if it failed.
    """
    if state.get("error_message"):
        return END
    if not state.get("hypothesis"):
        # If no hypothesis was generated, we can't produce an experimental plan
        return END
    return "plan_experiment"


def build_graph() -> StateGraph:
    """
    Constructs and compiles the full LangGraph StateGraph.

    This function:
        1. Creates a new StateGraph with AgentState as the shared state schema
        2. Adds each agent as a named node
        3. Connects nodes with edges (direct or conditional)
        4. Compiles the graph into an executable runnable

    Returns:
        A compiled LangGraph application ready to invoke with an initial state.
    """

    # ── Import agent node functions ───────────────────────────────────────
    # We import here (inside the function) rather than at the top of the file
    # to avoid circular imports — agents import from core, and core shouldn't
    # import from agents at module load time.
    from agents.retrieval_agent import run_retrieval_agent
    from agents.analysis_agent import run_analysis_agent
    from agents.planning_agent import run_planning_agent

    # ── Create the graph with our shared state schema ─────────────────────
    # Passing AgentState tells LangGraph what shape the state object is,
    # enabling proper type checking and state management throughout execution.
    graph = StateGraph(AgentState)

    # ── Add agent nodes ───────────────────────────────────────────────────
    # Each node is a named entry point that maps to a Python function.
    # When the graph reaches a node, it calls that function with the
    # current state and merges the returned dict back into the state.
    graph.add_node("retrieve_literature", run_retrieval_agent)
    graph.add_node("analyze_gaps", run_analysis_agent)
    graph.add_node("plan_experiment", run_planning_agent)

    # ── Define the entry point ────────────────────────────────────────────
    # This tells LangGraph which node to execute first when the graph is invoked.
    graph.set_entry_point("retrieve_literature")

    # ── Add conditional edges ─────────────────────────────────────────────
    # Conditional edges call a function with the current state and route
    # to different nodes based on the returned string.
    # This is how we implement error handling and flow control.
    graph.add_conditional_edges(
        "retrieve_literature",          # From this node...
        should_continue_after_retrieval,  # ...call this function to decide...
        {                               # ...and map return values to node names
            "analyze_gaps": "analyze_gaps",
            END: END
        }
    )

    graph.add_conditional_edges(
        "analyze_gaps",
        should_continue_after_analysis,
        {
            "plan_experiment": "plan_experiment",
            END: END
        }
    )

    # ── Add final direct edge ─────────────────────────────────────────────
    # After the Planning Agent finishes, the pipeline always ends.
    # No condition needed here — planning is always the terminal step.
    graph.add_edge("plan_experiment", END)

    # ── Compile and return the executable graph ───────────────────────────
    # compile() validates the graph structure (checks for disconnected nodes,
    # missing entry points, etc.) and returns an executable Runnable object
    # that can be invoked with graph.invoke(initial_state).
    compiled_graph = graph.compile()

    return compiled_graph


# ── Convenience function for getting an initialized empty state ───────────────
def get_initial_state(research_topic: str) -> AgentState:
    """
    Returns a properly initialized AgentState for a new pipeline run.

    Every field is set to a safe empty/default value so that agents can
    safely read any field without worrying about KeyError or None surprises.

    Args:
        research_topic: The research domain the user wants to explore.

    Returns:
        A fully initialized AgentState dictionary ready to pass to the graph.
    """
    return AgentState(
        research_topic=research_topic,
        fetched_papers=[],
        indexed_paper_ids=[],
        retrieval_context={},
        identified_gaps=[],
        selected_gap=None,
        hypothesis=None,
        experimental_plan=None,
        current_stage="initializing",
        error_message=None,
        is_complete=False
    )