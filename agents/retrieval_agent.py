# agents/retrieval_agent.py
# ─────────────────────────────────────────────────────────────────────────────
# Literature Retrieval Agent — the first node in the LangGraph pipeline.
#
# Responsibilities:
#   1. Fetch real research papers from ArXiv based on the research topic
#   2. Embed and index those papers into ChromaDB (the vector memory store)
#   3. Run four targeted semantic queries to pre-build retrieval context
#   4. Update the shared AgentState with all results for downstream agents
#
# This agent satisfies the following project requirements:
#   ✅ Tool Usage          — calls ArXiv API via arxiv_tool
#   ✅ RAG                 — embeds papers and queries ChromaDB semantically
#   ✅ Memory Integration  — persists embeddings to ChromaDB for reuse
#   ✅ Agent Orchestration — operates as a named node in the LangGraph graph
# ─────────────────────────────────────────────────────────────────────────────

from typing import Dict, Any
from core.state import AgentState, ResearchPaper
from core.memory import memory_store
from core.config import Config
from tools.arxiv_tool import search_papers


def run_retrieval_agent(state: AgentState) -> Dict[str, Any]:
    """
    Main entry point for the Literature Retrieval Agent.
    Called automatically by LangGraph when the graph reaches the
    'retrieve_literature' node.

    LangGraph passes the current AgentState to this function.
    This function must return a dictionary containing only the fields
    it wants to update in the shared state — LangGraph merges these
    updates back into the master state automatically before passing
    it to the next agent.

    Args:
        state: The current shared AgentState containing at minimum
               the 'research_topic' field set by the user.

    Returns:
        A dict with updated state fields:
        - fetched_papers: list of paper dicts from ArXiv
        - indexed_paper_ids: list of paper IDs successfully stored in ChromaDB
        - retrieval_context: dict of semantic query results from ChromaDB
        - current_stage: updated to "analyzing" to signal next agent
        - error_message: set only if something goes wrong
    """

    print("\n" + "="*60)
    print("🤖 LITERATURE RETRIEVAL AGENT — Starting")
    print("="*60)

    # ── Step 1: Extract the research topic from shared state ──────────────
    # The research_topic is set by the user via the Gradio interface before
    # the graph is invoked. We validate it here before doing any work.
    research_topic = state.get("research_topic", "").strip()

    if not research_topic:
        # If no topic was provided, we cannot proceed — signal the error
        # clearly so the graph's conditional edge routes to END gracefully
        print("❌ No research topic found in state. Cannot proceed.")
        return {
            "error_message": "Research topic is empty. Please provide a valid topic.",
            "current_stage": "error"
        }

    print(f"📌 Research Topic: '{research_topic}'")
    print(f"📄 Max papers to fetch: {Config.MAX_PAPERS_TO_FETCH}")

    # ── Step 2: Fetch papers from ArXiv ───────────────────────────────────
    # We use sort_by_recent=False here (relevance sort) to ensure the papers
    # returned are genuinely about the research topic rather than just the
    # most recently submitted papers that loosely match keywords.
    # For a project like this, topical relevance matters more than recency.
    print("\n📡 Fetching papers from ArXiv...")

    try:
        raw_papers = search_papers(
            research_topic=research_topic,
            max_results=Config.MAX_PAPERS_TO_FETCH
        )
    except Exception as e:
        # If the ArXiv API call fails entirely (network error, timeout, etc.)
        # we return an error state rather than crashing the pipeline
        error_msg = f"ArXiv API fetch failed: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "error_message": error_msg,
            "current_stage": "error"
        }

    if not raw_papers:
        print("❌ No papers returned from ArXiv.")
        return {
            "error_message": f"No papers found for topic: '{research_topic}'. "
                             "Try a broader research topic.",
            "current_stage": "error"
        }

    print(f"✅ Successfully fetched {len(raw_papers)} papers from ArXiv.")

    # ── Step 3: Convert raw dicts to typed ResearchPaper objects ──────────
    # This conversion ensures the data stored in state matches the
    # ResearchPaper TypedDict schema we defined in core/state.py.
    # Type consistency here prevents subtle bugs in downstream agents
    # that expect specific field names and types.
    fetched_papers: list[ResearchPaper] = []

    for paper in raw_papers:
        typed_paper = ResearchPaper(
            paper_id=paper.get("paper_id", ""),
            title=paper.get("title", ""),
            authors=paper.get("authors", []),
            abstract=paper.get("abstract", ""),
            published=paper.get("published", ""),
            url=paper.get("url", "")
        )
        fetched_papers.append(typed_paper)

    # ── Step 4: Index papers into ChromaDB ────────────────────────────────
    # We pass the raw paper dicts (not the TypedDict versions) to the
    # memory store because it expects plain dicts with known keys.
    # store_papers() handles chunking, embedding, and ChromaDB insertion.
    print(f"\n💾 Indexing {len(fetched_papers)} papers into ChromaDB...")

    try:
        indexed_ids = memory_store.store_papers(raw_papers)
    except Exception as e:
        error_msg = f"ChromaDB indexing failed: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "fetched_papers": fetched_papers,
            "error_message": error_msg,
            "current_stage": "error"
        }

    if not indexed_ids:
        print("⚠️  No papers were successfully indexed into ChromaDB.")
        return {
            "fetched_papers": fetched_papers,
            "error_message": "Paper indexing failed — no embeddings were stored.",
            "current_stage": "error"
        }

    print(f"✅ Successfully indexed {len(indexed_ids)} papers into ChromaDB.")

    # ── Step 5: Run targeted semantic queries to build retrieval context ──
    # This is the RAG step. We run four different semantic queries against
    # the newly indexed knowledge base, each designed to extract a specific
    # type of information that the Analysis Agent will need.
    #
    # Why four separate queries instead of one broad query?
    # Different questions surface different passages. A query about
    # "limitations" will retrieve passages from limitations/future work
    # sections. A query about "datasets" will retrieve methodology sections.
    # Running targeted queries gives the Analysis Agent a rich, multi-faceted
    # view of the literature rather than a single general retrieval.
    print("\n🔎 Running semantic queries to build retrieval context...")

    # These four query strings are carefully chosen to cover the four
    # most important dimensions of understanding a research field:
    # what methods exist, what data was used, what failed, what metrics matter
    semantic_queries = {
        "methodologies": (
            f"what machine learning methods, models, and architectures "
            f"are used for {research_topic}?"
        ),
        "datasets": (
            f"what datasets, benchmarks, and data sources are used "
            f"in {research_topic} research?"
        ),
        "limitations": (
            f"what are the limitations, weaknesses, shortcomings, "
            f"and challenges in current {research_topic} research?"
        ),
        "evaluation_metrics": (
            f"what evaluation metrics, performance measures, and "
            f"benchmarks are reported in {research_topic} studies?"
        )
    }

    try:
        retrieval_context = memory_store.query_multiple(semantic_queries)
    except Exception as e:
        # If semantic queries fail, we still have the papers — we can
        # continue with empty context rather than failing the pipeline
        print(f"⚠️  Semantic query failed: {e}. Continuing with empty context.")
        retrieval_context = {key: "Query failed." for key in semantic_queries}

    # Print a summary of what was retrieved for each query
    print("\n📊 Retrieval Context Summary:")
    for context_key, context_text in retrieval_context.items():
        preview = context_text[:120].replace('\n', ' ')
        print(f"   [{context_key}]: {preview}...")

    # ── Step 6: Return state updates ──────────────────────────────────────
    # We return only the fields this agent is responsible for updating.
    # LangGraph will merge these into the master AgentState automatically.
    # Setting current_stage to "analyzing" signals that retrieval is complete
    # and the pipeline is ready for the Analysis Agent to take over.
    print("\n✅ Literature Retrieval Agent complete.")
    print(f"   Papers fetched     : {len(fetched_papers)}")
    print(f"   Papers indexed     : {len(indexed_ids)}")
    print(f"   Context categories : {list(retrieval_context.keys())}")
    print("="*60)

    return {
        "fetched_papers": fetched_papers,
        "indexed_paper_ids": indexed_ids,
        "retrieval_context": retrieval_context,
        "current_stage": "analyzing",
        "error_message": None  # Explicitly clear any previous error
    }