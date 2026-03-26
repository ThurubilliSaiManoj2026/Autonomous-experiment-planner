# core/config.py
# ─────────────────────────────────────────────────────────────────────────────
# Centralized configuration loader for the entire project.
#
# Every agent, tool, and component imports settings from here using:
#     from core.config import Config
#
# All sensitive values (API keys) are loaded from the .env file via
# python-dotenv, which means they are never hardcoded in source files
# and are protected from accidental GitHub commits by .gitignore.
# ─────────────────────────────────────────────────────────────────────────────

import os
from dotenv import load_dotenv

# load_dotenv() reads every key=value pair from the .env file in your
# project root and injects them into the process environment.
# This call must happen before any os.getenv() calls below.
load_dotenv()


class Config:
    """
    All project-wide settings as class-level attributes.

    Design choice — why a class with class attributes instead of a dict?
    Because class attributes give you IDE autocomplete, type annotations,
    and a clear namespace (Config.GROQ_API_KEY) that makes it immediately
    obvious where any setting comes from when reading agent code.
    """

    # ── Cloud Model Settings (Planning Agent — uses Groq, free tier) ──────
    # Groq provides free, blazing-fast inference for large open-source models.
    # Their free tier allows ~14,400 requests/day on Llama 3.3 70B —
    # far more than enough for development, testing, and your demo.
    # Get your free API key at: https://console.groq.com
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # Llama 3.3 70B Versatile is Groq's most capable free model.
    # At 70 billion parameters it rivals GPT-4o in instruction following
    # and structured JSON generation — exactly what the Planning Agent needs.
    GROQ_MODEL_NAME: str = "llama-3.3-70b-versatile"

    # ── Local Model Settings (Analysis Agent — uses Ollama, fully offline) ─
    # llama3.2:1b runs on CPU in under 30 seconds — reliable within timeout.
    # This satisfies the "offline/local model" requirement from the project
    # document. The model runs entirely on your machine with zero internet.
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # llama3.2:1b is 1.3GB and responds in 20-40 seconds on typical CPU.
    # It handles structured JSON generation reliably when given concise prompts.
    LOCAL_MODEL_NAME: str = "llama3.2:1b"

    # ── ChromaDB Vector Store Settings ────────────────────────────────────
    # ChromaDB persists its vector index to disk at this directory.
    # Persisting to disk means previously indexed papers survive restarts —
    # you don't need to re-embed everything every time the app starts.
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")

    # The name of the ChromaDB collection that stores all paper embeddings.
    CHROMA_COLLECTION_NAME: str = "research_papers"

    # ── Embedding Model Settings ──────────────────────────────────────────
    # This sentence transformer converts text into 384-dimensional vectors.
    # It runs entirely locally (~80MB download, cached after first use).
    # The SAME model must be used for both storing and querying —
    # different models produce vectors in incompatible geometric spaces.
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # ── RAG Pipeline Tuning Parameters ────────────────────────────────────
    # Maximum number of papers fetched from ArXiv per research topic.
    MAX_PAPERS_TO_FETCH: int = 15

    # Each paper's text is split into chunks of this many characters.
    # 512 characters balances retrieval precision with context richness.
    CHUNK_SIZE: int = 512

    # Consecutive chunks overlap by this many characters to prevent
    # sentences from being split across chunk boundaries and lost.
    CHUNK_OVERLAP: int = 64

    # How many most-similar chunks ChromaDB returns per semantic query.
    TOP_K_RETRIEVAL: int = 5

    # ── Startup Validation ────────────────────────────────────────────────
    @classmethod
    def validate(cls) -> None:
        """
        Validates that all required configuration values are present
        before any agent tries to use them. Call this once at startup
        in main.py before building the LangGraph or launching Gradio.
        """
        if not cls.GROQ_API_KEY:
            raise EnvironmentError(
                "\n❌  GROQ_API_KEY is missing from your .env file!\n"
                "    Open .env and add the line:\n"
                "    GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxx\n"
                "    Get your free key at: https://console.groq.com\n"
            )

        if "localhost" not in cls.OLLAMA_BASE_URL and \
                "127.0.0.1" not in cls.OLLAMA_BASE_URL:
            print(f"⚠️  Unusual OLLAMA_BASE_URL: {cls.OLLAMA_BASE_URL}")
            print("    Expected something like http://localhost:11434")

        print("✅  Configuration validated successfully.")
        print(f"    Cloud model    : {cls.GROQ_MODEL_NAME} via Groq (free tier)")
        print(f"    Local model    : {cls.LOCAL_MODEL_NAME} @ {cls.OLLAMA_BASE_URL}")
        print(f"    ChromaDB dir   : {cls.CHROMA_PERSIST_DIR}")
        print(f"    Embedding model: {cls.EMBEDDING_MODEL}")
        print(f"    Max papers     : {cls.MAX_PAPERS_TO_FETCH}")