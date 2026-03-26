# Autonomous Scientific Experiment Planner

An agentic AI system that autonomously retrieves research literature,
identifies knowledge gaps, generates scientific hypotheses, and produces
complete experimental proposals — given only a plain-English research topic.

---

## What the System Does

Given a research topic such as *"Deepfake Detection Systems"* or
*"Post-Quantum Cryptography"*, the pipeline:

1. Retrieves 15 relevant research papers from ArXiv using a year-wise
   backward search strategy (2026 → earlier)
2. Embeds paper chunks into a ChromaDB vector store for semantic RAG retrieval
3. Uses a local offline model (llama3.2:1b via Ollama) to identify 4
   specific research gaps and generate a testable scientific hypothesis
4. Uses a cloud model (Llama 3.3 70B via Groq) to produce a complete
   experimental blueprint covering methodology, architecture, metrics,
   baselines, datasets, contribution, and timeline

Agent Orchestration, Planning and Reasoning, Memory Integration (RAG),
Offline/Local Model, Cloud Model, and Tool Usage.

---

## System Architecture
```
User Input (Research Topic)
        │
        ▼
[Agent 1: Literature Retrieval]  ← ArXiv API + ChromaDB + sentence-transformers
        │
        ▼
[Agent 2: Analysis Agent]        ← llama3.2:1b via Ollama (LOCAL, offline)
        │
        ▼
[Agent 3: Planning Agent]        ← Llama 3.3 70B via Groq (CLOUD)
        │                          Papers With Code API
        ▼
Structured Experimental Proposal (Gradio UI — 6 result tabs)
```

Orchestration: LangGraph StateGraph with conditional edges and shared
AgentState TypedDict. Interface: Gradio Blocks with real-time streaming.

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Orchestration | LangGraph 0.2.28 |
| Local Model | llama3.2:1b via Ollama 0.18.2 |
| Cloud Model | Llama 3.3 70B via Groq API (free tier) |
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | ChromaDB 0.5.15 |
| Literature API | ArXiv API |
| Dataset API | Papers With Code public REST API |
| User Interface | Gradio 5.6.0 |
| Language | Python 3.12 |

---

## Project Structure
```
autonomous-experiment-planner/
├── agents/
│   ├── retrieval_agent.py   # Agent 1 — ArXiv + ChromaDB
│   ├── analysis_agent.py    # Agent 2 — local model, gap analysis
│   └── planning_agent.py    # Agent 3 — cloud model, experimental plan
├── core/
│   ├── state.py             # AgentState TypedDict schema
│   ├── graph.py             # LangGraph pipeline definition
│   ├── memory.py            # ChromaDB vector store interface
│   └── config.py            # Centralised configuration loader
├── tools/
│   └── arxiv_tool.py        # Year-wise ArXiv search tool
├── ui/
│   └── app.py               # Gradio streaming interface
├── main.py                  # Application entry point
├── requirements.txt         # Python dependencies
└── .env.example             # Template for required environment variables
```

---

## Setup and Installation

### Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.com/) installed on your machine
- A free [Groq API key](https://console.groq.com/)

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/autonomous-experiment-planner.git
cd autonomous-experiment-planner

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull the local model via Ollama
ollama pull llama3.2:1b

# 5. Set up your environment variables
copy .env.example .env
# Open .env and add your Groq API key
```

### Environment Variables

Create a `.env` file in the project root with the following content:
```
GROQ_API_KEY=your_groq_api_key_here
OLLAMA_BASE_URL=http://localhost:11434
CHROMA_PERSIST_DIR=./data/chroma_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Get your free Groq API key at [console.groq.com](https://console.groq.com).

### Running the Application
```bash
python main.py
```

Then open your browser at `http://localhost:7860`.

---

## Supported Research Domains

The system produces domain-accurate output for:

- Medical Imaging (skin lesion, chest X-ray, radiology)
- Brain/Neuroimaging (tumor segmentation, Alzheimer's, MRI)
- Natural Language Processing (sentiment analysis, LLMs)
- Cryptography / Post-Quantum Cryptography
- Human-Computer Interaction (HCI) and Emerging Technology
- Autonomous Driving and Object Detection
- General AI/ML research

---

## Key Design Decisions

**Why ArXiv?** ArXiv is the primary publication channel for AI/ML research.
It is free, requires no API key, and provides a structured API with
date-range filtering — essential for the year-wise backward search strategy.

**Why llama3.2:1b locally?** Satisfies the offline model requirement. 1.3 GB
runs on CPU in 30–60 seconds. Reliable for structured JSON generation when
given concise, one-sentence-per-field prompts.

**Why Llama 3.3 70B via Groq?** The planning task requires broad domain
knowledge across cryptography, HCI, medical imaging, and more. 70B parameters
provide that breadth. Groq's free tier delivers responses in 5–15 seconds.

**Why year-wise backward search?** Date-sorted ArXiv queries return the newest
papers that incidentally match keywords, not the most topically relevant ones.
Year-wise search with relevance sorting within each year ensures both recency
and topical accuracy.
