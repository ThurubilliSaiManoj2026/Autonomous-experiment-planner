# agents/planning_agent.py
# ─────────────────────────────────────────────────────────────────────────────
# Planning Agent — the final synthesizer in the three-agent pipeline.
#
# Uses the CLOUD model (Groq — Llama 3.3 70B, free tier) to generate a
# complete experimental blueprint from the hypothesis produced by the
# Analysis Agent.
#
# Domain detection covers: cryptography, HCI/emerging tech, NLP,
# medical imaging, brain imaging, autonomous driving, and general ML.
# The HCI domain was added so that interaction design topics receive
# user-study methodology, HCI-appropriate metrics (SUS, NASA-TLX),
# and real HCI datasets rather than ML training pipeline descriptions.
# ─────────────────────────────────────────────────────────────────────────────

import json
import re
import requests
from typing import Dict, Any, List, Optional
from groq import Groq
from core.state import (
    AgentState,
    Hypothesis,
    ResearchGap,
    ExperimentalPlan,
    DatasetSuggestion
)
from core.config import Config


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Domain Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_domain(research_topic: str) -> str:
    """
    Detects the research domain from the topic string.

    This drives both dataset selection and planning prompt customization,
    ensuring the experimental plan contains vocabulary, architectures,
    metrics, and datasets appropriate to the actual field rather than
    defaulting to computer vision or NLP frameworks for every topic.

    Returns one of: "cryptography", "hci", "nlp", "medical_imaging",
    "brain_imaging", "autonomous_driving", or "general_ml".

    The "hci" case was added to handle Human-Computer Interaction,
    User Experience, Emerging Technology, and Wearable Computing topics
    that previously fell through to "general_ml" and received incorrect
    ML-focused output.
    """
    topic_lower = research_topic.lower()

    if any(w in topic_lower for w in [
        "cryptograph", "quantum", "pqc", "encryption", "cipher",
        "lattice", "hash", "security protocol", "blockchain",
        "post-quantum", "key exchange", "digital signature",
        "kyber", "dilithium", "falcon", "sphincs"
    ]):
        return "cryptography"

    elif any(w in topic_lower for w in [
        "human-computer", "hci", "human computer", "interaction design",
        "user interface", "user experience", "ux", "ui design",
        "usability", "emerging tech", "emerging technology",
        "wearable", "augmented reality", "virtual reality", "xr",
        "tangible", "gesture", "accessibility", "cognitive load",
        "attention", "immersive"
    ]):
        return "hci"

    elif any(w in topic_lower for w in [
        "sentiment", "text classification", "language model",
        "llm", "bert", "gpt", "nlp", "natural language",
        "opinion mining", "named entity"
    ]):
        return "nlp"

    elif any(w in topic_lower for w in [
        "skin", "lesion", "dermoscopy", "melanoma", "dermatology",
        "chest", "x-ray", "xray", "lung", "pneumonia", "radiology"
    ]):
        return "medical_imaging"

    elif any(w in topic_lower for w in [
        "brain", "tumor", "mri", "glioma", "alzheimer",
        "neuroimaging", "segmentation"
    ]):
        return "brain_imaging"

    elif any(w in topic_lower for w in [
        "object detection", "autonomous", "driving", "yolo",
        "vehicle", "pedestrian", "lidar", "radar"
    ]):
        return "autonomous_driving"

    else:
        return "general_ml"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Domain Context Instructions
# ─────────────────────────────────────────────────────────────────────────────

def get_domain_context(domain: str, research_topic: str) -> str:
    """
    Returns domain-specific instructions injected into the planning prompt.

    These instructions tell the cloud model what kind of architecture,
    metrics, datasets, and baselines are appropriate for the detected domain.
    Without this context, the model defaults to neural network architectures
    and ML training pipelines regardless of the actual research field.

    The "hci" case ensures that topics about human-computer interaction
    receive user-study methodology descriptions, HCI evaluation metrics
    (SUS, NASA-TLX, task completion time), and appropriate baselines
    (WIMP interfaces, ISO 9241) instead of EfficientNet and image datasets.
    """
    if domain == "cryptography":
        return f"""
DOMAIN CONTEXT: This is a CRYPTOGRAPHY / SECURITY research topic, not a machine learning topic.
The experimental plan must reflect cryptographic research practices:

- Proposed Architecture: Describe a CRYPTOGRAPHIC IMPLEMENTATION approach — e.g., software
  implementation in C/Python using constant-time arithmetic, hardware implementation on FPGA,
  or a benchmarking framework comparing multiple PQC algorithm families (lattice-based,
  hash-based, code-based). Do NOT suggest neural network architectures like EfficientNet or BERT.

- Evaluation Metrics: Use CRYPTOGRAPHIC metrics such as:
  Key generation time (ms), Encapsulation/Decapsulation time (ms),
  Ciphertext size (bytes), Public key size (bytes),
  Security level (NIST Level 1/3/5), Side-channel leakage score.

- Baseline Comparisons: Compare against existing algorithms such as:
  Classical RSA-2048, ECDH (Elliptic Curve Diffie-Hellman),
  CRYSTALS-Kyber, CRYSTALS-Dilithium, FALCON, SPHINCS+, FrodoKEM.

- Suggested Datasets: Reference CRYPTOGRAPHIC benchmarks such as:
  NIST PQC Standardization submission packages, SUPERCOP benchmarking suite,
  eBACS (ECRYPT Benchmarking of Cryptographic Systems), or the
  PQClean reference implementation library.

- Methodology: Describe implementation, profiling, and evaluation steps
  appropriate for cryptographic algorithm research — not ML training pipelines.
"""

    elif domain == "hci":
        return f"""
DOMAIN CONTEXT: This is a HUMAN-COMPUTER INTERACTION (HCI) research topic.
HCI research evaluates how people interact with technology through user studies,
not purely through machine learning model training pipelines.

- Proposed Architecture/Approach: Describe a STUDY DESIGN or SYSTEM ARCHITECTURE
  appropriate for HCI research — e.g., a mixed-methods user study combining
  physiological sensing (EEG, eye-tracking, galvanic skin response) with
  behavioral logging and semi-structured interviews. Or a system architecture
  for a novel interaction paradigm (gesture-based, voice-based, multimodal).
  Do NOT suggest neural network architectures like EfficientNet as the primary contribution.

- Evaluation Metrics: Use HCI-appropriate metrics such as:
  Task completion time, Error rate, System Usability Scale (SUS) score,
  NASA Task Load Index (NASA-TLX) for cognitive load, User engagement score,
  Presence/immersion rating (for VR/AR), Learnability curve, Retention rate.

- Baseline Comparisons: Compare against existing interaction paradigms such as:
  Traditional WIMP (Windows, Icons, Menus, Pointer) interfaces,
  State-of-the-art systems in the specific interaction domain (e.g.,
  current voice assistants, existing gesture recognition systems),
  Established usability benchmarks from ISO 9241.

- Suggested Datasets: Reference HCI-specific datasets and resources such as:
  CHI proceedings datasets, ELAN annotation corpora,
  MIT Reality Mining dataset, ACM Digital Library HCI corpus,
  or publicly available interaction logs from prior published studies.

- Methodology: Describe participant recruitment strategy, study protocol,
  ethical considerations, measurement instruments, and analysis approach
  (e.g., thematic analysis for qualitative data, ANOVA for quantitative).
  This is a USER STUDY methodology, not an ML training pipeline.
"""

    elif domain == "nlp":
        return f"""
DOMAIN CONTEXT: This is a NATURAL LANGUAGE PROCESSING research topic.
- Architecture: Suggest transformer-based models (BERT, RoBERTa, LLaMA, GPT variants).
- Metrics: Accuracy, F1-score, BLEU, ROUGE, Perplexity as appropriate.
- Baselines: Compare against relevant NLP baselines (BERT, RoBERTa, traditional ML like SVM/Naive Bayes).
- Datasets: Suggest NLP benchmark datasets relevant to the specific task (SST-2, IMDb, GLUE, SQuAD, etc.).
"""

    elif domain == "medical_imaging":
        return f"""
DOMAIN CONTEXT: This is a MEDICAL IMAGING research topic.
- Architecture: Suggest CNN architectures proven for medical imaging (ResNet, EfficientNet, U-Net for segmentation, Vision Transformers).
- Metrics: AUC-ROC, Sensitivity, Specificity, F1-score, Dice coefficient (for segmentation).
- Baselines: Compare against established medical imaging models.
- Datasets: Suggest real medical imaging datasets (ISIC, NIH ChestX-ray14, CheXpert).
"""

    elif domain == "brain_imaging":
        return f"""
DOMAIN CONTEXT: This is a BRAIN/NEUROIMAGING research topic.
- Architecture: Suggest architectures appropriate for volumetric 3D medical data (3D U-Net, nnU-Net, TransUNet).
- Metrics: Dice coefficient, Hausdorff distance, sensitivity, specificity.
- Baselines: Compare against established segmentation methods (U-Net, DeepMedic, nnU-Net).
- Datasets: Suggest neuroimaging datasets (BraTS, ADNI, TCGA-GBM).
"""

    elif domain == "autonomous_driving":
        return f"""
DOMAIN CONTEXT: This is an AUTONOMOUS DRIVING / COMPUTER VISION research topic.
- Architecture: Suggest detection architectures (YOLO, EfficientDet, PointPillars for LiDAR).
- Metrics: mAP, IoU, FPS (frames per second), NDS (nuScenes Detection Score).
- Baselines: Compare against established detectors (YOLO, Faster R-CNN, PointPillars).
- Datasets: Suggest autonomous driving datasets (KITTI, nuScenes, Waymo Open Dataset).
"""

    else:
        return f"""
DOMAIN CONTEXT: This is a general AI/ML research topic.
Suggest appropriate architectures, metrics, and datasets for the specific task described.
"""


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Dataset Selection
# ─────────────────────────────────────────────────────────────────────────────

def get_topic_aware_fallback_datasets(
    research_topic: str,
    domain: str
) -> List[DatasetSuggestion]:
    """
    Returns domain-appropriate datasets based on the detected domain.

    All datasets listed are real, publicly accessible, and widely used
    as benchmarks in their respective research communities.

    The "hci" case returns genuinely HCI-appropriate resources —
    ACM CHI proceedings, MIT Reality Mining, and ELAN annotation corpora —
    rather than general-purpose ML repositories that have no HCI-specific data.
    """

    if domain == "cryptography":
        return [
            DatasetSuggestion(
                name="NIST PQC Standardization Submissions",
                description=(
                    "The official NIST Post-Quantum Cryptography "
                    "standardization package containing reference "
                    "implementations, test vectors, and specification "
                    "documents for all Round 4 candidates including "
                    "CRYSTALS-Kyber, CRYSTALS-Dilithium, FALCON, and SPHINCS+."
                ),
                url="https://csrc.nist.gov/projects/post-quantum-cryptography",
                size="Reference implementations + test vectors for 4 algorithms"
            ),
            DatasetSuggestion(
                name="SUPERCOP Benchmarking Suite",
                description=(
                    "System for Unified Performance Evaluation Related to "
                    "Cryptographic Operations and Primitives — provides "
                    "standardized benchmarking infrastructure for measuring "
                    "cryptographic implementation performance across hardware."
                ),
                url="https://bench.cr.yp.to/supercop.html",
                size="Benchmarks across 2000+ systems"
            ),
            DatasetSuggestion(
                name="PQClean Reference Library",
                description=(
                    "A collection of clean, portable, and reviewed "
                    "implementations of post-quantum cryptographic schemes, "
                    "providing a standardized baseline for implementation "
                    "quality and performance comparison."
                ),
                url="https://github.com/PQClean/PQClean",
                size="20+ PQC algorithm implementations"
            )
        ]

    elif domain == "hci":
        # HCI-specific resources — added to replace the generic UCI/HuggingFace
        # fallback that appeared for interaction design topics previously.
        return [
            DatasetSuggestion(
                name="CHI Proceedings Dataset (ACM Digital Library)",
                description=(
                    "The complete archive of ACM CHI Conference proceedings "
                    "containing over 10,000 peer-reviewed HCI papers with "
                    "structured metadata — used for literature analysis, "
                    "trend identification, and gap discovery in HCI research."
                ),
                url="https://dl.acm.org/conference/chi",
                size="10,000+ papers"
            ),
            DatasetSuggestion(
                name="MIT Reality Mining Dataset",
                description=(
                    "Longitudinal behavioral dataset from 100 subjects over "
                    "9 months capturing mobile phone usage, location, "
                    "communication patterns, and social proximity — ideal "
                    "for studying naturalistic technology interaction patterns."
                ),
                url="https://realitycommons.media.mit.edu/realitymining.html",
                size="100 subjects, 9 months"
            ),
            DatasetSuggestion(
                name="ELAN Interaction Annotation Corpus",
                description=(
                    "Multimodal interaction annotation dataset using ELAN "
                    "software for gesture, gaze, speech, and facial expression "
                    "analysis — standard tool for qualitative and quantitative "
                    "HCI interaction data coding."
                ),
                url="https://archive.mpi.nl/tla/elan",
                size="Variable — community-contributed corpora"
            )
        ]

    elif domain == "nlp":
        return [
            DatasetSuggestion(
                name="SST-2 (Stanford Sentiment Treebank)",
                description="Binary sentiment benchmark with 67,349 movie review sentences.",
                url="https://huggingface.co/datasets/sst2",
                size="67,349 sentences"
            ),
            DatasetSuggestion(
                name="IMDb Movie Reviews",
                description="50,000 movie reviews for binary sentiment classification.",
                url="https://huggingface.co/datasets/imdb",
                size="50,000 reviews"
            ),
            DatasetSuggestion(
                name="TweetEval Sentiment",
                description="Twitter-based multi-class sentiment benchmark.",
                url="https://huggingface.co/datasets/tweet_eval",
                size="45,000 tweets"
            )
        ]

    elif domain == "medical_imaging":
        return [
            DatasetSuggestion(
                name="ISIC 2020 Challenge Dataset",
                description="33,126 dermoscopic images across 9 diagnostic categories.",
                url="https://challenge2020.isic-archive.com/",
                size="33,126 images"
            ),
            DatasetSuggestion(
                name="HAM10000",
                description="10,015 dermoscopic images from diverse acquisition sites.",
                url="https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T",
                size="10,015 images"
            ),
            DatasetSuggestion(
                name="NIH ChestX-ray14",
                description="112,120 frontal chest X-rays with 14 disease labels.",
                url="https://nihcc.app.box.com/v/ChestXray-NIHCC",
                size="112,120 images"
            )
        ]

    elif domain == "brain_imaging":
        return [
            DatasetSuggestion(
                name="BraTS 2023",
                description="Multimodal MRI scans with expert annotations for glioma segmentation.",
                url="https://www.synapse.org/Synapse:syn51156910/wiki/622351",
                size="1,251 subjects"
            ),
            DatasetSuggestion(
                name="ADNI (Alzheimer's Disease Neuroimaging Initiative)",
                description="Longitudinal MRI and PET data for Alzheimer's research.",
                url="https://adni.loni.usc.edu/",
                size="800+ subjects"
            )
        ]

    elif domain == "autonomous_driving":
        return [
            DatasetSuggestion(
                name="KITTI Vision Benchmark",
                description="Real-world autonomous driving dataset with stereo images and 3D annotations.",
                url="https://www.cvlibs.net/datasets/kitti/",
                size="15,000 images"
            ),
            DatasetSuggestion(
                name="nuScenes",
                description="1,000 driving scenes with full sensor suite and 3D bounding box annotations.",
                url="https://www.nuscenes.org/",
                size="1,000 scenes"
            )
        ]

    else:
        return [
            DatasetSuggestion(
                name="UCI Machine Learning Repository",
                description="Comprehensive dataset collection for ML research across many domains.",
                url="https://archive.ics.uci.edu/",
                size="600+ datasets"
            ),
            DatasetSuggestion(
                name="Hugging Face Datasets Hub",
                description="Largest open-source dataset repository for NLP, CV, and multimodal tasks.",
                url="https://huggingface.co/datasets",
                size="50,000+ datasets"
            )
        ]


def fetch_datasets_from_paperswithcode(
    research_topic: str,
    domain: str
) -> List[DatasetSuggestion]:
    """
    Queries the Papers With Code API for real dataset suggestions.
    Falls back to domain-aware curated datasets if the API is unavailable.
    The two-tier approach ensures the pipeline never fails due to network issues
    while always producing relevant dataset suggestions.
    """
    print(f"\n   Fetching datasets from Papers With Code API...")

    try:
        response = requests.get(
            "https://paperswithcode.com/api/v1/datasets/",
            params={"q": research_topic, "limit": 5},
            timeout=15,
            headers={"User-Agent": "AutonomousExperimentPlanner/1.0"}
        )
        response.raise_for_status()
        data = response.json()

        datasets: List[DatasetSuggestion] = []
        for item in data.get("results", [])[:5]:
            introduced_in = item.get("introduced_in") or {}
            source_title = introduced_in.get("title", "Available on Papers With Code")
            datasets.append(DatasetSuggestion(
                name=item.get("name", "Unknown Dataset"),
                description=(
                    f"{item.get('full_name', item.get('name', ''))} "
                    f"— Introduced in: {source_title}"
                ),
                url=item.get("url", "https://paperswithcode.com/datasets"),
                size=f"{item.get('num_papers', 'Unknown')} papers use this"
            ))

        if datasets:
            print(f"   Found {len(datasets)} datasets from Papers With Code.")
            return datasets

        print("   Papers With Code returned empty results.")

    except requests.exceptions.RequestException as e:
        print(f"   Papers With Code API error: {e}")

    fallback = get_topic_aware_fallback_datasets(research_topic, domain)
    print(f"   Using {len(fallback)} domain-aware fallback datasets.")
    return fallback


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Prompt Construction
# ─────────────────────────────────────────────────────────────────────────────

def build_planning_prompt(
    research_topic: str,
    hypothesis: Hypothesis,
    selected_gap: ResearchGap,
    retrieval_context: Dict[str, str],
    datasets: List[DatasetSuggestion],
    domain: str
) -> str:
    """
    Builds the comprehensive planning prompt for Llama 3.3 70B.

    The domain_context parameter is injected at the start of the prompt
    to override the model's tendency to default to ML architectures and
    datasets regardless of the actual research domain. This is the most
    important fix for domain alignment — without it, the model applies
    its strongest prior (computer vision / NLP) to every topic.
    """
    existing_methods = retrieval_context.get("methodologies", "")[:700]
    existing_metrics = retrieval_context.get("evaluation_metrics", "")[:400]
    existing_limitations = retrieval_context.get("limitations", "")[:500]

    domain_context = get_domain_context(domain, research_topic)

    dataset_lines = "\n".join([
        f"  - Name: {d['name']}\n"
        f"    Description: {d['description']}\n"
        f"    Size: {d['size']}\n"
        f"    URL: {d['url']}"
        for d in datasets
    ])

    prompt = f"""You are a world-class research scientist and experimental designer.
Create a complete experimental plan to test this scientific hypothesis.

{domain_context}

=======================================================
RESEARCH CONTEXT
=======================================================

Research Domain: {research_topic}

Research Gap: {selected_gap.get('title', '')}
{selected_gap['description']}

Scientific Hypothesis:
"{hypothesis['statement']}"

Hypothesis Rationale:
{hypothesis['rationale']}

=======================================================
EXISTING LITERATURE
=======================================================

Current Methodologies:
{existing_methods}

Current Evaluation Metrics:
{existing_metrics}

Known Limitations:
{existing_limitations}

=======================================================
AVAILABLE DATASETS / BENCHMARKS
=======================================================

{dataset_lines}

=======================================================
TASK
=======================================================

Design a complete experimental plan that is APPROPRIATE TO THE DOMAIN
described in the DOMAIN CONTEXT above. Match the domain exactly.
Use the datasets/benchmarks listed above — do not invent new ones.

Respond with ONLY valid JSON starting with {{ and ending with }}:

{{
    "objective": "One precise sentence: what this experiment proves and why it matters for {research_topic}.",

    "methodology": "5-6 sentences: setup, procedure, evaluation, and hypothesis validation. Must be appropriate for {research_topic} — not a generic ML training pipeline.",

    "proposed_architecture": "3-4 sentences: specific implementation or study design appropriate for {research_topic}. Name exact real methods or frameworks.",

    "evaluation_metrics": [
        "Metric name — justification specific to {research_topic}",
        "Metric name — justification",
        "Metric name — justification",
        "Metric name — justification"
    ],

    "baseline_comparisons": [
        "Method/approach name — why this is the key baseline for {research_topic}",
        "Method/approach name — comparison value",
        "Method/approach name — comparison value"
    ],

    "suggested_datasets": [
        {{
            "name": "Exact name from the list above",
            "description": "One sentence: why this is ideal for testing this hypothesis in {research_topic}",
            "url": "Exact URL from above",
            "size": "Exact size from above"
        }}
    ],

    "expected_contribution": "3 sentences: new knowledge if confirmed, what is learned if rejected, how future researchers build on this.",

    "estimated_timeline": "Phase 1 (X weeks): activity. Phase 2 (X weeks): activity. Phase 3 (X weeks): activity. Phase 4 (X weeks): activity."
}}"""

    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — JSON Extraction Helper
# ─────────────────────────────────────────────────────────────────────────────

def extract_json_safely(response_text: str) -> Optional[Dict]:
    """
    Robustly extracts and parses a JSON object from the Groq response.
    Tries three strategies: markdown code fences, outermost braces, direct parse.
    """
    if not response_text or not response_text.strip():
        return None

    # Strategy 1: markdown code fences
    code_fence_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    fence_matches = re.findall(code_fence_pattern, response_text)
    if fence_matches:
        for match in fence_matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

    # Strategy 2: outermost braces
    first_brace = response_text.find("{")
    last_brace = response_text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(response_text[first_brace: last_brace + 1])
        except json.JSONDecodeError:
            pass

    # Strategy 3: direct parse
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Main Agent Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def run_planning_agent(state: AgentState) -> Dict[str, Any]:
    """
    Main entry point for the Planning Agent.
    Called automatically by LangGraph when the graph reaches
    the 'plan_experiment' node.

    Orchestrates:
        1. Domain detection to drive domain-appropriate output
        2. Dataset fetching (Papers With Code API with domain-aware fallback)
        3. Groq API call with domain-context-enriched prompt
        4. JSON parsing and ExperimentalPlan construction
    """

    print("\n" + "=" * 60)
    print("PLANNING AGENT — Starting")
    print(f"   Using: Groq API — {Config.GROQ_MODEL_NAME}")
    print("=" * 60)

    research_topic = state.get("research_topic", "").strip()
    hypothesis: Optional[Hypothesis] = state.get("hypothesis")
    selected_gap: Optional[ResearchGap] = state.get("selected_gap")
    retrieval_context: Dict[str, str] = state.get("retrieval_context", {})

    if not hypothesis:
        error_msg = (
            "No hypothesis found in state. "
            "The Analysis Agent must complete before the Planning Agent."
        )
        print(f"   {error_msg}")
        return {"error_message": error_msg, "current_stage": "error"}

    if not selected_gap:
        error_msg = "No selected_gap found. Analysis Agent output is incomplete."
        print(f"   {error_msg}")
        return {"error_message": error_msg, "current_stage": "error"}

    # ── Detect domain ──────────────────────────────────────────────────────
    domain = detect_domain(research_topic)
    print(f"   Research topic : '{research_topic}'")
    print(f"   Detected domain: {domain}")
    print(f"   Hypothesis     : {hypothesis['statement'][:100]}...")
    print(f"   Addressing gap : [{selected_gap['gap_id']}] "
          f"{selected_gap.get('title', '')}")

    # ── Fetch domain-appropriate datasets ─────────────────────────────────
    datasets = fetch_datasets_from_paperswithcode(research_topic, domain)

    # ── Build domain-aware planning prompt ────────────────────────────────
    planning_prompt = build_planning_prompt(
        research_topic=research_topic,
        hypothesis=hypothesis,
        selected_gap=selected_gap,
        retrieval_context=retrieval_context,
        datasets=datasets,
        domain=domain
    )

    # ── Call Groq API ──────────────────────────────────────────────────────
    print(f"\n   Calling Groq ({Config.GROQ_MODEL_NAME})...")
    print(f"   Expect a response in 5-15 seconds.")

    try:
        client = Groq(api_key=Config.GROQ_API_KEY)

        response = client.chat.completions.create(
            model=Config.GROQ_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert research scientist and experimental "
                        "designer. You always respond with valid JSON only. "
                        "Never add text before or after the JSON object. "
                        "Start your response directly with { and end with }. "
                        "Always respect the DOMAIN CONTEXT provided in the "
                        "user message — match architectures and methods to "
                        "the actual research domain, not to generic ML."
                    )
                },
                {
                    "role": "user",
                    "content": planning_prompt
                }
            ],
            temperature=0.4,
            max_tokens=2000
        )

        raw_response = response.choices[0].message.content
        print(f"   Response received: {len(raw_response)} characters")

    except Exception as e:
        error_msg = f"Groq API call failed: {str(e)}"
        print(f"   {error_msg}")
        return {
            "error_message": error_msg,
            "current_stage": "error",
            "is_complete": False
        }

    # ── Parse JSON response ────────────────────────────────────────────────
    plan_data = extract_json_safely(raw_response)

    if not plan_data:
        error_msg = (
            "Could not parse the experimental plan as JSON. "
            f"Raw preview: {raw_response[:300]}"
        )
        print(f"   {error_msg}")
        return {
            "error_message": error_msg,
            "current_stage": "error",
            "is_complete": False
        }

    # ── Build typed ExperimentalPlan ───────────────────────────────────────
    plan_datasets: List[DatasetSuggestion] = []
    raw_datasets = plan_data.get("suggested_datasets", [])

    if raw_datasets and isinstance(raw_datasets, list):
        for d in raw_datasets:
            if isinstance(d, dict):
                plan_datasets.append(DatasetSuggestion(
                    name=d.get("name", ""),
                    description=d.get("description", ""),
                    url=d.get("url", ""),
                    size=d.get("size", "")
                ))

    if not plan_datasets:
        plan_datasets = datasets

    experimental_plan = ExperimentalPlan(
        objective=plan_data.get("objective", ""),
        methodology=plan_data.get("methodology", ""),
        proposed_architecture=plan_data.get("proposed_architecture", ""),
        evaluation_metrics=plan_data.get("evaluation_metrics", []),
        baseline_comparisons=plan_data.get("baseline_comparisons", []),
        suggested_datasets=plan_datasets,
        expected_contribution=plan_data.get("expected_contribution", ""),
        estimated_timeline=plan_data.get("estimated_timeline", "")
    )

    # ── Print summary ──────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("EXPERIMENTAL PLAN — GENERATED")
    print("─" * 60)
    print(f"Objective     : {experimental_plan['objective'][:100]}...")
    print(f"Architecture  : {experimental_plan['proposed_architecture'][:100]}...")
    metrics = experimental_plan['evaluation_metrics']
    metric_names = [m.split("—")[0].strip() for m in metrics[:3]]
    print(f"Metrics       : {', '.join(metric_names)}")
    print(f"Baselines     : {len(experimental_plan['baseline_comparisons'])} specified")
    ds_names = [d['name'] for d in experimental_plan['suggested_datasets']]
    print(f"Datasets      : {', '.join(ds_names)}")
    print("─" * 60)
    print("\nPlanning Agent complete. Full pipeline finished.")
    print("=" * 60)

    return {
        "experimental_plan": experimental_plan,
        "current_stage": "complete",
        "is_complete": True,
        "error_message": None
    }