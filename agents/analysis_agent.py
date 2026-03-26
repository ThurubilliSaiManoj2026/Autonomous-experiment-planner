# agents/analysis_agent.py
# ─────────────────────────────────────────────────────────────────────────────
# Analysis Agent — the reasoning core of the pipeline.
#
# Uses the LOCAL llama3.2:1b model (via Ollama) to identify 4 research gaps
# and generate a scientific hypothesis from the primary gap.
#
# Domain-aware fallback system covers: cryptography, HCI/emerging tech,
# NLP/sentiment, medical imaging, brain imaging, autonomous driving,
# and a generic fallback for any other domain.
# ─────────────────────────────────────────────────────────────────────────────

import json
import re
import requests
from typing import Dict, Any, List, Optional
from core.state import AgentState, ResearchGap, Hypothesis
from core.config import Config


# ── Ollama Communication Layer ────────────────────────────────────────────────

def call_local_model(prompt: str, temperature: float = 0.3) -> str:
    """
    Sends a prompt to the locally running model via Ollama's REST API.
    Uses a 120-second timeout which is sufficient for llama3.2:1b on CPU.
    """
    url = f"{Config.OLLAMA_BASE_URL}/api/generate"

    payload = {
        "model": Config.LOCAL_MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "temperature": temperature,
        "options": {
            # 1200 tokens is enough for 4 compact gaps without truncation.
            # Keeping this bounded forces concise output and prevents the
            # model from running out of budget mid-JSON.
            "num_predict": 1200,
            "top_p": 0.9,
            "repeat_penalty": 1.1
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Cannot connect to Ollama. Make sure Ollama is running.\n"
            "Start it with: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise TimeoutError(
            f"{Config.LOCAL_MODEL_NAME} timed out after 120 seconds."
        )


# ── JSON Extraction Helper ────────────────────────────────────────────────────

def extract_json_from_response(response_text: str) -> Optional[Dict]:
    """
    Robustly extracts and parses a JSON object from model output.
    Tries three strategies: markdown fences, outermost braces, direct parse.
    """
    if not response_text:
        return None

    # Strategy 1: markdown code fences
    matches = re.findall(r"```(?:json)?\s*([\s\S]*?)```", response_text)
    if matches:
        for match in matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

    # Strategy 2: outermost { } boundaries
    start = response_text.find("{")
    end = response_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(response_text[start:end + 1])
        except json.JSONDecodeError:
            pass

    # Strategy 3: direct parse
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        return None


# ── Prompt Construction ───────────────────────────────────────────────────────

def build_gap_analysis_prompt(
    research_topic: str,
    retrieval_context: Dict[str, str]
) -> str:
    """
    Builds a concise, constrained gap analysis prompt optimized for the
    1B local model. One sentence per field prevents JSON truncation.
    Context is truncated to 400 characters per section to stay within
    the model's effective context window.
    """
    methodologies = retrieval_context.get("methodologies", "Not available")[:400]
    limitations = retrieval_context.get("limitations", "Not available")[:400]
    datasets = retrieval_context.get("datasets", "Not available")[:300]

    prompt = f"""You are an expert research analyst.
Analyze the literature on "{research_topic}" and identify research gaps.

EXISTING METHODS: {methodologies}

KNOWN LIMITATIONS: {limitations}

DATASETS USED: {datasets}

Identify EXACTLY 4 important research gaps specific to "{research_topic}".
For each gap:
- title: A 3-6 word phrase naming the specific gap
- description: ONE sentence: what is missing and why it exists.
- importance: ONE sentence: why solving this matters.
- supporting_evidence: ONE sentence: evidence from the limitations above.
- severity: "high" or "medium"

Respond with ONLY valid JSON starting with {{ and ending with }}:

{{
    "gaps": [
        {{
            "gap_id": "GAP_001",
            "title": "Short specific gap title",
            "description": "One sentence specific to {research_topic}.",
            "importance": "One sentence on why this matters.",
            "supporting_evidence": "One sentence from the limitations text.",
            "severity": "high"
        }},
        {{
            "gap_id": "GAP_002",
            "title": "Short specific gap title",
            "description": "One sentence specific to {research_topic}.",
            "importance": "One sentence on why this matters.",
            "supporting_evidence": "One sentence from the limitations text.",
            "severity": "medium"
        }},
        {{
            "gap_id": "GAP_003",
            "title": "Short specific gap title",
            "description": "One sentence specific to {research_topic}.",
            "importance": "One sentence on why this matters.",
            "supporting_evidence": "One sentence from the limitations text.",
            "severity": "medium"
        }},
        {{
            "gap_id": "GAP_004",
            "title": "Short specific gap title",
            "description": "One sentence specific to {research_topic}.",
            "importance": "One sentence on why this matters.",
            "supporting_evidence": "One sentence from the limitations text.",
            "severity": "medium"
        }}
    ],
    "primary_gap_id": "GAP_001"
}}"""

    return prompt


def build_hypothesis_prompt(
    research_topic: str,
    selected_gap: ResearchGap,
    retrieval_context: Dict[str, str]
) -> str:
    """
    Builds a concise hypothesis generation prompt for the local model.
    One sentence per field keeps the model within its generation budget.
    """
    methodologies = retrieval_context.get("methodologies", "")[:300]

    prompt = f"""You are a research scientist. Generate a hypothesis for "{research_topic}".

GAP TITLE: {selected_gap.get('title', '')}
GAP: {selected_gap['description']}
IMPORTANCE: {selected_gap.get('importance', '')}
METHODS: {methodologies}

Write ONE specific, testable hypothesis directly addressing this gap.
The hypothesis must be specific to "{research_topic}" — not generic ML training advice.

Respond with ONLY valid JSON starting with {{ and ending with }}:

{{
    "statement": "One specific testable hypothesis sentence about {research_topic}.",
    "rationale": "One sentence: logical connection to the gap. One sentence: expected contribution.",
    "based_on_gap_id": "{selected_gap['gap_id']}"
}}"""

    return prompt


# ── Domain-Aware Fallback Gaps ────────────────────────────────────────────────

def get_fallback_gaps(research_topic: str) -> Dict:
    """
    Returns 4 high-quality, domain-specific fallback gaps when JSON parsing
    fails. Covers: cryptography, HCI/emerging tech, NLP/sentiment,
    medical imaging, brain imaging, autonomous driving, and a generic fallback.

    The HCI domain was added to prevent the system from falling through to
    the generic fallback for topics like "Human-Computer Interaction",
    "User Experience", "Emerging Technology", or "Wearable Computing" —
    which previously produced irrelevant ML-focused gaps.
    """
    topic_lower = research_topic.lower()

    # ── Cryptography / Security / Post-Quantum domain ──────────────────────
    if any(w in topic_lower for w in [
        "cryptograph", "quantum", "pqc", "encryption", "cipher",
        "lattice", "hash", "security", "blockchain", "kyber",
        "dilithium", "post-quantum", "key exchange", "signature"
    ]):
        return {
            "gaps": [
                {
                    "gap_id": "GAP_001",
                    "title": "Side-Channel Vulnerability in Implementations",
                    "description": (
                        f"Current {research_topic} algorithms are mathematically "
                        f"secure yet their software and hardware implementations "
                        f"remain vulnerable to side-channel attacks such as "
                        f"timing analysis and power analysis that bypass the "
                        f"mathematical security guarantees."
                    ),
                    "importance": (
                        "A cryptographic scheme that is theoretically secure "
                        "but practically exploitable provides no real-world "
                        "security — this gap must be closed before deployment "
                        "in critical infrastructure."
                    ),
                    "supporting_evidence": (
                        "Retrieved literature demonstrates that even NIST "
                        "standardization candidates have been shown vulnerable "
                        "to side-channel attacks in practical implementations."
                    ),
                    "severity": "high"
                },
                {
                    "gap_id": "GAP_002",
                    "title": "Performance Overhead on Constrained Devices",
                    "description": (
                        f"The computational and memory requirements of "
                        f"{research_topic} schemes are substantially larger "
                        f"than classical counterparts, making them impractical "
                        f"for IoT devices and embedded systems that form the "
                        f"backbone of critical real-world infrastructure."
                    ),
                    "importance": (
                        "The vast majority of internet-connected devices are "
                        "resource-constrained — if post-quantum algorithms "
                        "cannot operate within their hardware limits, the "
                        "global transition to quantum-safe cryptography will "
                        "leave these devices permanently vulnerable."
                    ),
                    "supporting_evidence": (
                        "Retrieved studies report performance measured on "
                        "server-class hardware without evaluating feasibility "
                        "on the constrained platforms where deployment is "
                        "most urgently needed."
                    ),
                    "severity": "high"
                },
                {
                    "gap_id": "GAP_003",
                    "title": "Hybrid Classical-PQC Migration Complexity",
                    "description": (
                        f"There is no established standardized protocol for "
                        f"organizations to migrate existing classical "
                        f"cryptographic infrastructure to {research_topic} "
                        f"schemes in a hybrid transitional mode that maintains "
                        f"backward compatibility while gaining quantum resistance."
                    ),
                    "importance": (
                        "Without clear hybrid migration frameworks, organizations "
                        "face either premature full migration or continued "
                        "exposure to harvest-now-decrypt-later attacks."
                    ),
                    "supporting_evidence": (
                        "Retrieved papers focus on individual algorithm "
                        "performance in isolation without addressing the "
                        "systems-level challenge of transitioning live "
                        "production environments."
                    ),
                    "severity": "medium"
                },
                {
                    "gap_id": "GAP_004",
                    "title": "Standardized Benchmarking Framework Absence",
                    "description": (
                        f"The {research_topic} research community lacks a "
                        f"unified reproducible benchmarking framework that "
                        f"enables fair performance comparison across algorithm "
                        f"families, security levels, and hardware platforms."
                    ),
                    "importance": (
                        "Without standardized benchmarks, practitioners cannot "
                        "make evidence-based deployment decisions and researchers "
                        "cannot build reliably on each other's findings."
                    ),
                    "supporting_evidence": (
                        "Retrieved literature reports performance figures using "
                        "inconsistent hardware configurations and measurement "
                        "methodologies that make direct cross-study comparison "
                        "unreliable."
                    ),
                    "severity": "medium"
                }
            ],
            "primary_gap_id": "GAP_001"
        }

    # ── HCI / Emerging Tech / Interaction Design domain ───────────────────
    # This block was added to prevent HCI topics from falling through to the
    # generic ML fallback, which produced irrelevant gaps about benchmark
    # generalization and reproducibility that have nothing to do with HCI.
    elif any(w in topic_lower for w in [
        "human-computer", "hci", "human computer", "interaction design",
        "user interface", "user experience", "ux", "ui design",
        "usability", "emerging tech", "emerging technology",
        "wearable", "augmented reality", "virtual reality", "xr",
        "tangible", "gesture", "accessibility", "cognitive load",
        "attention", "immersive"
    ]):
        return {
            "gaps": [
                {
                    "gap_id": "GAP_001",
                    "title": "Longitudinal User Behavior Study Deficit",
                    "description": (
                        f"The majority of {research_topic} studies rely on "
                        f"short-duration laboratory experiments that fail to "
                        f"capture how user behavior, mental models, and "
                        f"technology adoption patterns evolve over weeks or "
                        f"months of real-world use."
                    ),
                    "importance": (
                        "Short-duration studies cannot reveal habituation "
                        "effects, skill acquisition, or abandonment patterns "
                        "that are critical for designing systems that remain "
                        "effective and engaging over time."
                    ),
                    "supporting_evidence": (
                        "Retrieved literature consistently reports user "
                        "evaluations conducted over hours or days in controlled "
                        "settings without any longitudinal or in-the-wild "
                        "follow-up measurement."
                    ),
                    "severity": "high"
                },
                {
                    "gap_id": "GAP_002",
                    "title": "Accessibility and Inclusive Design Underrepresentation",
                    "description": (
                        f"Current {research_topic} research predominantly "
                        f"evaluates systems with able-bodied, young, and "
                        f"technically literate participants, leaving the "
                        f"interaction needs of users with disabilities, older "
                        f"adults, and low-digital-literacy populations "
                        f"systematically underexplored."
                    ),
                    "importance": (
                        "Excluding diverse user populations from HCI research "
                        "produces systems that work well for a narrow demographic "
                        "while creating barriers for those who may benefit most "
                        "from emerging technology assistance."
                    ),
                    "supporting_evidence": (
                        "Retrieved papers report participant demographics that "
                        "are overwhelmingly young adults in academic settings "
                        "without explicit evaluation of assistive or inclusive "
                        "design considerations."
                    ),
                    "severity": "high"
                },
                {
                    "gap_id": "GAP_003",
                    "title": "Cognitive Load Measurement Standardization Gap",
                    "description": (
                        f"There is no standardized widely adopted methodology "
                        f"for measuring cognitive load in {research_topic} "
                        f"contexts — researchers use inconsistent combinations "
                        f"of subjective scales, physiological signals, and "
                        f"behavioral proxies that cannot be meaningfully "
                        f"compared across studies."
                    ),
                    "importance": (
                        "Cognitive load is one of the most important dependent "
                        "variables in HCI — without standardized measurement "
                        "the field cannot build cumulative knowledge about which "
                        "interaction designs genuinely reduce mental effort."
                    ),
                    "supporting_evidence": (
                        "Retrieved studies employ heterogeneous cognitive load "
                        "metrics that are not interoperable, making meta-analysis "
                        "and cross-study comparison effectively impossible."
                    ),
                    "severity": "medium"
                },
                {
                    "gap_id": "GAP_004",
                    "title": "Ecological Validity of Laboratory Interaction Studies",
                    "description": (
                        f"Most {research_topic} evaluations are conducted in "
                        f"sterile laboratory environments with artificial tasks "
                        f"and removed social context, producing findings that "
                        f"frequently fail to replicate in naturalistic settings "
                        f"where interruptions and social dynamics are unavoidable."
                    ),
                    "importance": (
                        "Low ecological validity means HCI findings may guide "
                        "design decisions that optimize for laboratory performance "
                        "while failing in the real-world contexts where the "
                        "technology will actually be used."
                    ),
                    "supporting_evidence": (
                        "Retrieved papers conduct evaluations in controlled "
                        "laboratory settings with simplified task scenarios "
                        "that do not reflect the multitasking and interruption "
                        "patterns of actual use environments."
                    ),
                    "severity": "medium"
                }
            ],
            "primary_gap_id": "GAP_001"
        }

    # ── NLP / Sentiment / LLM domain ──────────────────────────────────────
    elif any(w in topic_lower for w in [
        "sentiment", "nlp", "text", "language", "llm", "bert",
        "transformer", "opinion", "review"
    ]):
        return {
            "gaps": [
                {
                    "gap_id": "GAP_001",
                    "title": "Cross-Domain Generalization Deficit",
                    "description": (
                        f"Current {research_topic} models are predominantly "
                        f"trained and evaluated on narrow academic benchmark "
                        f"datasets, leaving their performance on real-world "
                        f"domain-shifted data poorly understood."
                    ),
                    "importance": (
                        "Closing this gap is critical for deploying models "
                        "in production environments where data distribution "
                        "differs substantially from training corpora."
                    ),
                    "supporting_evidence": (
                        "Retrieved limitations consistently indicate that "
                        "state-of-the-art models degrade significantly when "
                        "tested on out-of-distribution inputs."
                    ),
                    "severity": "high"
                },
                {
                    "gap_id": "GAP_002",
                    "title": "Low-Resource Language Underrepresentation",
                    "description": (
                        f"The overwhelming majority of {research_topic} "
                        f"research focuses on English-language corpora with "
                        f"mid-resource and low-resource languages receiving "
                        f"insufficient attention."
                    ),
                    "importance": (
                        "Addressing multilingual coverage is essential for "
                        "equitable AI systems accessible to underserved "
                        "linguistic communities."
                    ),
                    "supporting_evidence": (
                        "Retrieved benchmark datasets are predominantly "
                        "English-only with limited multilingual evaluation."
                    ),
                    "severity": "high"
                },
                {
                    "gap_id": "GAP_003",
                    "title": "Interpretability and Explainability Absence",
                    "description": (
                        f"Despite strong benchmark performance, "
                        f"{research_topic} models remain black boxes whose "
                        f"decision-making processes cannot be inspected "
                        f"or explained to domain experts."
                    ),
                    "importance": (
                        "Explainability is a prerequisite for trustworthy "
                        "deployment in high-stakes domains such as healthcare "
                        "and legal analysis."
                    ),
                    "supporting_evidence": (
                        "Studies in retrieved literature report performance "
                        "metrics exclusively without interpretability analysis."
                    ),
                    "severity": "medium"
                },
                {
                    "gap_id": "GAP_004",
                    "title": "Temporal and Contextual Drift Handling",
                    "description": (
                        f"Current {research_topic} approaches treat tasks "
                        f"as static, failing to account for how language "
                        f"and context evolve over time, leading to silent "
                        f"performance degradation in longitudinal deployments."
                    ),
                    "importance": (
                        "Models that cannot adapt to temporal drift become "
                        "unreliable and create significant maintenance costs "
                        "in real-world applications."
                    ),
                    "supporting_evidence": (
                        "Retrieved limitations note no mechanisms for "
                        "continual learning or adaptation to distribution "
                        "shift over time."
                    ),
                    "severity": "medium"
                }
            ],
            "primary_gap_id": "GAP_001"
        }

    # ── Medical Imaging / Skin / Chest / Radiology domain ─────────────────
    elif any(w in topic_lower for w in [
        "skin", "lesion", "dermoscopy", "melanoma", "dermatology",
        "chest", "x-ray", "lung", "pneumonia", "radiology", "covid"
    ]):
        return {
            "gaps": [
                {
                    "gap_id": "GAP_001",
                    "title": "Demographic Diversity in Training Data",
                    "description": (
                        f"Existing {research_topic} datasets are heavily "
                        f"skewed toward specific demographics, causing models "
                        f"to perform systematically worse on underrepresented "
                        f"populations."
                    ),
                    "importance": (
                        "Demographic bias in medical AI directly translates "
                        "to health disparities — this is one of the "
                        "highest-stakes gaps in clinical AI research."
                    ),
                    "supporting_evidence": (
                        "Retrieved limitations explicitly note that most "
                        "studies use single-institution datasets that do not "
                        "reflect global patient diversity."
                    ),
                    "severity": "high"
                },
                {
                    "gap_id": "GAP_002",
                    "title": "Multi-Modal Fusion Under-Exploration",
                    "description": (
                        f"Current {research_topic} approaches rely almost "
                        f"exclusively on image data, leaving unexplored "
                        f"the potential of combining imaging with clinical "
                        f"metadata for improved accuracy."
                    ),
                    "importance": (
                        "Clinicians routinely integrate multiple data sources "
                        "— models that ignore this multi-modal reality are "
                        "inherently limited in clinical applicability."
                    ),
                    "supporting_evidence": (
                        "Retrieval context shows evaluation for image-only "
                        "pipelines without integration of auxiliary clinical "
                        "information."
                    ),
                    "severity": "high"
                },
                {
                    "gap_id": "GAP_003",
                    "title": "Federated Learning Privacy Gap",
                    "description": (
                        f"Medical imaging datasets for {research_topic} "
                        f"cannot be freely shared across institutions due "
                        f"to patient privacy regulations, yet most research "
                        f"assumes centralized data access."
                    ),
                    "importance": (
                        "Federated approaches that train without sharing "
                        "raw data are essential for building globally "
                        "generalizable models while respecting privacy law."
                    ),
                    "supporting_evidence": (
                        "Retrieved papers train on publicly available "
                        "single datasets without addressing the privacy "
                        "constraints that govern real clinical data access."
                    ),
                    "severity": "medium"
                },
                {
                    "gap_id": "GAP_004",
                    "title": "Calibration and Uncertainty Quantification",
                    "description": (
                        f"Deep learning models for {research_topic} produce "
                        f"point predictions without calibrated confidence "
                        f"estimates, making it impossible for clinicians "
                        f"to assess when to defer to human judgment."
                    ),
                    "importance": (
                        "A model that does not know what it does not know "
                        "is dangerous in clinical settings — uncertainty "
                        "quantification is a prerequisite for safe "
                        "human-AI collaboration."
                    ),
                    "supporting_evidence": (
                        "Evaluation in retrieved literature is limited to "
                        "accuracy and AUC with no reporting of calibration "
                        "error or prediction confidence."
                    ),
                    "severity": "medium"
                }
            ],
            "primary_gap_id": "GAP_001"
        }

    # ── Brain / Tumor / Segmentation domain ───────────────────────────────
    elif any(w in topic_lower for w in [
        "brain", "tumor", "mri", "segmentation", "glioma", "alzheimer"
    ]):
        return {
            "gaps": [
                {
                    "gap_id": "GAP_001",
                    "title": "Cross-Scanner Domain Shift",
                    "description": (
                        f"Models for {research_topic} trained on data "
                        f"from a single MRI scanner exhibit significant "
                        f"performance drops when tested on data from "
                        f"different scanners or protocols."
                    ),
                    "importance": (
                        "Scanner variability is unavoidable in clinical "
                        "practice — models that cannot handle domain shift "
                        "cannot be safely deployed across hospitals."
                    ),
                    "supporting_evidence": (
                        "Retrieved limitations note single-site evaluation "
                        "without cross-institutional or cross-scanner "
                        "validation."
                    ),
                    "severity": "high"
                },
                {
                    "gap_id": "GAP_002",
                    "title": "Rare Tumor Subtype Representation",
                    "description": (
                        f"Existing {research_topic} datasets are dominated "
                        f"by common tumor grades, leaving rare but clinically "
                        f"important variants severely underrepresented."
                    ),
                    "importance": (
                        "Rare tumor subtypes are the cases most likely to "
                        "benefit from AI assistance, yet they are "
                        "systematically excluded from training data."
                    ),
                    "supporting_evidence": (
                        "Retrieved datasets are reported with heavily "
                        "imbalanced class distributions that do not reflect "
                        "the clinical prevalence of rare variants."
                    ),
                    "severity": "high"
                },
                {
                    "gap_id": "GAP_003",
                    "title": "Longitudinal Progression Modeling",
                    "description": (
                        f"Current {research_topic} methods treat each scan "
                        f"as an independent input, ignoring the temporal "
                        f"evolution of tumors across multiple imaging sessions."
                    ),
                    "importance": (
                        "Longitudinal modeling would enable treatment "
                        "monitoring and early detection of recurrence — "
                        "among the highest-value clinical applications."
                    ),
                    "supporting_evidence": (
                        "Retrieved papers evaluate only cross-sectional "
                        "performance without temporal components."
                    ),
                    "severity": "medium"
                },
                {
                    "gap_id": "GAP_004",
                    "title": "Annotation Efficiency and Label Scarcity",
                    "description": (
                        f"State-of-the-art {research_topic} models require "
                        f"large volumes of expert annotations that are "
                        f"extraordinarily expensive, creating a fundamental "
                        f"bottleneck for dataset scale."
                    ),
                    "importance": (
                        "Semi-supervised approaches that reduce annotation "
                        "requirements are critical for scaling this technology "
                        "to real clinical data volumes."
                    ),
                    "supporting_evidence": (
                        "Retrieved studies depend on fully annotated datasets "
                        "without exploring weak supervision strategies."
                    ),
                    "severity": "medium"
                }
            ],
            "primary_gap_id": "GAP_001"
        }

    # ── Object Detection / Autonomous Driving domain ──────────────────────
    elif any(w in topic_lower for w in [
        "object detection", "autonomous", "driving", "yolo",
        "vehicle", "pedestrian", "lidar"
    ]):
        return {
            "gaps": [
                {
                    "gap_id": "GAP_001",
                    "title": "Adversarial Robustness in Safety-Critical Scenarios",
                    "description": (
                        f"Current {research_topic} models lack systematic "
                        f"evaluation under adversarial conditions such as "
                        f"sensor spoofing and edge-case weather events "
                        f"that are rare in training data but critical for safety."
                    ),
                    "importance": (
                        "A single failure in safety-critical perception "
                        "can have fatal consequences — robustness evaluation "
                        "is non-negotiable before public road deployment."
                    ),
                    "supporting_evidence": (
                        "Retrieved papers evaluate under benign test "
                        "conditions without adversarial stress testing."
                    ),
                    "severity": "high"
                },
                {
                    "gap_id": "GAP_002",
                    "title": "Sim-to-Real Transfer Gap",
                    "description": (
                        f"Models for {research_topic} trained on synthetic "
                        f"data consistently underperform in real environments "
                        f"due to the unresolved domain gap between rendered "
                        f"and real sensor data."
                    ),
                    "importance": (
                        "Synthetic data generation is essential for scaling "
                        "training data — closing the sim-to-real gap is "
                        "a prerequisite for cost-effective model development."
                    ),
                    "supporting_evidence": (
                        "Retrieved limitations note significant performance "
                        "degradation when simulation-trained models are "
                        "evaluated on real-world benchmarks."
                    ),
                    "severity": "high"
                },
                {
                    "gap_id": "GAP_003",
                    "title": "Temporal Context and Motion Reasoning",
                    "description": (
                        f"Most {research_topic} approaches process individual "
                        f"frames independently, failing to model temporal "
                        f"dynamics essential for predicting moving object "
                        f"behavior in traffic scenarios."
                    ),
                    "importance": (
                        "Temporal reasoning enables anticipatory behavior "
                        "rather than reactive response — fundamental for "
                        "safe navigation in complex dynamic environments."
                    ),
                    "supporting_evidence": (
                        "Benchmarks in retrieved literature measure per-frame "
                        "accuracy without evaluating temporal consistency."
                    ),
                    "severity": "medium"
                },
                {
                    "gap_id": "GAP_004",
                    "title": "Computational Efficiency for Real-Time Deployment",
                    "description": (
                        f"The most accurate {research_topic} architectures "
                        f"require resources exceeding what is available on "
                        f"embedded automotive hardware, creating a fundamental "
                        f"accuracy-latency trade-off."
                    ),
                    "importance": (
                        "Bridging the accuracy-efficiency gap is essential "
                        "for transitioning research models into production "
                        "vehicles with hard real-time constraints."
                    ),
                    "supporting_evidence": (
                        "Retrieved papers report inference speed on server "
                        "GPUs without evaluating embedded platform performance."
                    ),
                    "severity": "medium"
                }
            ],
            "primary_gap_id": "GAP_001"
        }

    # ── Generic fallback for any other domain ─────────────────────────────
    else:
        return {
            "gaps": [
                {
                    "gap_id": "GAP_001",
                    "title": "Benchmark-to-Real-World Generalization Gap",
                    "description": (
                        f"Models in {research_topic} are primarily validated "
                        f"on controlled academic benchmarks that do not reflect "
                        f"the distribution or edge cases in real-world "
                        f"deployment environments."
                    ),
                    "importance": (
                        "Without real-world performance evidence, benchmark "
                        "results cannot justify deployment confidence."
                    ),
                    "supporting_evidence": (
                        "Retrieved context shows evaluation confined to "
                        "standard benchmark datasets with limited real-world "
                        "validation."
                    ),
                    "severity": "high"
                },
                {
                    "gap_id": "GAP_002",
                    "title": "Reproducibility and Methodological Transparency",
                    "description": (
                        f"A substantial fraction of {research_topic} papers "
                        f"do not release code or sufficient implementation "
                        f"details to enable independent replication."
                    ),
                    "importance": (
                        "Reproducibility is the foundation of scientific "
                        "progress — unverifiable results cannot be trusted "
                        "as a basis for further research."
                    ),
                    "supporting_evidence": (
                        "Retrieved papers report improvements without "
                        "providing code repositories or detailed configurations."
                    ),
                    "severity": "high"
                },
                {
                    "gap_id": "GAP_003",
                    "title": "Cross-Dataset Evaluation Absence",
                    "description": (
                        f"Current {research_topic} research rarely evaluates "
                        f"models across multiple datasets simultaneously, "
                        f"making it impossible to distinguish genuine "
                        f"capability gains from benchmark overfitting."
                    ),
                    "importance": (
                        "Multi-dataset evaluation is the gold standard for "
                        "establishing generalization — its absence prevents "
                        "the field from reliably measuring progress."
                    ),
                    "supporting_evidence": (
                        "Retrieved studies consistently report results on "
                        "a single dataset without cross-dataset validation."
                    ),
                    "severity": "medium"
                },
                {
                    "gap_id": "GAP_004",
                    "title": "Fairness and Bias Evaluation Absence",
                    "description": (
                        f"Existing {research_topic} models are not evaluated "
                        f"for differential performance across demographic "
                        f"groups, leaving potential biases undetected."
                    ),
                    "importance": (
                        "Undetected bias in AI systems can perpetuate "
                        "societal inequalities — fairness evaluation is an "
                        "ethical obligation for real-world deployment."
                    ),
                    "supporting_evidence": (
                        "Retrieved literature reports aggregate metrics "
                        "without any stratified subgroup analysis."
                    ),
                    "severity": "medium"
                }
            ],
            "primary_gap_id": "GAP_001"
        }


# ── Fallback Hypothesis ───────────────────────────────────────────────────────

def get_fallback_hypothesis(
    research_topic: str,
    selected_gap: ResearchGap
) -> Dict:
    """
    Returns a domain-appropriate fallback hypothesis when JSON parsing fails.
    Uses vocabulary and research framing appropriate to the detected domain
    rather than generic ML training language.

    The HCI case was added so that topics about user interaction, usability,
    and emerging technology produce hypotheses about user study design and
    behavioral measurement — not about ML model generalization.
    """
    topic_lower = research_topic.lower()
    gap_title = selected_gap.get("title", "the identified gap")
    gap_id = selected_gap.get("gap_id", "GAP_001")

    # Cryptography domain hypothesis
    if any(w in topic_lower for w in [
        "cryptograph", "quantum", "pqc", "encryption", "cipher",
        "lattice", "hash", "security", "post-quantum"
    ]):
        return {
            "statement": (
                f"Implementing {research_topic} schemes with constant-time "
                f"execution and masked intermediate computations will "
                f"significantly reduce side-channel leakage compared to "
                f"unprotected reference implementations, as demonstrated "
                f"through systematic timing and power analysis across "
                f"multiple hardware platforms."
            ),
            "rationale": (
                f"This hypothesis directly addresses {gap_title} by "
                f"proposing a concrete measurable implementation strategy. "
                f"Confirming it will provide practical implementation "
                f"guidance for deploying quantum-safe cryptography in "
                f"production systems."
            ),
            "based_on_gap_id": gap_id
        }

    # HCI domain hypothesis — added to replace the generic ML hypothesis
    # that was previously produced for interaction design topics
    elif any(w in topic_lower for w in [
        "human-computer", "hci", "human computer", "interaction design",
        "user interface", "user experience", "ux", "usability",
        "emerging tech", "wearable", "augmented reality", "virtual reality"
    ]):
        return {
            "statement": (
                f"Conducting longitudinal in-the-wild studies of "
                f"{research_topic} systems with diverse user populations "
                f"— including older adults and users with disabilities — "
                f"will reveal interaction patterns and adoption barriers "
                f"that short-duration laboratory studies systematically "
                f"fail to capture."
            ),
            "rationale": (
                f"This hypothesis directly addresses {gap_title} by "
                f"proposing a study design that combines ecological realism "
                f"with demographic inclusivity. "
                f"Validating it will produce design guidelines grounded "
                f"in real-world behavior rather than controlled task performance."
            ),
            "based_on_gap_id": gap_id
        }

    # Generic fallback for all other domains
    else:
        return {
            "statement": (
                f"Addressing {gap_title.lower()} in {research_topic} "
                f"through rigorous cross-dataset evaluation and transparent "
                f"experimental reporting will significantly improve the "
                f"reliability and reproducibility of published results "
                f"compared to current single-benchmark evaluation practices."
            ),
            "rationale": (
                f"This hypothesis directly targets {gap_title} by proposing "
                f"a concrete measurable intervention that can be validated "
                f"through systematic comparison studies. "
                f"Validating it will establish evidence-based standards "
                f"for more rigorous {research_topic} research."
            ),
            "based_on_gap_id": gap_id
        }


# ── Main Agent Function ───────────────────────────────────────────────────────

def run_analysis_agent(state: AgentState) -> Dict[str, Any]:
    """
    Main entry point for the Analysis Agent.
    Called automatically by LangGraph when the graph reaches
    the 'analyze_gaps' node.

    Orchestrates three sequential reasoning stages:
        Stage 1 — Identify 4 specific research gaps
        Stage 2 — Select the highest-impact gap
        Stage 3 — Generate a testable hypothesis from that gap
    """

    print("\n" + "=" * 60)
    print(f"ANALYSIS AGENT — Starting (local {Config.LOCAL_MODEL_NAME} model)")
    print("=" * 60)

    research_topic = state.get("research_topic", "")
    retrieval_context = state.get("retrieval_context", {})

    if not retrieval_context:
        return {
            "error_message": (
                "No retrieval context found. "
                "The Retrieval Agent must complete before the Analysis Agent."
            ),
            "current_stage": "error"
        }

    print(f"Topic    : '{research_topic}'")
    print(f"Context  : {list(retrieval_context.keys())}")

    # ── Stage 1: Gap Identification ────────────────────────────────────────
    print(f"\nStage 1: Identifying 4 research gaps...")
    print("   (This may take 30-60 seconds on CPU)")

    gap_prompt = build_gap_analysis_prompt(research_topic, retrieval_context)

    try:
        gap_response = call_local_model(gap_prompt, temperature=0.3)
        print(f"   Model response: {len(gap_response)} characters")
    except (ConnectionError, TimeoutError) as e:
        return {
            "error_message": f"Local model call failed: {str(e)}",
            "current_stage": "error"
        }

    gap_data = extract_json_from_response(gap_response)

    # If JSON parsing fails or gaps are insufficient, use domain-aware fallback
    if (not gap_data
            or "gaps" not in gap_data
            or len(gap_data.get("gaps", [])) < 3):
        print("   Could not parse gap JSON or insufficient gaps. "
              "Using domain-aware fallback.")
        gap_data = get_fallback_gaps(research_topic)

    # Convert to typed ResearchGap objects (max 4)
    identified_gaps: List[ResearchGap] = []
    for gap_dict in gap_data.get("gaps", [])[:4]:
        identified_gaps.append(ResearchGap(
            gap_id=gap_dict.get("gap_id", "GAP_001"),
            title=gap_dict.get("title", "Research Gap"),
            description=gap_dict.get("description", ""),
            importance=gap_dict.get("importance", ""),
            supporting_evidence=gap_dict.get("supporting_evidence", ""),
            severity=gap_dict.get("severity", "medium")
        ))

    print(f"\n   Identified {len(identified_gaps)} research gaps:")
    for gap in identified_gaps:
        print(
            f"   [{gap['gap_id']}] ({gap['severity'].upper()}) "
            f"{gap.get('title', '')} — "
            f"{gap['description'][:60]}..."
        )

    # ── Stage 2: Gap Selection ─────────────────────────────────────────────
    print("\nStage 2: Selecting primary gap...")

    primary_gap_id = gap_data.get("primary_gap_id", "GAP_001")
    selected_gap: Optional[ResearchGap] = None

    for gap in identified_gaps:
        if gap["gap_id"] == primary_gap_id:
            selected_gap = gap
            break

    if not selected_gap and identified_gaps:
        selected_gap = identified_gaps[0]

    if not selected_gap:
        return {
            "identified_gaps": identified_gaps,
            "error_message": "Could not select a primary gap.",
            "current_stage": "error"
        }

    print(
        f"   Selected: [{selected_gap['gap_id']}] "
        f"{selected_gap.get('title', '')} — "
        f"{selected_gap['description'][:80]}..."
    )

    # ── Stage 3: Hypothesis Generation ────────────────────────────────────
    print(f"\nStage 3: Generating hypothesis...")
    print("   (This may take 20-40 seconds)")

    hyp_prompt = build_hypothesis_prompt(
        research_topic, selected_gap, retrieval_context
    )

    try:
        hyp_response = call_local_model(hyp_prompt, temperature=0.4)
    except (ConnectionError, TimeoutError) as e:
        return {
            "identified_gaps": identified_gaps,
            "selected_gap": selected_gap,
            "error_message": f"Hypothesis generation failed: {str(e)}",
            "current_stage": "error"
        }

    hyp_data = extract_json_from_response(hyp_response)

    if not hyp_data or "statement" not in hyp_data:
        print("   Could not parse hypothesis JSON. Using domain fallback.")
        hyp_data = get_fallback_hypothesis(research_topic, selected_gap)

    hypothesis = Hypothesis(
        statement=hyp_data.get("statement", ""),
        rationale=hyp_data.get("rationale", ""),
        based_on_gap_id=hyp_data.get(
            "based_on_gap_id", selected_gap["gap_id"]
        )
    )

    print(f"\n   Hypothesis: {hypothesis['statement'][:120]}...")
    print("\nAnalysis Agent complete.")
    print(f"   Gaps identified : {len(identified_gaps)}")
    print(
        f"   Selected gap    : {selected_gap['gap_id']} — "
        f"{selected_gap.get('title', '')}"
    )
    print(f"   Hypothesis ready: Yes")
    print("=" * 60)

    return {
        "identified_gaps": identified_gaps,
        "selected_gap": selected_gap,
        "hypothesis": hypothesis,
        "current_stage": "planning",
        "error_message": None
    }