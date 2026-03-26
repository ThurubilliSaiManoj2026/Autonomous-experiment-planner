# tools/arxiv_tool.py
# ─────────────────────────────────────────────────────────────────────────────
# ArXiv Tool — fetches real research papers using a year-wise backward
# search strategy.
#
# Core search strategy:
#   Starting from the current year (2026), the tool queries ArXiv for
#   papers matching the research topic within that specific year's date
#   range. If enough highly relevant papers are found, the search stops.
#   If not, it moves backward one year at a time (2025, 2024, 2023, ...)
#   until 15 papers have been collected. This guarantees that the most
#   recent relevant literature is always prioritized, while ensuring
#   coverage is never sacrificed when a topic has limited recent work.
#
# Relevance strategy:
#   Within each year window, papers are sorted by relevance (not date)
#   so that the highest-quality topical matches surface first. An
#   additional relevance confidence check rejects papers whose title
#   and abstract share no keywords with the research topic — this
#   prevents loosely matching papers from polluting the results.
#
# Why year-wise instead of a single broad query?
#   A single broad query sorted by relevance can surface high-relevance
#   papers from 2007 because they are genuinely semantically similar to
#   the query — ArXiv's relevance model does not consider age. By
#   constraining each batch to a specific year, we enforce recency
#   preference explicitly while still using relevance ranking within
#   each year to pick the best papers from that period.
# ─────────────────────────────────────────────────────────────────────────────

import arxiv
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from tenacity import retry, stop_after_attempt, wait_exponential
from core.config import Config


# ── Query Construction ────────────────────────────────────────────────────────

def build_arxiv_query(research_topic: str, year: int) -> str:
    """
    Constructs an ArXiv query that combines topical relevance with a
    strict year-based date range filter.

    The date range filter uses ArXiv's submittedDate field with the format
    [YYYYMMDD TO YYYYMMDD], which constrains results to papers submitted
    within that calendar year. This is how we enforce the year-wise
    backward search — each call to this function targets exactly one year.

    The topical part of the query searches both the title field (ti:) and
    abstract field (abs:) to maximize coverage. Title matches carry the
    strongest relevance signal; abstract matches catch papers where the
    topic appears in the body but not the title.

    Args:
        research_topic: Plain English topic, e.g. "global warming solutions"
        year: The specific calendar year to search within, e.g. 2026

    Returns:
        A complete ArXiv query string with both topic and date constraints.
    """
    # Build the topical query component
    topic_clean = research_topic.strip()

    stop_words = {
        "a", "an", "the", "in", "on", "of", "for", "with",
        "using", "based", "via", "and", "or", "from", "to",
        "by", "as", "is", "are", "was", "were", "be", "been"
    }

    keywords = [
        word for word in topic_clean.lower().split()
        if word not in stop_words and len(word) > 2
    ]

    if not keywords:
        topic_query = f'abs:"{topic_clean}"'
    elif len(keywords) <= 2:
        phrase = " ".join(keywords)
        topic_query = f'ti:"{phrase}" OR abs:"{phrase}"'
    else:
        # Use first two keywords as title anchor (high precision)
        # and full phrase in abstract (broader recall)
        title_anchor = " ".join(keywords[:2])
        topic_query = f'ti:"{title_anchor}" OR abs:"{topic_clean}"'

    # Build the date range filter for this specific year.
    # ArXiv submittedDate format: YYYYMMDDHHMMSS, but date-only works fine.
    date_filter = (
        f"submittedDate:[{year}0101 TO {year}1231]"
    )

    # Combine topic query with date filter using AND
    full_query = f"({topic_query}) AND {date_filter}"

    return full_query


# ── Relevance Confidence Check ────────────────────────────────────────────────

def is_relevant(
    paper: Dict[str, Any],
    research_topic: str,
    min_keyword_matches: int = 1
) -> bool:
    """
    Performs a lightweight keyword-overlap relevance check on a paper.

    ArXiv's semantic ranking is generally good, but date-constrained
    queries can sometimes surface papers that technically match the query
    syntax but are not genuinely about the topic. This function provides
    a second line of defense by verifying that at least one meaningful
    keyword from the research topic appears in the paper's title or
    abstract.

    The check is intentionally permissive (requiring only 1 keyword match)
    so it catches obviously irrelevant papers without being so strict that
    it rejects legitimate interdisciplinary work that uses different
    vocabulary. For a topic like "global warming solutions," keywords like
    "warming," "climate," "carbon," "emission," "mitigation," "greenhouse,"
    "renewable," "temperature" would all count as valid matches — a paper
    containing any one of these is likely genuinely relevant.

    Args:
        paper: A paper dictionary with title and abstract fields.
        research_topic: The original research topic string from the user.
        min_keyword_matches: Minimum number of topic keywords that must
                             appear in the title or abstract combined.

    Returns:
        True if the paper passes the relevance check, False otherwise.
    """
    stop_words = {
        "a", "an", "the", "in", "on", "of", "for", "with",
        "using", "based", "via", "and", "or", "from", "to",
        "by", "as", "is", "are", "was", "were", "be", "been",
        "that", "this", "which", "have", "has", "its", "their"
    }

    # Extract meaningful keywords from the research topic
    topic_keywords = set(
        word.lower() for word in research_topic.split()
        if word.lower() not in stop_words and len(word) > 2
    )

    if not topic_keywords:
        return True  # Can't check, so accept the paper

    # Build a searchable text blob from title and abstract
    title = paper.get("title", "").lower()
    abstract = paper.get("abstract", "").lower()
    text_blob = f"{title} {abstract}"

    # Count how many topic keywords appear in the paper's text
    matches = sum(1 for kw in topic_keywords if kw in text_blob)

    return matches >= min_keyword_matches


# ── Year-wise Paper Fetcher ───────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
def fetch_papers_for_year(
    research_topic: str,
    year: int,
    max_results: int = 8,
    seen_ids: Optional[Set[str]] = None
) -> List[Dict[str, Any]]:
    """
    Fetches relevant papers from ArXiv for a single specific year.

    This is the atomic unit of the year-wise backward search strategy.
    Each call targets one calendar year, sorts results by relevance within
    that year, applies quality and relevance filtering, and returns the
    accepted papers. The calling function (search_papers) aggregates
    results across years until the target quota is reached.

    The seen_ids parameter enables deduplication across year batches —
    if a paper was already collected from an earlier (more recent) year,
    it won't be added again when the same paper appears in an older year's
    results.

    Args:
        research_topic: The research topic from the user.
        year: The specific year to search within.
        max_results: Maximum candidates to fetch from ArXiv for this year
                     before filtering. Set higher than needed to account
                     for papers that fail quality or relevance checks.
        seen_ids: Set of paper_id strings already collected in earlier
                  year batches, used for deduplication.

    Returns:
        A list of relevant, deduplicated paper dictionaries for this year.
        May be empty if no relevant papers exist for this year.
    """
    if seen_ids is None:
        seen_ids = set()

    query = build_arxiv_query(research_topic, year)

    try:
        search = arxiv.Search(
            query=query,
            # Fetch more candidates than needed to account for filtering.
            # If we need 5 papers from a year, fetch 15 candidates —
            # some will fail quality or relevance checks.
            max_results=max_results * 3,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending
        )

        client = arxiv.Client()
        raw_results = list(client.results(search))

    except Exception as e:
        print(f"      ArXiv API error for year {year}: {e}")
        return []

    accepted_papers = []

    for result in raw_results:

        # Quality filter 1: meaningful abstract required
        if not result.summary or len(result.summary.strip()) < 100:
            continue

        # Quality filter 2: meaningful title required
        title = result.title.strip()
        if len(title) < 10:
            continue

        # Extract clean paper ID and deduplicate
        paper_id = result.entry_id.split("/abs/")[-1].split("v")[0]
        if paper_id in seen_ids:
            continue

        authors = [author.name for author in result.authors]
        published_str = (
            result.published.strftime("%Y-%m-%d")
            if result.published else f"{year}-01-01"
        )

        paper = {
            "paper_id": paper_id,
            "title": title,
            "authors": authors,
            "abstract": result.summary.strip(),
            "published": published_str,
            "url": f"https://arxiv.org/abs/{paper_id}",
            "categories": result.categories
        }

        # Relevance confidence check — reject papers with no keyword overlap
        if not is_relevant(paper, research_topic):
            continue

        accepted_papers.append(paper)
        seen_ids.add(paper_id)

        # Stop once we have enough papers from this year
        if len(accepted_papers) >= max_results:
            break

    return accepted_papers


# ── Main Entry Point ──────────────────────────────────────────────────────────

def search_papers(
    research_topic: str,
    max_results: int = None
) -> List[Dict[str, Any]]:
    """
    Main entry point for the ArXiv tool. Implements the year-wise
    backward search strategy.

    The algorithm works as follows. It starts at the current year (2026)
    and fetches the most relevant papers published in that year for the
    given topic. If those papers plus any previously collected papers reach
    the target of 15, the search stops. If not, it moves to the previous
    year (2025) and repeats. This continues backward through the years
    until either 15 papers have been collected or the search exhausts a
    reasonable historical range (defaulting to 10 years back from 2026,
    i.e., back to 2016).

    Within each year, papers are sorted by relevance using ArXiv's own
    semantic ranking, so the best matching papers from each year are
    always selected first. Papers are also checked for keyword overlap
    with the research topic to reject loosely matching results.

    The strategy deliberately collects at most 5 papers per year in its
    primary pass. This ensures temporal diversity — rather than filling
    all 15 slots from 2023 alone (a particularly prolific year for many
    ML topics), the system spreads coverage across recent years, giving
    the Analysis Agent a richer picture of how the field has evolved.

    Args:
        research_topic: The research domain entered by the user.
        max_results: Total number of papers to collect across all years.

    Returns:
        A list of up to max_results paper dictionaries, sorted from
        most recent to least recent, all verified as relevant to the topic.
    """
    if max_results is None:
        max_results = Config.MAX_PAPERS_TO_FETCH

    print(f"\n   Starting year-wise backward search for: '{research_topic}'")
    print(f"   Target: {max_results} relevant papers")

    all_papers: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()

    # Determine the search range.
    # Start from the current year and go back up to 10 years.
    current_year = datetime.now().year
    earliest_year = current_year - 10

    # Papers per year in the primary collection pass.
    # 5 per year means we spread across at least 3 years before filling
    # the quota, giving temporal diversity in the literature coverage.
    papers_per_year_primary = 5

    # ── Primary pass: 5 papers per year, most recent first ────────────────
    for year in range(current_year, earliest_year - 1, -1):

        if len(all_papers) >= max_results:
            break

        remaining_needed = max_results - len(all_papers)
        # Don't collect more than papers_per_year_primary in primary pass,
        # but also don't collect more than we actually still need
        fetch_this_year = min(papers_per_year_primary, remaining_needed)

        print(f"\n   Searching year {year}...")

        year_papers = fetch_papers_for_year(
            research_topic=research_topic,
            year=year,
            max_results=fetch_this_year,
            seen_ids=seen_ids
        )

        if year_papers:
            all_papers.extend(year_papers)
            print(
                f"      Accepted {len(year_papers)} paper(s) from {year}. "
                f"Total so far: {len(all_papers)}"
            )
        else:
            print(f"      No relevant papers found for {year}.")

        # Brief pause between year queries to be a good API citizen
        # and avoid triggering ArXiv's rate limiting
        if year > earliest_year:
            time.sleep(0.5)

    # ── Secondary pass: fill remaining quota if primary pass fell short ───
    # If the primary pass collected fewer than max_results papers
    # (which can happen for niche or emerging topics), we do a second
    # pass over the same years but allow more papers per year.
    if len(all_papers) < max_results:
        print(
            f"\n   Primary pass yielded {len(all_papers)} papers. "
            f"Running secondary pass to fill remaining {max_results - len(all_papers)} slots..."
        )

        for year in range(current_year, earliest_year - 1, -1):

            if len(all_papers) >= max_results:
                break

            remaining_needed = max_results - len(all_papers)

            year_papers = fetch_papers_for_year(
                research_topic=research_topic,
                year=year,
                max_results=remaining_needed,
                seen_ids=seen_ids
            )

            if year_papers:
                all_papers.extend(year_papers)
                print(
                    f"      Secondary pass: added {len(year_papers)} "
                    f"from {year}. Total: {len(all_papers)}"
                )

            time.sleep(0.5)

    # ── Final sort: most recent papers first ──────────────────────────────
    # Sort the collected papers by publication date, newest first.
    # This ensures the Literature tab in the UI displays papers
    # chronologically from most to least recent, matching user expectation.
    all_papers.sort(
        key=lambda p: p.get("published", "0000-00-00"),
        reverse=True
    )

    # Trim to exact quota
    all_papers = all_papers[:max_results]

    print(f"\n   Year-wise search complete.")
    print(f"   Total papers collected: {len(all_papers)}")
    print(f"   Year range covered: ", end="")

    if all_papers:
        years = sorted(
            set(p["published"][:4] for p in all_papers),
            reverse=True
        )
        print(", ".join(years))
    else:
        print("none")

    print(f"\n   Final paper list:")
    for i, paper in enumerate(all_papers, 1):
        print(
            f"   {i:2}. [{paper['published'][:4]}] "
            f"{paper['title'][:65]}..."
        )

    return all_papers


def get_paper_by_id(paper_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetches a single specific paper by its ArXiv ID.

    Args:
        paper_id: ArXiv paper ID like "2301.07041"

    Returns:
        A single paper dictionary, or None if not found.
    """
    try:
        search = arxiv.Search(id_list=[paper_id])
        client = arxiv.Client()
        results = list(client.results(search))

        if not results:
            return None

        result = results[0]
        clean_id = result.entry_id.split("/abs/")[-1].split("v")[0]

        return {
            "paper_id": clean_id,
            "title": result.title.strip(),
            "authors": [a.name for a in result.authors],
            "abstract": result.summary.strip(),
            "published": (
                result.published.strftime("%Y-%m-%d")
                if result.published else "Unknown"
            ),
            "url": f"https://arxiv.org/abs/{clean_id}",
            "categories": result.categories
        }

    except Exception as e:
        print(f"   Could not fetch paper {paper_id}: {e}")
        return None