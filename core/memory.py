# core/memory.py
# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB Memory Layer — the RAG infrastructure for the entire project.
#
# This module handles four responsibilities:
#   1. Initializing and persisting the ChromaDB vector store
#   2. Embedding text chunks using a local sentence transformer model
#   3. Storing paper chunks with metadata into ChromaDB
#   4. Querying the store semantically to retrieve relevant passages
#
# All three agents interact with this module:
#   - Retrieval Agent: calls store_papers() after fetching from ArXiv
#   - Analysis Agent: calls query_papers() to find gaps and limitations
#   - Planning Agent: calls query_papers() to find dataset information
# ─────────────────────────────────────────────────────────────────────────────

import os
import hashlib
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from core.config import Config


class MemoryStore:
    """
    Singleton-style wrapper around ChromaDB and the embedding model.

    Why a class rather than standalone functions?
    Loading a sentence transformer model takes a few seconds the first time.
    By wrapping everything in a class, we load the model once when the object
    is created and reuse that same loaded model for every subsequent call —
    rather than reloading it on every function call, which would be very slow.

    Usage:
        memory = MemoryStore()
        memory.store_papers(papers_list)
        results = memory.query("what limitations exist in current studies?")
    """

    def __init__(self):
        """
        Initializes ChromaDB client, creates/loads the collection,
        and loads the sentence transformer embedding model.
        Called once when the application starts up.
        """
        print("🔧 Initializing Memory Store...")

        # ── Step 1: Set up ChromaDB persistent client ─────────────────────
        # PersistentClient saves the vector store to disk at CHROMA_PERSIST_DIR.
        # This means if you restart the application, previously indexed papers
        # are still available — you don't need to re-embed everything every run.
        os.makedirs(Config.CHROMA_PERSIST_DIR, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=Config.CHROMA_PERSIST_DIR
        )

        # ── Step 2: Get or create the papers collection ───────────────────
        # get_or_create_collection is idempotent — if the collection already
        # exists from a previous run, it loads it; otherwise it creates it fresh.
        # We use cosine similarity as the distance metric because it measures
        # the angle between vectors (semantic similarity) rather than Euclidean
        # distance, which is much more appropriate for text embeddings.
        self.collection = self.client.get_or_create_collection(
            name=Config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
            # hnsw:space tells ChromaDB to use cosine distance for its
            # Hierarchical Navigable Small World (HNSW) approximate nearest
            # neighbor index — the algorithm that makes vector search fast.
        )

        # ── Step 3: Load the sentence transformer embedding model ─────────
        # This downloads the model on first run (~80MB) and caches it locally.
        # all-MiniLM-L6-v2 produces 384-dimensional embeddings and runs fast
        # on CPU, making it ideal for a local development environment.
        print(f"   Loading embedding model: {Config.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)

        print(f"✅ Memory Store ready. Collection: '{Config.CHROMA_COLLECTION_NAME}'")
        print(f"   Documents currently indexed: {self.collection.count()}")

    # ─────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPER METHODS
    # ─────────────────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str) -> List[str]:
        """
        Splits a long text (e.g., a paper abstract or full text) into smaller
        overlapping chunks suitable for embedding.

        Why do we chunk instead of embedding the whole paper at once?
        Embedding models have a maximum input length (usually 512 tokens).
        More importantly, smaller chunks lead to more precise retrieval —
        if we embed an entire paper as one vector, a query about "dataset
        limitations" might match a paper that only mentions datasets briefly
        in one sentence, because the overall paper vector averages everything.
        Smaller chunks let us pinpoint the exact relevant passage.

        Why overlap?
        If a sentence falls exactly on a chunk boundary and gets split,
        its meaning is partially lost. Overlapping consecutive chunks by
        CHUNK_OVERLAP tokens ensures every sentence appears complete in
        at least one chunk.

        Args:
            text: The raw text to split into chunks.

        Returns:
            A list of text chunks, each at most CHUNK_SIZE characters long,
            with consecutive chunks overlapping by CHUNK_OVERLAP characters.
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # Calculate end position for this chunk
            end = min(start + Config.CHUNK_SIZE, text_length)
            chunk = text[start:end].strip()

            # Only store non-empty chunks with meaningful content
            if len(chunk) > 50:
                chunks.append(chunk)

            # Move start forward by (CHUNK_SIZE - CHUNK_OVERLAP) so the
            # next chunk begins CHUNK_OVERLAP characters before this one ended
            start += Config.CHUNK_SIZE - Config.CHUNK_OVERLAP

        return chunks

    def _generate_chunk_id(self, paper_id: str, chunk_index: int) -> str:
        """
        Generates a unique, deterministic ID for each chunk.

        ChromaDB requires every document to have a unique string ID.
        We derive it from the paper_id and chunk position so the same
        chunk always gets the same ID — this prevents duplicate insertions
        if the same paper is processed more than once.

        Args:
            paper_id: The ArXiv paper ID (e.g., "2301.07041")
            chunk_index: The position of this chunk within its paper (0-indexed)

        Returns:
            A short hash string that uniquely identifies this specific chunk.
        """
        raw = f"{paper_id}_chunk_{chunk_index}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC METHODS — called by agents
    # ─────────────────────────────────────────────────────────────────────────

    def store_papers(self, papers: List[Dict[str, Any]]) -> List[str]:
        """
        Embeds and stores a list of research papers into ChromaDB.

        For each paper, this method:
            1. Combines title + abstract into a single text for embedding
            2. Splits that text into overlapping chunks
            3. Embeds each chunk using the sentence transformer
            4. Stores each chunk with its embedding and metadata into ChromaDB

        The metadata stored alongside each chunk (title, authors, paper_id, url)
        is crucial — when we retrieve a chunk during analysis, we can trace it
        back to its source paper and include proper attribution in the output.

        Args:
            papers: List of paper dictionaries, each containing at minimum
                    'paper_id', 'title', 'abstract', 'authors', 'url', 'published'

        Returns:
            List of paper_ids that were successfully indexed.
        """
        if not papers:
            print("⚠️  No papers provided to store.")
            return []

        successfully_indexed = []

        for paper in papers:
            paper_id = paper.get("paper_id", "unknown")
            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            authors = paper.get("authors", [])
            url = paper.get("url", "")
            published = paper.get("published", "")

            # Combine title and abstract — the title provides critical context
            # that helps the embedding model understand the abstract's domain
            full_text = f"Title: {title}\n\nAbstract: {abstract}"

            # Split into chunks
            chunks = self._chunk_text(full_text)

            if not chunks:
                print(f"⚠️  Paper '{title[:50]}...' produced no chunks — skipping.")
                continue

            # Embed all chunks for this paper in one batch call.
            # Batch embedding is significantly faster than embedding one-by-one
            # because the model can process multiple inputs in parallel.
            embeddings = self.embedding_model.encode(
                chunks,
                show_progress_bar=False,
                convert_to_list=True  # ChromaDB expects plain Python lists
            )

            # Prepare data structures for ChromaDB batch insertion
            chunk_ids = []
            chunk_embeddings = []
            chunk_documents = []
            chunk_metadatas = []

            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = self._generate_chunk_id(paper_id, i)

                chunk_ids.append(chunk_id)
                chunk_embeddings.append(embedding)
                chunk_documents.append(chunk)
                chunk_metadatas.append({
                    "paper_id": paper_id,
                    "title": title,
                    "authors": ", ".join(authors[:3]),  # Store first 3 authors
                    "published": published,
                    "url": url,
                    "chunk_index": i
                })

            # Insert all chunks for this paper into ChromaDB in one operation.
            # upsert (update + insert) is used instead of add so that if a paper
            # was previously indexed, its chunks are updated rather than duplicated.
            try:
                self.collection.upsert(
                    ids=chunk_ids,
                    embeddings=chunk_embeddings,
                    documents=chunk_documents,
                    metadatas=chunk_metadatas
                )
                successfully_indexed.append(paper_id)
                print(f"   ✅ Indexed: '{title[:60]}...' ({len(chunks)} chunks)")

            except Exception as e:
                print(f"   ❌ Failed to index '{title[:50]}': {e}")

        print(f"\n📚 Total papers indexed this run: {len(successfully_indexed)}")
        print(f"   Total chunks in store: {self.collection.count()}")
        return successfully_indexed

    def query(
        self,
        query_text: str,
        n_results: int = None,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Performs a semantic similarity search against the ChromaDB collection.

        This is the core of the RAG system. When the Analysis Agent wants to
        know "what limitations do existing papers acknowledge?", it calls this
        method with that query. The method:
            1. Embeds the query text into a 384-dim vector using the same model
            2. Asks ChromaDB to find the N most similar vectors in the store
            3. Returns those chunks with their text and source metadata

        The key insight is that similarity is semantic, not keyword-based.
        A query about "weaknesses in existing methods" will retrieve passages
        that discuss "limitations", "shortcomings", "gaps", and "constraints"
        even if none of those exact words appear in the query.

        Args:
            query_text: The natural language question to search for.
            n_results: Number of top results to return (defaults to Config.TOP_K_RETRIEVAL).
            filter_metadata: Optional ChromaDB metadata filter dict.

        Returns:
            A list of result dicts, each containing:
                - 'text': The retrieved chunk text
                - 'paper_id': Source paper ArXiv ID
                - 'title': Source paper title
                - 'authors': Source paper authors
                - 'url': Source paper URL
                - 'published': Publication date
                - 'distance': Cosine similarity score (lower = more similar)
        """
        if n_results is None:
            n_results = Config.TOP_K_RETRIEVAL

        # Don't query if the store is empty — return gracefully
        if self.collection.count() == 0:
            print("⚠️  ChromaDB collection is empty. No results to return.")
            return []

        # Embed the query using the same model that embedded the documents.
        # Using the same model is non-negotiable — different models produce
        # vectors in different spaces, making comparisons meaningless.
        query_embedding = self.embedding_model.encode(
            query_text,
            convert_to_list=True
        )

        # Execute the similarity search
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, self.collection.count()),
            "include": ["documents", "metadatas", "distances"]
        }

        if filter_metadata:
            query_kwargs["where"] = filter_metadata

        results = self.collection.query(**query_kwargs)

        # Flatten and restructure ChromaDB's nested response format
        # into a clean list of dicts that agents can easily iterate over
        formatted_results = []
        if results and results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                formatted_results.append({
                    "text": doc,
                    "paper_id": meta.get("paper_id", ""),
                    "title": meta.get("title", ""),
                    "authors": meta.get("authors", ""),
                    "url": meta.get("url", ""),
                    "published": meta.get("published", ""),
                    "distance": round(dist, 4)
                })

        return formatted_results

    def query_multiple(self, queries: Dict[str, str]) -> Dict[str, str]:
        """
        Runs multiple semantic queries at once and returns a consolidated
        context dictionary. This is what the Analysis Agent calls to build
        its full understanding of the literature in one step.

        For example, the Analysis Agent calls this with:
            {
                "methodologies": "what methods and models are used?",
                "datasets": "what datasets are used in experiments?",
                "limitations": "what limitations do the authors acknowledge?",
                "metrics": "what evaluation metrics are reported?"
            }

        And receives back a dictionary where each key maps to a multi-sentence
        summary of the most relevant retrieved passages for that query.

        Args:
            queries: Dict mapping context_key → query_string

        Returns:
            Dict mapping context_key → concatenated retrieved text
        """
        consolidated = {}

        for context_key, query_text in queries.items():
            results = self.query(query_text)

            if results:
                # Join the top retrieved chunks into a single context string
                # with source attribution so agents can cite their reasoning
                passages = []
                for r in results:
                    passages.append(
                        f"[From: {r['title'][:60]} ({r['published'][:4]})]\n{r['text']}"
                    )
                consolidated[context_key] = "\n\n---\n\n".join(passages)
            else:
                consolidated[context_key] = "No relevant passages found."

        return consolidated

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Returns basic statistics about the current state of the vector store.
        Useful for logging and debugging during development.
        """
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection_name": Config.CHROMA_COLLECTION_NAME,
            "persist_directory": Config.CHROMA_PERSIST_DIR
        }

    def clear_collection(self) -> None:
        """
        Deletes and recreates the collection — effectively wiping all
        indexed papers. Use this during development when you want to
        re-index papers from scratch with different chunking parameters.
        WARNING: This operation is irreversible.
        """
        self.client.delete_collection(Config.CHROMA_COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=Config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print("🗑️  Collection cleared. Starting fresh.")


# ── Module-level singleton ─────────────────────────────────────────────────────
# This creates one shared MemoryStore instance that the entire application uses.
# Importing this anywhere gives you the same object — no re-initialization needed.
# Usage in any agent file: from core.memory import memory_store
memory_store = MemoryStore()