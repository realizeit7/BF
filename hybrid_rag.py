"""Minimal hybrid-search RAG over a small document set.

Dense retrieval:  sentence-transformers embeddings + FAISS (inner product on
                  L2-normalised vectors = cosine similarity).
Sparse retrieval: BM25 (rank_bm25).
Fusion:           Reciprocal Rank Fusion (default) or weighted score fusion.
Generation:       optional, via the Claude API (needs ANTHROPIC_API_KEY).

Usage:
    python hybrid_rag.py "how do I fix scanner error E-4012?"
    python hybrid_rag.py "parental leave length" --mode dense -k 3
    python hybrid_rag.py "what is the parcel weight limit?" --generate
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DOCS_PATH = Path(__file__).parent / "data" / "docs.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Keeps codes like "e-4012" or "x-nimbus-key" as single tokens.
TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "be", "to", "of", "and", "or", "in",
    "on", "at", "for", "with", "by", "it", "this", "that", "do", "does", "i",
    "how", "what", "when", "can", "my", "we", "our", "if", "from", "as",
}


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class Chunk:
    chunk_id: int
    doc_id: str
    title: str
    text: str


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None


def chunk_documents(docs: list[dict], chunk_words: int = 60, overlap: int = 15) -> list[Chunk]:
    """Split each document into overlapping word windows, prefixed with the title."""
    chunks = []
    step = chunk_words - overlap
    for doc in docs:
        words = doc["text"].split()
        for start in range(0, max(len(words) - overlap, 1), step):
            piece = " ".join(words[start:start + chunk_words])
            chunks.append(Chunk(len(chunks), doc["id"], doc["title"], f"{doc['title']}. {piece}"))
    return chunks


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], model_name: str = EMBED_MODEL):
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)

        # Dense index: normalised embeddings + inner product = cosine similarity.
        embeddings = self.model.encode(
            [c.text for c in chunks], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        # Sparse index.
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])

    def dense_search(self, query: str, k: int) -> list[tuple[int, float]]:
        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        scores, ids = self.index.search(q, min(k, len(self.chunks)))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    def search(self, query: str, k: int = 5, mode: str = "hybrid", fusion: str = "rrf",
               alpha: float = 0.5, candidates: int = 20, rrf_k: int = 60) -> list[Hit]:
        dense = self.dense_search(query, candidates) if mode in ("dense", "hybrid") else []
        sparse = self.bm25_search(query, candidates) if mode in ("bm25", "hybrid") else []

        if mode == "dense":
            return [Hit(self.chunks[i], s, dense_rank=r) for r, (i, s) in enumerate(dense[:k], 1)]
        if mode == "bm25":
            return [Hit(self.chunks[i], s, bm25_rank=r) for r, (i, s) in enumerate(sparse[:k], 1)]

        dense_rank = {i: r for r, (i, _) in enumerate(dense, 1)}
        bm25_rank = {i: r for r, (i, _) in enumerate(sparse, 1)}
        fused: dict[int, float] = {}

        if fusion == "rrf":
            # score = sum over retrievers of 1 / (rrf_k + rank)
            for ranks in (dense_rank, bm25_rank):
                for i, r in ranks.items():
                    fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + r)
        elif fusion == "weighted":
            # Min-max normalise each list, then alpha * dense + (1 - alpha) * bm25.
            for results, weight in ((dense, alpha), (sparse, 1 - alpha)):
                for i, s in _min_max(results).items():
                    fused[i] = fused.get(i, 0.0) + weight * s
        else:
            raise ValueError(f"unknown fusion: {fusion}")

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
        return [Hit(self.chunks[i], s, dense_rank.get(i), bm25_rank.get(i)) for i, s in ranked]


def _min_max(results: list[tuple[int, float]]) -> dict[int, float]:
    if not results:
        return {}
    scores = [s for _, s in results]
    lo, hi = min(scores), max(scores)
    return {i: (s - lo) / (hi - lo) if hi > lo else 1.0 for i, s in results}


def generate_answer(query: str, hits: list[Hit]) -> str:
    import anthropic

    context = "\n\n".join(
        f'<source id="{h.chunk.doc_id}" title="{h.chunk.title}">\n{h.chunk.text}\n</source>' for h in hits
    )
    client = anthropic.Anthropic()
    response = client.beta.messages.create(
        model="claude-opus-5-5",
        max_tokens=16000,
        output_config={"effort": "low"},
        # On a safety-classifier decline, the API retries on a fallback model in the same call.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=(
            "Answer the question using only the provided sources. "
            "Cite source ids in square brackets, e.g. [doc05]. "
            "If the sources do not contain the answer, say you don't know."
        ),
        messages=[{"role": "user", "content": f"{context}\n\nQuestion: {query}"}],
    )
    if response.stop_reason == "refusal":
        return "(the model declined to answer)"
    return "".join(b.text for b in response.content if b.type == "text")


def load_retriever() -> HybridRetriever:
    docs = json.loads(DOCS_PATH.read_text())
    return HybridRetriever(chunk_documents(docs))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default="hybrid")
    parser.add_argument("--fusion", choices=["rrf", "weighted"], default="rrf")
    parser.add_argument("--alpha", type=float, default=0.5, help="dense weight for --fusion weighted")
    parser.add_argument("--generate", action="store_true", help="generate an answer with Claude")
    args = parser.parse_args()

    retriever = load_retriever()
    hits = retriever.search(args.query, k=args.k, mode=args.mode, fusion=args.fusion, alpha=args.alpha)

    print(f"\nQuery: {args.query}   [mode={args.mode}, fusion={args.fusion}]\n")
    for rank, h in enumerate(hits, 1):
        ranks = f"dense#{h.dense_rank or '-'} bm25#{h.bm25_rank or '-'}"
        print(f"{rank}. [{h.chunk.doc_id}] {h.chunk.title}  score={h.score:.4f}  ({ranks})")
        print(f"   {h.chunk.text[:160]}...")

    if args.generate:
        print("\nAnswer:\n" + generate_answer(args.query, hits))


if __name__ == "__main__":
    main()
