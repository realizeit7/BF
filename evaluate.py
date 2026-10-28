"""Compare BM25, dense, and hybrid retrieval on a small labelled query set.

Metrics (doc level): Hit@1, Recall@3, MRR@5.
    python evaluate.py
"""

from hybrid_rag import load_retriever

# (query, relevant doc id). Mix of keyword-style and paraphrased queries.
EVAL_SET = [
    # Exact codes / names: BM25 should shine.
    ("scanner shows E-4012", "doc05"),
    ("E-7730 won't charge", "doc06"),
    ("X-Nimbus-Key header", "doc23"),
    ("what does ATTEMPTED status mean", "doc24"),
    ("GreenMile programme", "doc22"),
    ("Pathfinder lock time", "doc14"),
    # Paraphrases with little word overlap: dense should shine.
    ("I'm adopting a child, how much time off do I get?", "doc02"),
    ("can I get my money back if my urgent package showed up a day late", "doc10"),
    ("what should I do after crashing the delivery truck", "doc08"),
    ("my work computer was stolen", "doc17"),
    ("how heavy can a box be", "doc12"),
    ("can I send a phone with a battery inside", "doc13"),
    ("who teaches me the ropes when I join", "doc18"),
    ("how many vacation days do I have", "doc20"),
    ("nobody was home to receive my package", "doc25"),
    ("is wine at a client dinner refunded", "doc03"),
    ("should I fly or take the train to a city 3 hours away", "doc04"),
    ("can I take holidays in early December", "doc26"),
    ("a supplier gave me an expensive present", "doc27"),
    ("my API calls are being throttled", "doc23"),
]


def evaluate(retriever, **search_kwargs) -> dict[str, float]:
    hit1 = recall3 = mrr = 0.0
    for query, gold in EVAL_SET:
        hits = retriever.search(query, k=5, **search_kwargs)
        doc_ids = list(dict.fromkeys(h.chunk.doc_id for h in hits))  # dedupe, keep order
        hit1 += doc_ids[:1] == [gold]
        recall3 += gold in doc_ids[:3]
        mrr += 1 / (doc_ids.index(gold) + 1) if gold in doc_ids else 0
    n = len(EVAL_SET)
    return {"Hit@1": hit1 / n, "Recall@3": recall3 / n, "MRR@5": mrr / n}


def main():
    retriever = load_retriever()
    configs = {
        "bm25": dict(mode="bm25"),
        "dense": dict(mode="dense"),
        "hybrid rrf": dict(mode="hybrid", fusion="rrf"),
        "hybrid weighted a=0.5": dict(mode="hybrid", fusion="weighted", alpha=0.5),
        "hybrid weighted a=0.7": dict(mode="hybrid", fusion="weighted", alpha=0.7),
    }
    print(f"\n{len(EVAL_SET)} queries, {len(retriever.chunks)} chunks\n")
    print(f"{'config':<24}{'Hit@1':>8}{'Recall@3':>10}{'MRR@5':>8}")
    for name, kwargs in configs.items():
        m = evaluate(retriever, **kwargs)
        print(f"{name:<24}{m['Hit@1']:>8.2f}{m['Recall@3']:>10.2f}{m['MRR@5']:>8.2f}")


if __name__ == "__main__":
    main()
