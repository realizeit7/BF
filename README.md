# Hybrid search RAG with FAISS (practice project)

A small, readable hybrid retrieval + RAG pipeline over 27 short documents
(a fictional logistics company's internal knowledge base, in `data/docs.json`).

| Piece | What it uses |
|---|---|
| Chunking | Overlapping word windows (60 words, 15 overlap), title prefixed |
| Dense retrieval | `all-MiniLM-L6-v2` embeddings in a FAISS `IndexFlatIP` (normalised vectors, so inner product = cosine) |
| Sparse retrieval | BM25 (`rank_bm25`), with a tokenizer that keeps codes like `E-4012` intact |
| Fusion | Reciprocal Rank Fusion (default) or min-max weighted score fusion |
| Generation (optional) | Claude via the Anthropic SDK, answering only from retrieved sources with citations |

## Setup

```bash
pip install -r requirements.txt
```

The embedding model (~90 MB) downloads from Hugging Face on first run.

## Run a query

```bash
python hybrid_rag.py "scanner shows E-4012"
python hybrid_rag.py "I'm adopting a child, how much time off do I get?" --mode dense -k 3
python hybrid_rag.py "how heavy can a box be" --fusion weighted --alpha 0.7
ANTHROPIC_API_KEY=... python hybrid_rag.py "how heavy can a box be" --generate
```

Each hit shows its fused score and its rank in each retriever (`dense#1 bm25#-`),
so you can see which side found it.

## Compare retrievers

```bash
python evaluate.py
```

20 labelled queries: half keyword-style (error codes, system names), half
paraphrases with little word overlap. Results from one run:

| config | Hit@1 | Recall@3 | MRR@5 |
|---|---|---|---|
| bm25 | 0.70 | 0.75 | 0.72 |
| dense | 0.85 | 1.00 | 0.93 |
| hybrid rrf | 0.80 | 0.95 | 0.87 |
| hybrid weighted a=0.5 | 0.85 | 1.00 | 0.93 |
| hybrid weighted a=0.7 | 0.90 | 1.00 | 0.95 |

On a corpus this small, dense alone is already strong; weighted fusion leaning
on dense gives the best result, and plain RRF is pulled down when BM25 ranks
an off-topic doc highly. Hybrid search matters more as the corpus grows and
more queries hinge on exact identifiers.

## Ideas to practise next

- Persist the index with `faiss.write_index` / `faiss.read_index`.
- Try `IndexHNSWFlat` or `IndexIVFFlat` and compare speed/recall.
- Add a cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) after fusion.
- Swap in a stronger embedding model (e.g. `BAAI/bge-small-en-v1.5`).
- Add harder eval queries and tune `alpha`, `rrf_k`, and chunk size.
