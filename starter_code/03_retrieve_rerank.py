"""
03_retrieve_rerank.py — Module 4.6 / 4.8: Hybrid Search + Reranking

GOAL: Given a query, retrieve top-k chunks using BOTH vector similarity and BM25
keyword search, merge results using Reciprocal Rank Fusion (RRF), and (optional
stretch) rerank the merged results with a cross-encoder.

This mirrors your Module 1 Option C lab (BM25 vs semantic) and Module 4 Option C lab
(hybrid search) — same idea, just without Azure AI Search.

Time budget: ~20 minutes (skip reranking first if short on time — it's marked optional)
"""

import os
import json
import chromadb
from rank_bm25 import BM25Okapi
from config import get_embedding_model, SCENARIO

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CHUNKS_PATH = os.path.join(BASE_DIR, "outputs", "chunks.json")
CHROMA_DIR = os.path.join(BASE_DIR, "outputs", "chroma_db")
BM25_CORPUS_PATH = os.path.join(BASE_DIR, "outputs", "bm25_corpus.json")
COLLECTION_NAME = f"capstone_{SCENARIO}"

TOP_K_VECTOR = 5
TOP_K_BM25 = 5
TOP_K_FINAL = 4
RRF_K = 60  # standard RRF smoothing constant


def load_chunks_lookup() -> dict:
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return {c["id"]: c for c in chunks}


def vector_search(query: str, top_k: int = TOP_K_VECTOR) -> list[str]:
    """Returns a ranked list of chunk IDs from vector similarity search."""
    model = get_embedding_model()
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results["ids"][0]  # ranked list of chunk ids


def bm25_search(query: str, top_k: int = TOP_K_BM25) -> list[str]:
    """Returns a ranked list of chunk IDs from BM25 keyword search."""
    with open(BM25_CORPUS_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    ids = payload["ids"]
    tokenized_corpus = payload["tokenized_corpus"]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
    return [chunk_id for chunk_id, score in ranked[:top_k]]


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = RRF_K) -> list[str]:
    """
    Merge multiple ranked lists of chunk IDs using RRF (Module 4.6).
    score(doc) = sum over lists of 1 / (k + rank_in_that_list)
    """
    scores = {}
    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunk_id for chunk_id, score in fused]


def hybrid_retrieve(query: str, top_k: int = TOP_K_FINAL) -> list[dict]:
    """Full hybrid retrieval pipeline: vector + BM25 -> RRF -> top-k chunk records."""
    vector_ids = vector_search(query)
    bm25_ids = bm25_search(query)
    fused_ids = reciprocal_rank_fusion([vector_ids, bm25_ids])[:top_k]

    lookup = load_chunks_lookup()
    return [lookup[cid] for cid in fused_ids if cid in lookup]


def rerank_cross_encoder(query: str, chunks: list[dict], top_k: int = TOP_K_FINAL) -> list[dict]:
    """
    OPTIONAL / STRETCH (Module 4.8): cross-encoder reranking.

    TODO if attempting this stretch goal:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [[query, c["text"]] for c in chunks]
        scores = reranker.predict(pairs)
        reranked = [c for _, c in sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)]
        return reranked[:top_k]

    For now this is a pass-through no-op so the pipeline runs end-to-end without it.
    """
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [[query, c["text"]] for c in chunks]
    scores = reranker.predict(pairs)
    reranked = [c for _, c in sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)]
    return reranked[:top_k]
    # return chunks[:top_k]


def main():
    # A few sample queries per scenario — edit to match your chosen scenario
    sample_queries = {
        "banking": [
            "Can I postpone my EMI payment?",
            "What happens if I don't report a fraudulent transaction quickly?",
        ],
        "healthcare": [
            "What should I do if I miss a dose of my medication?",
            "How often should I get my HbA1c checked?",
        ],
    }

    queries = sample_queries.get(SCENARIO, sample_queries["healthcare"])

    for query in queries:
        print(f"\n{'='*70}\nQuery: {query}\n{'='*70}")
        results = hybrid_retrieve(query)
        results = rerank_cross_encoder(query, results)  # uncomment for stretch goal
        for r in results:
            preview = r["text"][:150].replace("\n", " ")
            print(f"[{r['source']}] {preview}...")


if __name__ == "__main__":
    main()
