#!/usr/bin/env python3
"""
Retrieval Pipeline for SIGGRAPH 2025 Papers.

Implements hybrid search:
1. Semantic search (embeddings via OpenRouter + Qdrant Cloud)
2. Keyword search (BM25 - runs locally)
3. Reranking (Cohere API - optional)

Usage:
    from retrieval_pipeline import RetrievalPipeline

    pipeline = RetrievalPipeline()
    results = pipeline.retrieve("3D Gaussian Splatting", top_k=5)
"""

import json
import os
import re
import requests
import numpy as np
from typing import Optional, List
from dataclasses import dataclass
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()

# Must match the collection name used in upload_to_qdrant.py
COLLECTION_NAME = "siggraph2025_papers"


@dataclass
class RetrievalResult:
    """
    Represents a single search result.
    The api_server.py expects these exact fields - do not change!
    """

    chunk_id: str
    paper_id: str
    title: str
    authors: str
    text: str
    score: float
    chunk_type: str = ""
    chunk_section: str = ""
    pdf_url: Optional[str] = None
    github_link: Optional[str] = None
    video_link: Optional[str] = None
    acm_url: Optional[str] = None
    abstract_url: Optional[str] = None


@dataclass
class RetrievalPipelineConfig:
    """Configuration for the retrieval pipeline."""

    qdrant_url: str
    qdrant_api_key: str
    openrouter_api_key: str
    embedding_model: str = "baai/bge-large-en-v1.5"
    chunks_path: str = "./chunks.json"
    semantic_weight: float = 0.7
    bm25_weight: float = 0.3
    use_reranker: bool = True
    cohere_api_key: Optional[str] = None


class OpenRouterEmbedder:
    """
    Generate embeddings using OpenRouter API.
    Used to embed user queries for semantic search.
    """

    def __init__(self, api_key: str, model: str = "baai/bge-large-en-v1.5"):
        """
        Initialize the embedder.
        Args:
            api_key: OpenRouter API key
            model: Embedding model to use
        """
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"

    def embed_query(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        Args:
            text: Query text to embed

        Returns:
            Embedding vector as numpy array
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "input": text}
        url = f"{self.base_url}/embeddings"
        resp = requests.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            raise ValueError(f"Embedding API Error {resp.status_code}:{resp.text}")

        response_data = resp.json()

        embedding = response_data["data"][0]["embedding"]
        return np.array(embedding, dtype=np.float32)


class BM25Index:
    """
    BM25 index for keyword search.
    This runs entirely locally - no API calls needed!
    BM25 is good at finding exact keyword matches that semantic search might miss.
    """

    def __init__(self, chunks: list[dict]):
        """
        Build BM25 index from chunks.

        Args:
            chunks: List of chunk dictionaries from chunks.json
        """
        self.chunks = chunks
        self.chunk_id_to_idx = {c["chunk_id"]: i for i, c in enumerate(chunks)}
        self.tokenized_docs = [self._tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def _tokenize(self, text: str) -> list[str]:
        """
        Simple tokenization: lowercase and extract alphanumeric words.

        Args:
            text: Text to tokenize

        Returns:
            List of lowercase word tokens
        """
        text = text.lower()
        tokens = re.findall(r"\w+", text)
        return tokens

    def search(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        """
        Search for query and return top-k results.
        Args:
            query: Search query string
            top_k: Maximum number of results to return

        Returns:
            List of (chunk_index, score) tuples, sorted by score descending
        """
        query_tknz = self._tokenize(query)
        scores = self.bm25.get_scores(query_tknz)
        top_scores = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_scores if scores[i] > 0]


class RetrievalPipeline:
    """
    Main retrieval pipeline combining semantic search + BM25 + reranking.
    This is what api_server.py uses to find relevant chunks.
    """

    def __init__(self, config: Optional[RetrievalPipelineConfig] = None):
        """
        Initialize all components of the retrieval pipeline.
        Args:
            config: Optional configuration. If None, loads from environment variables.
        """
        if config is None:
            config = RetrievalPipelineConfig(
                qdrant_url=os.getenv("QDRANT_URL"),
                qdrant_api_key=os.getenv("QDRANT_API_KEY"),
                openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
                cohere_api_key=os.getenv("COHERE_API_KEY"),
                chunks_path=os.getenv("CHUNKS_PATH", "./chunks.json"),
            )
        required = {
            "QDRANT_URL": config.qdrant_url,
            "QDRANT_API_KEY": config.qdrant_api_key,
            "OPENROUTER_API_KEY": config.openrouter_api_key,
        }
        missing = [name for name, val in required.items() if not val]
        if missing:
            raise ValueError(f"Missing required config: {', '.join(missing)}")
        self.qdrant = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key)
        self.embedder = OpenRouterEmbedder(
            api_key=config.openrouter_api_key, model=config.embedding_model
        )
        with open(config.chunks_path, "r") as f:
            self.chunks = json.load(f)["chunks"]
        self.bm25_index = BM25Index(self.chunks)
        self.config = config

    def semantic_search(self, query: str, top_k: int = 30) -> list[dict]:
        """
        Perform semantic search using Qdrant.
        Returns:
            List of result dicts with chunk_id, score, and payload
        """
        query_embedding = self.embedder.embed_query(query)
        results = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding.tolist(),
            limit=top_k,
            with_payload=True,
        ).points
        return [
            {"chunk_id": r.payload["chunk_id"], "score": r.score, "payload": r.payload}
            for r in results
        ]

    def bm25_search(self, query: str, top_k: int = 30) -> list[dict]:
        """
        Perform BM25 keyword search.
        Returns:
            List of result dicts with chunk_id, score, and payload
        """
        results = self.bm25_index.search(query, top_k)
        return [
            {
                "chunk_id": self.chunks[idx]["chunk_id"],
                "score": score,
                "payload": self.chunks[idx],
            }
            for idx, score in results
        ]

    def hybrid_search(
        self, query: str, semantic_top_k: int = 30, bm25_top_k: int = 30
    ) -> list[dict]:
        """
        Combine semantic and BM25 results using weighted scoring.
        Args:
            query: Search query
            semantic_top_k: Max results from semantic search
            bm25_top_k: Max results from BM25 search

        Returns:
            Combined and sorted list of results
        """
        semantic_results = self.semantic_search(query, semantic_top_k)
        bm25_results = self.bm25_search(query, bm25_top_k)
        if semantic_results:
            max_semantic = max(r["score"] for r in semantic_results)
            for r in semantic_results:
                r["normalized_score"] = (
                    r["score"] / max_semantic if max_semantic > 0 else 0
                )
        if bm25_results:
            max_bm_scr = max(r["score"] for r in bm25_results)
            for r in bm25_results:
                r["normalized_score"] = r["score"] / max_bm_scr if max_bm_scr > 0 else 0
        combined = {}
        for r in semantic_results:
            cid = r["chunk_id"]
            combined[cid] = {
                **r,
                "semantic_score": r.get("normalized_score", 0),
                "bm25_score": 0,
                "combined_score": self.config.semantic_weight * r.get("normalized_score", 0),
            }

        for r in bm25_results:
            cid = r["chunk_id"]
            bm25_score = r.get("normalized_score", 0)
            if cid in combined:
                combined[cid]["bm25_score"] = bm25_score
                combined[cid]["combined_score"] += self.config.bm25_weight * bm25_score
            else:
                combined[cid] = {
                    **r,
                    "semantic_score": 0,
                    "bm25_score": bm25_score,
                    "combined_score": self.config.bm25_weight * bm25_score,
                }

        return sorted(combined.values(), key=lambda x: x["combined_score"], reverse=True)

    def rerank(self, query: str, results: list[dict], top_k: int = 10) -> list[dict]:
        """
        Rerank results using Cohere API (optional but improves quality).
        Args:
            query: Original query
            results: Results from hybrid_search
            top_k: Number of results to return after reranking

        Returns:
            Reranked list of results
        """
        if not self.config.cohere_api_key or not results:
            return results[:top_k]
        texts = [r["payload"]["text"] for r in results]
        try:
            resp = requests.post(
                "https://api.cohere.ai/v1/rerank",
                headers={"Authorization": f"Bearer {self.config.cohere_api_key}"},
                json={
                    "model": "rerank-english-v3.0",
                    "query": query,
                    "documents": texts,
                    "top_n": top_k,
                },
            )
            resp.raise_for_status()
            reranked_data = resp.json()["results"]
            reranked = []
            for item in reranked_data:
                r = results[item["index"]].copy()
                r["rerank_score"] = item["relevance_score"]
                reranked.append(r)
            return reranked
        except Exception as e:
            print(f"Rerank failed, falling back: {e}")
            return results[:top_k]


    def retrieve(self, query: str, top_k: int = 8) -> list[RetrievalResult]:
        """
        Full retrieval pipeline - THIS IS WHAT api_server.py CALLS!
        Args:
            query: User's search query
            top_k: Number of results to return

        Returns:
            List of RetrievalResult objects ready for RAG generation
        """
        candidates = self.hybrid_search(query)
        if self.config.use_reranker:
               reranked = self.rerank(query, candidates, top_k=min(top_k * 2, len(candidates)))
        else:
            reranked = candidates
        final = reranked[:top_k]
        return [
               RetrievalResult(
                   chunk_id=r["payload"]["chunk_id"],
                   paper_id=r["payload"]["paper_id"],
                   title=r["payload"]["title"],
                   authors=r["payload"]["authors"],
                   text=r["payload"]["text"],
                   score=r.get("rerank_score", r.get("combined_score", r.get("score", 0))),
                   chunk_type=r["payload"].get("chunk_type", ""),
                   chunk_section=r["payload"].get("chunk_section", ""),
                   pdf_url=r["payload"].get("pdf_url"),
                   github_link=r["payload"].get("github_link"),
                   video_link=r["payload"].get("video_link"),
                   acm_url=r["payload"].get("acm_url"),
                   abstract_url=r["payload"].get("abstract_url"),
               )
               for r in final
           ]


# For testing this file directly
if __name__ == "__main__":
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "3D Gaussian Splatting"

    print(f"Testing retrieval pipeline with query: '{query}'")
    print("=" * 60)

    pipeline = RetrievalPipeline()
    results = pipeline.retrieve(query, top_k=5)

    print(f"\nFound {len(results)} results:\n")

    for i, r in enumerate(results, 1):
        print(f"{i}. [{r.score:.4f}] {r.title[:60]}...")
        print(f"   Paper ID: {r.paper_id}")
        print(f"   Text preview: {r.text[:100]}...")
        print()
