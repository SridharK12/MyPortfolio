"""
Constitution RAG — Fully GCP-Native Pipeline
=============================================
Retrieval  : text-embedding-004  →  Vertex AI Vector Search ANN (k=20)
Reranking  : Vertex AI Ranking API (semantic-ranker-512@latest)
Generation : Vertex AI Gemini via REST (region-pinned, bypasses SDK global state)
Tracking   : MLflow experiment per query run

Environment variables required:
  GOOGLE_APPLICATION_CREDENTIALS  - path to GCP service account JSON
  VERTEX_PROJECT                   - GCP project ID
  VERTEX_REGION                    - region where Vector Search is deployed (e.g. us-east1)
  VERTEX_INDEX_ENDPOINT_ID         - numeric endpoint ID
  VERTEX_DEPLOYED_INDEX_ID         - deployed index ID string

Optional overrides:
  CHUNKS_CACHE_PATH  - default: chunks_cache.json
  TOP_K_RETRIEVE     - default: 20
  TOP_K_RERANK       - default: 5
  GEMINI_MODEL       - default: gemini-2.5-flash
  GEMINI_REGION      - default: us-central1
  RANKER_MODEL       - default: semantic-ranker-512@latest
  MLFLOW_TRACKING_URI - default: mlruns (local); set to remote URI if needed
  MLFLOW_EXPERIMENT   - default: constitution-rag
"""

import os
os.environ["GRPC_DNS_RESOLVER"] = "native"
os.environ["GRPC_VERBOSITY"] = "ERROR"
import json
import logging
import time
import google.auth
import google.auth.transport.requests
import requests as http_requests
import mlflow

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from google.cloud import aiplatform
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
    MatchingEngineIndexEndpoint,
)
from vertexai.language_models import TextEmbeddingModel
from google.cloud import discoveryengine_v1 as discoveryengine

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

GCP_PROJECT       = os.environ["VERTEX_PROJECT"]
GCP_REGION        = os.environ["VERTEX_REGION"]
INDEX_ENDPOINT_ID = os.environ["VERTEX_INDEX_ENDPOINT_ID"]
DEPLOYED_INDEX_ID = os.environ["VERTEX_DEPLOYED_INDEX_ID"]

EMBEDDING_MODEL_NAME = "text-embedding-004"
GEMINI_MODEL         = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_REGION        = os.getenv("GEMINI_REGION", "us-central1")
RANKER_MODEL         = os.getenv("RANKER_MODEL", "semantic-ranker-512@latest")

CHUNKS_CACHE_PATH   = os.getenv("CHUNKS_CACHE_PATH", "chunks_cache.json")
TOP_K_RETRIEVE      = int(os.getenv("TOP_K_RETRIEVE", "20"))
TOP_K_RERANK        = int(os.getenv("TOP_K_RERANK", "5"))

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
MLFLOW_EXPERIMENT   = os.getenv("MLFLOW_EXPERIMENT", "constitution-rag")


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: str
    text: str
    article: str = ""
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# MLflow tracker
# ─────────────────────────────────────────────────────────────────────────────

class MLflowTracker:
    """
    Wraps a single MLflow run for one RAG query.

    Usage (via context manager):
        with MLflowTracker(query) as tracker:
            chunks = retriever.retrieve(...)
            tracker.log_retrieval(chunks, latency_s)
            ...

    Logged per run
    ──────────────
    Tags
      query           : the raw question (searchable in MLflow UI)

    Params  (pipeline config — constant across runs)
      embedding_model, gemini_model, gemini_region,
      ranker_model, top_k_retrieve, top_k_rerank,
      gcp_region, chunks_cache_path

    Metrics (per-stage latency + counts)
      retrieval_latency_s, retrieved_chunk_count,
      rerank_latency_s,    reranked_chunk_count,
      generation_latency_s, total_latency_s

    Artifacts
      retrieved_chunks.json  — all k=20 candidates (article, chunk_id, text[:300])
      reranked_chunks.json   — top-k survivors as a structured table (log_table)
      answer.txt             — final generated answer
    """

    def __init__(self, query: str):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        self._query = query
        self._run = None
        self._t0 = None  # pipeline wall-clock start

    def __enter__(self):
        self._run = mlflow.start_run()
        self._t0 = time.perf_counter()

        # Pipeline-level config params
        mlflow.log_params({
            "embedding_model":   EMBEDDING_MODEL_NAME,
            "gemini_model":      GEMINI_MODEL,
            "gemini_region":     GEMINI_REGION,
            "ranker_model":      RANKER_MODEL,
            "top_k_retrieve":    TOP_K_RETRIEVE,
            "top_k_rerank":      TOP_K_RERANK,
            "gcp_region":        GCP_REGION,
            "chunks_cache_path": CHUNKS_CACHE_PATH,
        })

        # Tag the query so it's searchable in the MLflow UI
        mlflow.set_tag("query", self._query)
        return self

    def log_retrieval(self, chunks: list["Chunk"], latency_s: float) -> None:
        mlflow.log_metrics({
            "retrieval_latency_s":   latency_s,
            "retrieved_chunk_count": len(chunks),
        })
        # Full candidate list as a JSON artifact — useful for debugging
        # retrieval failures (e.g. Boolean AND misses, semantic confusion)
        records = [
            {
                "rank":     i + 1,
                "article":  c.article or c.chunk_id,
                "chunk_id": c.chunk_id,
                "text":     c.text[:300],   # preview only
            }
            for i, c in enumerate(chunks)
        ]
        mlflow.log_dict({"candidates": records}, "retrieved_chunks.json")

    def log_reranking(self, chunks: list["Chunk"], latency_s: float) -> None:
        mlflow.log_metrics({
            "rerank_latency_s":     latency_s,
            "reranked_chunk_count": len(chunks),
        })
        # log_table writes a structured table visible in the MLflow Artifacts tab
        # — best way to compare top-k chunks across runs when debugging retrieval
        if chunks:
            rows = [
                {
                    "rank":     i + 1,
                    "article":  c.article or c.chunk_id,
                    "chunk_id": c.chunk_id,
                    "text":     c.text[:500],
                }
                for i, c in enumerate(chunks)
            ]
            mlflow.log_table(
                data={
                    "columns": list(rows[0].keys()),
                    "data": [list(r.values()) for r in rows],
                },
                artifact_file="reranked_chunks.json",
            )

    def log_generation(self, answer: str, latency_s: float) -> None:
        mlflow.log_metric("generation_latency_s", latency_s)
        mlflow.log_text(answer, "answer.txt")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._t0 is not None:
            mlflow.log_metric("total_latency_s", time.perf_counter() - self._t0)
        if exc_type is not None:
            mlflow.set_tag("error", str(exc_val))
        mlflow.end_run()
        return False   # don't suppress exceptions


# ─────────────────────────────────────────────────────────────────────────────
# Chunk cache
# ─────────────────────────────────────────────────────────────────────────────

class ChunkCache:
    """Loads chunks_cache.json and provides O(1) lookup by chunk_id."""

    def __init__(self, path: str = CHUNKS_CACHE_PATH):
        self._cache: dict[str, Chunk] = {}
        self._load(path)

    def _load(self, path: str):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Chunk cache not found: {path}")
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
        for item in raw:
            meta = item.get("metadata", {})
            cid = f"article_{meta['article_number']}_chunk_{meta['chunk_number']}"
            self._cache[cid] = Chunk(
                chunk_id=cid,
                text=item["text"],
                article=meta.get("article_number", ""),
                metadata=meta,
            )
        log.info("Loaded %d chunks from cache", len(self._cache))

    def get(self, chunk_id: str) -> Optional[Chunk]:
        return self._cache.get(chunk_id)

    def __len__(self):
        return len(self._cache)


# ─────────────────────────────────────────────────────────────────────────────
# Vertex AI Vector Search retriever
# ─────────────────────────────────────────────────────────────────────────────

class VertexVectorSearchRetriever:
    """Embeds query with text-embedding-004, retrieves ANN from Vector Search."""

    def __init__(self, chunk_cache: ChunkCache):
        aiplatform.init(project=GCP_PROJECT, location=GCP_REGION)
        self._embed_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL_NAME)
        self._endpoint = MatchingEngineIndexEndpoint(
            index_endpoint_name=INDEX_ENDPOINT_ID
        )
        self._cache = chunk_cache
        log.info("Vertex retriever ready  endpoint=%s  deployed_index=%s",
                 INDEX_ENDPOINT_ID, DEPLOYED_INDEX_ID)

    def _embed(self, text: str) -> list[float]:
        result = self._embed_model.get_embeddings([text])
        return result[0].values

    def retrieve(self, query: str, k: int = TOP_K_RETRIEVE) -> list[Chunk]:
        query_vec = self._embed(query)
        response = self._endpoint.find_neighbors(
            deployed_index_id=DEPLOYED_INDEX_ID,
            queries=[query_vec],
            num_neighbors=k,
        )
        chunks: list[Chunk] = []
        for neighbor in response[0]:
            chunk = self._cache.get(neighbor.id)
            if chunk is None:
                log.warning("chunk_id not found in cache: %s", neighbor.id)
                continue
            chunks.append(chunk)
        log.info("Retrieved %d chunks for query", len(chunks))
        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Vertex AI Ranking API reranker
# ─────────────────────────────────────────────────────────────────────────────

class VertexRankingReranker:
    """
    Re-ranks candidate chunks using the Vertex AI Ranking API.
    Runs fully server-side on GCP — no local model loaded.

    Required IAM: roles/discoveryengine.viewer on the service account.
    """

    def __init__(self):
        self._client = discoveryengine.RankServiceClient()
        self._ranking_config = (
            f"projects/{GCP_PROJECT}/locations/global"
            "/rankingConfigs/default_ranking_config"
        )
        log.info("Vertex Ranking reranker ready  model=%s", RANKER_MODEL)

    def rerank(self, query: str, chunks: list[Chunk], top_k: int = TOP_K_RERANK) -> list[Chunk]:
        if not chunks:
            return []

        records = [
            discoveryengine.RankingRecord(
                id=str(i),
                content=c.text[:2000],
            )
            for i, c in enumerate(chunks)
        ]

        request = discoveryengine.RankRequest(
            ranking_config=self._ranking_config,
            model=RANKER_MODEL,
            top_n=top_k,
            query=query,
            records=records,
        )

        response = self._client.rank(request=request)

        id_to_chunk = {str(i): c for i, c in enumerate(chunks)}
        reranked = [id_to_chunk[r.id] for r in response.records if r.id in id_to_chunk]
        log.info("Reranked to top %d chunks", len(reranked))
        return reranked


# ─────────────────────────────────────────────────────────────────────────────
# Gemini generator — REST-based, region-pinned
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a precise legal assistant specialising in the Constitution of India.\n"
    "Answer the user's question using ONLY the provided context chunks.\n"
    "If the answer is not present in the context, say "
    "'I could not find this in the provided articles.'\n"
    "Cite the relevant article number(s) when possible."
)

class GeminiGenerator:
    """
    Calls Gemini via the Vertex AI REST endpoint directly.

    Why REST and not the vertexai SDK?
    vertexai.init() and aiplatform.init() share global region state.
    Calling vertexai.init(us-central1) after aiplatform.init(us-east1)
    gets silently overridden, routing Gemini calls to the wrong region.
    The REST approach pins the region in the URL, making it immune to
    SDK global state.
    """

    _URL_TEMPLATE = (
        "https://{region}-aiplatform.googleapis.com/v1"
        "/projects/{project}/locations/{region}"
        "/publishers/google/models/{model}:generateContent"
    )

    def __init__(self):
        self._creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        self._url = self._URL_TEMPLATE.format(
            region=GEMINI_REGION,
            project=GCP_PROJECT,
            model=GEMINI_MODEL,
        )
        log.info("Gemini generator ready  model=%s  region=%s", GEMINI_MODEL, GEMINI_REGION)

    def generate(self, query: str, chunks: list[Chunk]) -> str:
        if not chunks:
            return "No relevant context found to answer this question."

        context_parts = []
        for c in chunks:
            context_parts.append("[" + (c.article or c.chunk_id) + "]\n" + c.text)
        context = "\n\n---\n\n".join(context_parts)
        user_message = "Context:\n" + context + "\n\nQuestion: " + query

        # Refresh credentials if expired
        auth_req = google.auth.transport.requests.Request()
        self._creds.refresh(auth_req)

        payload = {
            "systemInstruction": {
                "parts": [{"text": SYSTEM_PROMPT}]
            },
            "contents": [
                {"role": "user", "parts": [{"text": user_message}]}
            ],
            "generationConfig": {"temperature": 0.0},
        }

        resp = http_requests.post(
            self._url,
            headers={"Authorization": "Bearer " + self._creds.token},
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# RAG pipeline
# ─────────────────────────────────────────────────────────────────────────────

class ConstitutionRAG:
    """
    End-to-end GCP-native pipeline with MLflow tracking per query.

      query
        → Vertex AI Vector Search ANN   (k=20, us-east1)
        → Vertex AI Ranking API rerank  (k=5,  global)
        → Vertex AI Gemini answer       (us-central1, via REST)
        → MLflow run logged             (constitution-rag experiment)
    """

    def __init__(self):
        self.cache     = ChunkCache(CHUNKS_CACHE_PATH)
        self.retriever = VertexVectorSearchRetriever(self.cache)
        self.reranker  = VertexRankingReranker()
        self.generator = GeminiGenerator()

    def ask(self, query: str, verbose: bool = False) -> str:
        with MLflowTracker(query) as tracker:

            # ── Stage 1: Retrieve ──────────────────────────────────────────
            t0 = time.perf_counter()
            candidates = self.retriever.retrieve(query, k=TOP_K_RETRIEVE)
            tracker.log_retrieval(candidates, time.perf_counter() - t0)

            # ── Stage 2: Rerank ────────────────────────────────────────────
            t0 = time.perf_counter()
            top_chunks = self.reranker.rerank(query, candidates, top_k=TOP_K_RERANK)
            tracker.log_reranking(top_chunks, time.perf_counter() - t0)

            if verbose:
                print("\n── Top chunks after reranking ──")
                for i, c in enumerate(top_chunks, 1):
                    preview = c.text[:120].replace("\n", " ")
                    print(f"  {i}. [{c.article or c.chunk_id}] {preview}…")
                print()

            # ── Stage 3: Generate ──────────────────────────────────────────
            t0 = time.perf_counter()
            answer = self.generator.generate(query, top_chunks)
            tracker.log_generation(answer, time.perf_counter() - t0)

        return answer


# ─────────────────────────────────────────────────────────────────────────────
# Interactive CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n=== Constitution of India — RAG (Fully GCP-Native) ===")
    print(f"Embedding : {EMBEDDING_MODEL_NAME}  ({GCP_REGION})")
    print(f"Reranker  : Vertex AI Ranking API  ({RANKER_MODEL})")
    print(f"Generator : Gemini REST             ({GEMINI_MODEL} / {GEMINI_REGION})")
    print(f"Tracking  : MLflow                  ({MLFLOW_TRACKING_URI} / {MLFLOW_EXPERIMENT})")
    print("Type 'quit' to exit.\n")

    rag = ConstitutionRAG()

    while True:
        try:
            query = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            break

        answer = rag.ask(query, verbose=True)
        print(f"\nAnswer:\n{answer}\n")
        print("─" * 60)


if __name__ == "__main__":
    main()
