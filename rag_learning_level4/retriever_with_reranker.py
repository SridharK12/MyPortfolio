import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import logging

# === CONFIGURATION ===
CHROMA_PATH = r"I:\Sridhar\rag_learning_level4\docs\chroma_store"
COLLECTION_NAME = "chemistry_rag"
TOP_K_INITIAL = 10   # initial retrieval from Chroma
TOP_K_FINAL = 5      # after reranking

# === LOGGING SETUP ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logging.info("🚀 Starting Chemistry RAG Retriever with Re-ranking")

# === CONNECT TO CHROMADB ===
db = chromadb.PersistentClient(path=CHROMA_PATH)
collection = db.get_or_create_collection(COLLECTION_NAME)

# === EMBEDDING & RERANKING MODELS ===
embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# === RETRIEVAL FUNCTION ===
def retrieve_with_rerank(query, top_k_initial=TOP_K_INITIAL, top_k_final=TOP_K_FINAL):
    """
    Perform semantic search on ChromaDB and re-rank results using a cross-encoder.
    """
    logging.info(f"🔍 Query: {query}")

    # Step 1: Encode query
    query_embedding = embedder.encode(query).tolist()

    # Step 2: Retrieve from ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k_initial
    )

    if not results["documents"]:
        print("⚠️ No documents found.")
        return []

    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    ids = results["ids"][0]

    # Step 3: Prepare pairs for re-ranking
    pairs = [(query, doc) for doc in docs]

    # Step 4: Compute re-rank scores
    scores = reranker.predict(pairs)
    scores = np.array(scores)

    # Step 5: Sort by score descending
    sorted_indices = np.argsort(-scores)
    reranked_docs = [(docs[i], metadatas[i], float(scores[i])) for i in sorted_indices[:top_k_final]]

    # Step 6: Display results
    print("\n=== 🔬 Top Retrieved Results (After Re-ranking) ===")
    for rank, (doc, meta, score) in enumerate(reranked_docs, start=1):
        print(f"\nRank {rank} | Score: {score:.4f}")
        print(f"Type: {meta.get('type', 'N/A')} | Page: {meta.get('page', '?')}")
        print(f"Content: {doc[:300].strip()}...")
        if 'local_path' in meta:
            print(f"Image Path: {meta['local_path']}")

    return reranked_docs

# === MAIN EXECUTION ===
if __name__ == "__main__":
    print("🧪 Chemistry RAG Retriever Ready")
    print("Type your query (or 'exit' to quit)")

    while True:
        user_query = input("\nEnter your chemistry question: ").strip()
        if user_query.lower() in ["exit", "quit", "q"]:
            print("👋 Exiting retriever.")
            break

        results = retrieve_with_rerank(user_query)
