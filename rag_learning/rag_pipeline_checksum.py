import os
import uuid
import json
import hashlib
import logging
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# -------------------- CONFIG -------------------- #
DATA_FILE = os.getenv("RAG_SOURCE_FILE", "./documents.csv")
CHUNK_STORE_PATH = os.getenv("CHROMA_STORE_PATH", "./chroma_store")
DOCS_COLLECTION = os.getenv("CHROMA_COLLECTION", "my_collection")

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------- HELPERS -------------------- #
def compute_checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# -------------------- DOCUMENT PIPELINE -------------------- #
def load_documents(csv_file: str):
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    df = pd.read_csv(csv_file)
    required_cols = {"text", "source_url"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV must include columns {required_cols}. Missing: {missing}")

    docs = []
    for _, row in df.iterrows():
        text = str(row["text"])
        source = str(row["source_url"])
        checksum = compute_checksum(text)
        docs.append(Document(page_content=text, metadata={"source": source, "checksum": checksum}))

    logging.info("Loaded %d documents", len(docs))
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    logging.info("Split into %d chunks", len(chunks))
    return chunks


def init_chroma_collection(collection_name: str, persist_path: str):
    client = chromadb.PersistentClient(path=persist_path)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=embedding_fn
    )
    return collection


def get_existing_checksums(collection):
    try:
        # Fetch all metadata to check for existing checksums
        results = collection.get(include=["metadatas"], limit=None)
        existing = {md.get("checksum") for md in results["metadatas"] if md.get("checksum")}
        return existing
    except Exception:
        return set()


def index_new_documents(docs, collection):
    existing_checksums = get_existing_checksums(collection)
    new_docs = [doc for doc in docs if doc.metadata["checksum"] not in existing_checksums]

    if not new_docs:
        logging.info("No new documents to index.")
        return

    chunks = split_documents(new_docs)

    df = pd.DataFrame(
        [
            {
                "id": str(uuid.uuid4()),
                "source": doc.metadata.get("source", ""),
                "chunk": doc.page_content,
                "checksum": doc.metadata.get("checksum"),
            }
            for doc in chunks
        ]
    )

    collection.add(
        documents=df["chunk"].tolist(),
        ids=df["id"].tolist(),
        metadatas=df[["source", "checksum"]].to_dict("records"),
    )
    logging.info("Indexed %d new chunks", len(df))


if __name__ == "__main__":
    docs = load_documents(DATA_FILE)
    doc_collection = init_chroma_collection(DOCS_COLLECTION, CHUNK_STORE_PATH)
    index_new_documents(docs, doc_collection)