
#!/usr/bin/env python
# coding: utf-8

"""
Single-File Professional RAG with Conversational Memory
- Document RAG backed by ChromaDB
- Memory of previous Q&A turns stored & retrieved from a separate Chroma collection
- Optional transcript persisted to JSONL
Author: Sridhar
"""

import os
import uuid
import json
import logging
from datetime import datetime
from typing import List, Optional

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

# -------------------- CONFIG -------------------- #
# Data & Vector Store
DATA_FILE = os.getenv("RAG_SOURCE_FILE", "./documents.csv")
CHUNK_STORE_PATH = os.getenv("CHROMA_STORE_PATH", "./chroma_store")
DOCS_COLLECTION = os.getenv("CHROMA_COLLECTION", "my_collection")

# Memory
MEMORY_COLLECTION = os.getenv("CHROMA_MEMORY_COLLECTION", "rag_memory")
SESSION_ID = os.getenv("RAG_SESSION_ID", "default-session")
MEMORY_TOP_K = int(os.getenv("RAG_MEMORY_TOP_K", "3"))
TRANSCRIPT_FILE = os.getenv("RAG_TRANSCRIPT_FILE", "./transcript.jsonl")

# Chunking
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))

# LLM
MODEL_NAME = os.getenv("RAG_MODEL_NAME", "gpt-4o-mini")
MODEL_PROVIDER = os.getenv("RAG_MODEL_PROVIDER", "openai")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------- DOCUMENT PIPELINE -------------------- #
# These functions handle loading documents from CSV, splitting into chunks,
# and indexing them into ChromaDB for vector search.


def load_documents(csv_file: str) -> List[Document]:
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    df = pd.read_csv(csv_file)
    required_cols = {"text", "source_url"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV must include columns {required_cols}. Missing: {missing}")
    docs = [
        Document(page_content=str(row["text"]), metadata={"source": str(row["source_url"])})
        for _, row in df.iterrows()
    ]
    logging.info("Loaded %d documents", len(docs))
    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
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
        name=collection_name,
        embedding_function=embedding_fn
    )
    return collection


def index_chunks(chunks: List[Document], collection) -> None:
    if not chunks:
        logging.warning("No chunks to index.")
        return
    df = pd.DataFrame([
        {"id": str(uuid.uuid4()), "source": doc.metadata.get("source", ""), "chunk": doc.page_content}
        for doc in chunks
    ])
    collection.add(
        documents=df["chunk"].tolist(),
        ids=df["id"].tolist(),
        metadatas=df[["source"]].to_dict("records")
    )
    logging.info("Indexed %d chunks into '%s'", len(df), collection.name)


def retrieve_context(query: str, collection, n_results: int = 3) -> str:
    if not query.strip():
        return ""
    results = collection.query(query_texts=[query], n_results=n_results)
    contexts = results.get("documents", [[]])[0]
    return "\n\n".join(contexts)

# -------------------- CONVERSATIONAL MEMORY -------------------- #
# ConversationMemory stores past Q&A turns into a Chroma collection and JSONL transcript.
# It allows retrieval of the most relevant memory snippets for a given query.


class ConversationMemory:
    """Lightweight memory that persists to a Chroma collection and JSONL transcript."""
    def __init__(self, persist_path: str, collection_name: str, session_id: str, transcript_file: str):
        self.session_id = session_id
        self.transcript_file = transcript_file
        self.collection = init_chroma_collection(collection_name, persist_path)

    def _now_iso(self) -> str:
        return datetime.utcnow().isoformat() + "Z"

    def add_turn(self, role: str, content: str) -> None:
        """Store a single turn (user/assistant) in vector memory and append to transcript."""
        turn_id = str(uuid.uuid4())
        # Add to vector memory
        self.collection.add(
            documents=[content],
            ids=[turn_id],
            metadatas=[{
                "session_id": self.session_id,
                "role": role,
                "timestamp": self._now_iso()
            }]
        )
        # Append to JSONL transcript
        record = {
            "id": turn_id,
            "session_id": self.session_id,
            "role": role,
            "content": content,
            "timestamp": self._now_iso()
        }
        try:
            with open(self.transcript_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as ex:
            logging.warning("Failed to write transcript: %s", ex)

    def retrieve_relevant(self, query: str, top_k: int = MEMORY_TOP_K) -> str:
        if not query.strip():
            return ""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"session_id": self.session_id}  # scope to current session
        )
        docs = results.get("documents", [[]])[0]
        # Label each memory snippet with role for clarity
        metadatas = results.get("metadatas", [[]])[0]
        labeled = []
        for text, md in zip(docs, metadatas):
            role = md.get("role", "unknown")
            ts = md.get("timestamp", "")
            labeled.append(f"[{role} @ {ts}] {text}")
        return "\n".join(labeled)


# -------------------- LLM ORCHESTRATION -------------------- #
# These functions orchestrate the retrieval of memory + document context
# and send them to the LLM to generate an answer.


def make_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Use the provided document context and relevant past conversation memory "
         "to answer the question. Be concise and cite facts from the context. "
         "If the answer is not in either context or memory, reply with 'I don't know'."),
        ("user",
         "Memory (most relevant first):\n{memory}\n\n"
         "Context from documents:\n{context}\n\n"
         "Question: {question}")
    ])


def answer_query(question: str, doc_collection, memory: ConversationMemory, ctx_k: int = 3) -> str:
    # 1) Retrieve relevant memory
    memory_text = memory.retrieve_relevant(question, top_k=MEMORY_TOP_K)
    # 2) Retrieve RAG context
    context_text = retrieve_context(question, doc_collection, n_results=ctx_k)
    # 3) LLM call
    model = init_chat_model(MODEL_NAME, model_provider=MODEL_PROVIDER)
    prompt = make_prompt()
    chain = prompt | model
    response = chain.invoke({"memory": memory_text, "context": context_text, "question": question})
    return response.content


# -------------------- MAIN (sample run) -------------------- #
# Demonstrates how the pipeline works with two queries.
# Memory ensures the second query can reference the first answer.


if __name__ == "__main__":
    try:
        # Prepare docs
        docs = load_documents(DATA_FILE)
        chunks = split_documents(docs)

        # Init collections
        doc_collection = init_chroma_collection(DOCS_COLLECTION, CHUNK_STORE_PATH)
        index_chunks(chunks, doc_collection)

        memory = ConversationMemory(
            persist_path=CHUNK_STORE_PATH,
            collection_name=MEMORY_COLLECTION,
            session_id=SESSION_ID,
            transcript_file=TRANSCRIPT_FILE
        )

        # --- Example interaction ---
        user_q1 = "What do keybullet kin drop?"
        memory.add_turn("user", user_q1)
        ans1 = answer_query(user_q1, doc_collection, memory, ctx_k=3)
        memory.add_turn("assistant", ans1)

        # Follow-up uses memory:
        user_q2 = "Are they the same drops as you mentioned earlier?"
        memory.add_turn("user", user_q2)
        ans2 = answer_query(user_q2, doc_collection, memory, ctx_k=3)
        memory.add_turn("assistant", ans2)

        logging.info("Q1: %s", user_q1)
        logging.info("A1: %s", ans1)
        logging.info("Q2: %s", user_q2)
        logging.info("A2: %s", ans2)

    except Exception as e:
        logging.error("Error in RAG pipeline with memory: %s", e, exc_info=True)
