#!/usr/bin/env python
# coding: utf-8

"""
RAG with Conversational Memory + PII Guardrails (SSN, Credit Cards, Phone, Email) + Token Tracking
- Document RAG backed by ChromaDB
- Memory of previous Q&A turns stored & retrieved from a separate Chroma collection
- Optional transcript persisted to JSONL
- Guardrails to mask SSN, credit card numbers, phone numbers, and email addresses
- Token tracking for prompt, completion, and estimated cost
Author: Sridhar
"""

import os
import uuid
import json
import logging
import re
import math
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

# Try tiktoken for accurate token counting
try:
    import tiktoken
    _HAS_TIKTOKEN = True
except ImportError:
    _HAS_TIKTOKEN = False

# -------------------- CONFIG -------------------- #
DATA_FILE = os.getenv("RAG_SOURCE_FILE", "./documents.csv")
CHUNK_STORE_PATH = os.getenv("CHROMA_STORE_PATH", "./chroma_store")
DOCS_COLLECTION = os.getenv("CHROMA_COLLECTION", "my_collection")

MEMORY_COLLECTION = os.getenv("CHROMA_MEMORY_COLLECTION", "rag_memory")
SESSION_ID = os.getenv("RAG_SESSION_ID", "default-session")
MEMORY_TOP_K = int(os.getenv("RAG_MEMORY_TOP_K", "3"))
TRANSCRIPT_FILE = os.getenv("RAG_TRANSCRIPT_FILE", "./transcript.jsonl")

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))

MODEL_NAME = os.getenv("RAG_MODEL_NAME", "gpt-4o-mini")
MODEL_PROVIDER = os.getenv("RAG_MODEL_PROVIDER", "openai")

# Cost parameters (adjust to your model pricing)
COST_PROMPT_PER_1K = float(os.getenv("RAG_TOKEN_COST_PROMPT_PER_1K", "0.150"))
COST_COMPLETION_PER_1K = float(os.getenv("RAG_TOKEN_COST_COMPLETION_PER_1K", "0.600"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------- TOKEN TRACKER -------------------- #
class TokenTracker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.prompt_tokens = 0
        self.completion_tokens = 0
        if _HAS_TIKTOKEN:
            try:
                self.enc = tiktoken.encoding_for_model(model_name)
            except Exception:
                self.enc = tiktoken.get_encoding("cl100k_base")
        else:
            self.enc = None

    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.enc:
            return len(self.enc.encode(text))
        return max(1, math.ceil(len(text) / 4))  # heuristic fallback

    def add_prompt(self, *parts: str) -> None:
        self.prompt_tokens += sum(self.estimate_tokens(p) for p in parts if p)

    def add_completion(self, text: str) -> None:
        self.completion_tokens += self.estimate_tokens(text)

    def cost_usd(self) -> float:
        return (self.prompt_tokens / 1000.0) * COST_PROMPT_PER_1K + \
               (self.completion_tokens / 1000.0) * COST_COMPLETION_PER_1K

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "estimated_cost_usd": round(self.cost_usd(), 6),
        }

# -------------------- DOCUMENT PIPELINE -------------------- #
def load_documents(csv_file: str) -> List[Document]:
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    df = pd.read_csv(csv_file)
    required_cols = {"text", "source_url"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV must include columns {required_cols}. Missing: {missing}")
    return [
        Document(page_content=str(row["text"]), metadata={"source": str(row["source_url"])})
        for _, row in df.iterrows()
    ]

def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs)

def init_chroma_collection(collection_name: str, persist_path: str):
    client = chromadb.PersistentClient(path=persist_path)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    return client.get_or_create_collection(name=collection_name, embedding_function=embedding_fn)

def index_chunks(chunks: List[Document], collection) -> None:
    if not chunks:
        return
    df = pd.DataFrame([
        {"id": str(uuid.uuid4()), "source": doc.metadata.get("source", ""), "chunk": doc.page_content}
        for doc in chunks
    ])
    collection.add(documents=df["chunk"].tolist(), ids=df["id"].tolist(), metadatas=df[["source"]].to_dict("records"))

def retrieve_context(query: str, collection, n_results: int = 3) -> str:
    if not query.strip():
        return ""
    results = collection.query(query_texts=[query], n_results=n_results)
    return "\n\n".join(results.get("documents", [[]])[0])

# -------------------- CONVERSATIONAL MEMORY -------------------- #
class ConversationMemory:
    def __init__(self, persist_path: str, collection_name: str, session_id: str, transcript_file: str):
        self.session_id = session_id
        self.transcript_file = transcript_file
        self.collection = init_chroma_collection(collection_name, persist_path)

    def _now_iso(self) -> str:
        return datetime.utcnow().isoformat() + "Z"

    def add_turn(self, role: str, content: str) -> None:
        turn_id = str(uuid.uuid4())
        self.collection.add(
            documents=[content],
            ids=[turn_id],
            metadatas=[{"session_id": self.session_id, "role": role, "timestamp": self._now_iso()}]
        )
        record = {"id": turn_id, "session_id": self.session_id, "role": role, "content": content, "timestamp": self._now_iso()}
        with open(self.transcript_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def retrieve_relevant(self, query: str, top_k: int = MEMORY_TOP_K) -> str:
        if not query.strip():
            return ""
        results = self.collection.query(query_texts=[query], n_results=top_k, where={"session_id": self.session_id})
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        return "\n".join(f"[{md.get('role','unknown')} @ {md.get('timestamp','')}] {t}" for t, md in zip(docs, metadatas))

# -------------------- PII GUARDRAILS -------------------- #
# 1. SSN
SSN_REGEX = re.compile(r"\b(?!000|666|9\d\d)(\d{3})-(?!00)(\d{2})-(?!0000)(\d{4})\b")
def mask_ssn(text: str) -> str: return SSN_REGEX.sub("XXX-XX-XXXX", text)

# 2. Credit Card
def _digits_only(s: str) -> str: return re.sub(r"\D", "", s)
def _luhn_checksum(num: str) -> bool:
    total = 0
    rev = num[::-1]
    for i, ch in enumerate(rev):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return total % 10 == 0

CC_CANDIDATE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
def mask_credit_cards(text: str) -> str:
    def _replacer(m):
        raw, digits = m.group(0), _digits_only(m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_checksum(digits):
            return "****-****-****-" + digits[-4:]
        return raw
    return CC_CANDIDATE.sub(_replacer, text)

# 3. Phone Numbers
PHONE_REGEX = re.compile(r"(?x)(?:\+\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}")
def mask_phone_numbers(text: str) -> str:
    def _mask(m):
        s = m.group(0)
        ccode = re.match(r"^\+(\d{1,3})", s)
        prefix = f"+{ccode.group(1)} " if ccode else ""
        return f"{prefix}XXX-XXX-XXXX"
    return PHONE_REGEX.sub(_mask, text)

# 4. Email Addresses
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
def mask_emails(text: str) -> str:
    return EMAIL_REGEX.sub("[EMAIL REDACTED]", text)

# 5. Combined wrapper
def mask_pii(text: str) -> str:
    if not text: return text
    return mask_emails(mask_phone_numbers(mask_credit_cards(mask_ssn(text))))

# -------------------- LLM ORCHESTRATION -------------------- #
def make_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the provided document context and relevant past conversation memory to answer the question."),
        ("user", "Memory:\n{memory}\n\nContext:\n{context}\n\nQuestion: {question}")
    ])

def answer_query(question: str, doc_collection, memory: ConversationMemory, ctx_k: int = 3) -> (str, Dict[str,Any]):
    tracker = TokenTracker(MODEL_NAME)
    memory_text = memory.retrieve_relevant(question, top_k=MEMORY_TOP_K)
    context_text = retrieve_context(question, doc_collection, n_results=ctx_k)
    model = init_chat_model(MODEL_NAME, model_provider=MODEL_PROVIDER)
    prompt = make_prompt()
    compiled = prompt.format_prompt(memory=memory_text, context=context_text, question=question)
    tracker.add_prompt(*(m.content for m in compiled.messages))
    response = (prompt | model).invoke({"memory": memory_text, "context": context_text, "question": question})
    raw_answer = getattr(response, 'content', '') or ''
    tracker.add_completion(raw_answer)
    safe_output = mask_pii(raw_answer)
    return safe_output, tracker.as_dict()

# -------------------- MAIN -------------------- #
if __name__ == "__main__":
    docs = load_documents(DATA_FILE)
    chunks = split_documents(docs)
    doc_collection = init_chroma_collection(DOCS_COLLECTION, CHUNK_STORE_PATH)
    index_chunks(chunks, doc_collection)
    memory = ConversationMemory(CHUNK_STORE_PATH, MEMORY_COLLECTION, SESSION_ID, TRANSCRIPT_FILE)

    q1 = "What do keybullet kin drop?"
    memory.add_turn("user", q1)
    a1, t1 = answer_query(q1, doc_collection, memory)
    memory.add_turn("assistant", a1)
    print("Q1:", q1, "\nA1:", a1, "\nTokens:", t1)

    q2 = "Are they the same drops as you mentioned earlier? Also my email is test.user@example.com"
    memory.add_turn("user", q2)
    a2, t2 = answer_query(q2, doc_collection, memory)
    memory.add_turn("assistant", a2)
    print("Q2:", q2, "\nA2:", a2, "\nTokens:", t2)
