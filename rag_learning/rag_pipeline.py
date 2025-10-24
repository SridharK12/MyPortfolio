#!/usr/bin/env python
# coding: utf-8

"""
RAG Pipeline with ChromaDB + LangChain + OpenAI/HuggingFace embeddings
Author: Sridhar
"""

import os
import uuid
import logging
import pandas as pd
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from chromadb.utils import embedding_functions
from langchain_openai import OpenAIEmbeddings

# -------------------- CONFIG -------------------- #
DATA_FILE = os.getenv("RAG_SOURCE_FILE", "./documents.csv")
CHUNK_STORE_PATH = os.getenv("CHROMA_STORE_PATH", "./chroma_store")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "my_collection")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------- FUNCTIONS -------------------- #

def load_documents(csv_file: str) -> list[Document]:
    """Load documents from CSV into LangChain Document objects."""
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    df = pd.read_csv(csv_file)
    docs = [
        Document(page_content=str(row["text"]), metadata={"source": str(row["source_url"])})
        for _, row in df.iterrows()
    ]
    logging.info(f"Loaded {len(docs)} documents from {csv_file}")
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    """Split documents into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    chunks = splitter.split_documents(docs)
    logging.info(f"Split into {len(chunks)} chunks")
    return chunks


def init_chroma(collection_name: str, persist_path: str):
    """Initialize ChromaDB persistent client and collection."""
    client = chromadb.PersistentClient(path=persist_path)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )
    return collection


def index_chunks(chunks: list[Document], collection):
    """Index chunks into Chroma collection."""
    df = pd.DataFrame([
        {"id": str(uuid.uuid4()), "source": doc.metadata["source"], "chunk": doc.page_content}
        for doc in chunks
    ])
    collection.add(
        documents=df["chunk"].tolist(),
        ids=df["id"].tolist(),
        metadatas=df[["source"]].to_dict("records")
    )
    logging.info(f"Indexed {len(df)} chunks into ChromaDB")


def retrieve(query: str, collection, n_results: int = 3) -> str:
    """Retrieve context for a query from Chroma."""
    results = collection.query(query_texts=[query], n_results=n_results)
    contexts = results.get("documents", [[]])[0]
    return "\n\n".join(contexts)


def get_answer(query: str, context: str) -> str:
    """Run LLM with context + query prompt."""
    model = init_chat_model("gpt-4o-mini", model_provider="openai")
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer only from the provided context. "
                   "If the answer is not in context, reply with 'I don’t know'."),
        ("user", "Context: {context}\n\nQuestion: {question}")
    ])
    chain = template | model
    response = chain.invoke({"context": context, "question": query})
    return response.content

# -------------------- MAIN PIPELINE -------------------- #

if __name__ == "__main__":
    try:
        # Step 1: Load and preprocess
        docs = load_documents(DATA_FILE)
        chunks = split_documents(docs)

        # Step 2: Initialize vector store
        collection = init_chroma(COLLECTION_NAME, CHUNK_STORE_PATH)

        # Step 3: Index chunks
        index_chunks(chunks, collection)

        # Step 4: Run a query
        query = "What do keybullet kin drop?"  # example query
        context_text = retrieve(query, collection)
        answer = get_answer(query, context_text)

        logging.info(f"Q: {query}")
        logging.info(f"A: {answer}")

    except Exception as e:
        logging.error(f"Error in RAG pipeline: {e}", exc_info=True)
