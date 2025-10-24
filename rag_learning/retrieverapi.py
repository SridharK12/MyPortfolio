# app.py
from fastapi import FastAPI, Query
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
from retrieve_utils.retrieve_from_collection import retrieve_from_collection

app = FastAPI(
    title="Chroma Retrieval API",
    version="1.0.0",
    description="API to retrieve most similar documents from a Chroma vector collection"
)

# Initialize embeddings (you can replace with your own)
embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")

# Request model
class RetrievalRequest(BaseModel):
    collection_name: str
    persist_directory: str
    query: str
    top_n: int = 3

@app.post("/retrieve")
def retrieve_documents(req: RetrievalRequest):
    """
    Retrieve top-k most similar documents from a Chroma collection.
    """
    result_text = retrieve_from_collection(
        collection_name=req.collection_name,
        embedding_function=embedding_function,
        persist_directory=req.persist_directory,
        query=req.query,
        top_n=req.top_n,
        retrievek = 10        
    )
    return {"result": result_text}
