from langchain_core.tools import StructuredTool 
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings
import retrieve_from_collection

# Initialize embeddings
embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")

# Define schema for tool input (LangChain standard)
class RetrieveInput(BaseModel):
    collection_name: str = Field(..., description="Name of the Chroma collection")
    persist_directory: str = Field(..., description="Path to Chroma persist directory")
    query: str = Field(..., description="The search query")
    top_n: int = Field(3, description="Number of top documents to return after re-ranking")
    retrievek: int = Field(10, description="Number of candidates to retrieve before re-ranking")

# Create StructuredTool
retrieve_tool = StructuredTool.from_function(
    func=lambda collection_name, persist_directory, query, top_n=3, retrievek=10: retrieve_from_collection(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory,
        query=query,
        top_n=top_n,
        retrievek=retrievek,
    ),
    name="retrieve_from_chroma",
    description="Retrieve documents from a Chroma vector store and re-rank them using a cross-encoder.",
    args_schema=RetrieveInput,
)
