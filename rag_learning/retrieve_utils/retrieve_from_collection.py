# retrieve_utils.py
from typing import Any
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder

def retrieve_from_collection(
    collection_name: str,
    embedding_function: Any,
    persist_directory: str,
    query: str,
    top_n: int,
    retrievek: int = 10
) -> str:

    """
    Retrieve top-retrievek similar documents from a Chroma vector store.
    Returns top_n documents after re ranking
    """
    top_n=3
    try:
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=persist_directory
        )

        results = vectorstore.similarity_search(query, k=k)
        if not results:
            return "No relevant documents found."
        
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2') #Initiate re ranker
    
        pairs = [(query, doc.page_content) for doc in results]
    
        scores = reranker.predict(pairs)
    
        ranked_results = sorted(
                        zip(results, scores), 
                        k = lambda x:x[1],
                        reverse = True
                        )
    
        top_docs = [doc.page_content for doc, _ in ranked_results[:top_n]]
    
        return "\n".join(top_docs)

    except Exception as e:
        return f"Error retrieving from collection: {str(e)}"
