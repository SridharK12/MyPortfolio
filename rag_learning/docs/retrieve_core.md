# 🧠 Core Logic Documentation — `retrieve_top_docs()`

**Module:** `retrieval_core.py`  
**Current Version:** `v1.0.0`  
**Maintainer:** AI Retrieval Team  
**Last Updated:** 24-Oct-2025  

---

## 📘 1. Overview

The `retrieve_top_docs()` function is the **core retrieval logic** used by both:
- the **FastAPI layer** (`retriever_api.py`) for HTTP access, and  
- the **MCP tool layer** (`chroma_retriever_tool.py`) for internal agent/tool integration.  

It retrieves documents semantically similar to a user query from a **Chroma vector database**, and then **re-ranks** them using a **cross-encoder model** to improve result quality.  

---

## ⚙️ 2. Functional Summary

| Function Name | `retrieve_top_docs` |
|----------------|----------------------|
| Purpose | Retrieve top-k relevant documents using vector similarity, re-rank them, and return top-N best matches |
| Input Source | Chroma vector store |
| Re-ranking Model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Embedding Function | `OpenAIEmbeddings` (or any compatible LangChain embedding function) |
| Interfaces Using This | `FastAPI` (`/retrieve` endpoint), `MCP Tool` (`retrieve_from_chroma`) |

---

## 🧩 3. Function Signature

```python
def retrieve_top_docs(
    collection_name: str,
    embedding_function: Any,
    persist_directory: str,
    query: str,
    k: int = 10,
    top_n: int = 3
) -> str:
    """
    Retrieve top-k documents from a Chroma collection, re-rank using a cross-encoder, return top_n.
    """
```

---

## 🧠 4. Detailed Parameter Reference

| Parameter | Type | Required | Default | Description |
|------------|------|-----------|----------|--------------|
| `collection_name` | `str` | ✅ | — | The name of the Chroma collection to query |
| `embedding_function` | `Any` | ✅ | — | Embedding model used to generate vector representations (e.g. `OpenAIEmbeddings`) |
| `persist_directory` | `str` | ✅ | — | Directory path where the Chroma vector store is persisted |
| `query` | `str` | ✅ | — | User’s natural language query |
| `k` | `int` | ❌ | `10` | Number of nearest documents to retrieve before re-ranking |
| `top_n` | `int` | ❌ | `3` | Number of top documents to return after re-ranking |

---

## 🔄 5. Processing Flow

```mermaid
flowchart TD
    A[Start: receive query & parameters] --> B[Load Chroma vectorstore]
    B --> C[Perform similarity_search (top k docs)]
    C --> D[Pair each doc with query]
    D --> E[Apply CrossEncoder for re-ranking]
    E --> F[Sort by score descending]
    F --> G[Select top_n docs]
    G --> H[Return concatenated text as string]
    H --> I[End]
```

---

## 🧮 6. Return Value

| Type | Description |
|------|--------------|
| `str` | Concatenated text of top re-ranked documents, separated by newlines (`\n`). |
| On Error | Returns a string with an error message (prefixed with `"Error retrieving from collection:"`) |

---

## ⚠️ 7. Error Handling

| Error Scenario | Behavior |
|----------------|-----------|
| Chroma collection not found | Returns `"Error retrieving from collection: <details>"` |
| Empty result set | Returns `"No relevant documents found."` |
| Model or file error | Captured in `Exception` block and returned as error string |

---

## 🧪 8. Example Usage

### Example 1 — Direct Function Call
```python
from langchain_openai import OpenAIEmbeddings
from retrieval_core import retrieve_top_docs

embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")

result = retrieve_top_docs(
    collection_name="finance_docs",
    embedding_function=embedding_function,
    persist_directory="./chroma_store",
    query="Explain the concept of derivative trading",
    k=10,
    top_n=3
)

print(result)
```

**Output:**
```
A derivative is a financial instrument whose value is derived from...
Derivatives are used to hedge risk or for speculation...
Common derivatives include futures, options, and swaps...
```

---

## 📈 9. Performance Notes

| Step | Approx Time (avg) | Remarks |
|------|--------------------|----------|
| Chroma `similarity_search(k=10)` | 50–150 ms | Depends on collection size |
| CrossEncoder re-ranking | 200–400 ms | CPU-intensive; GPU recommended |
| Total Latency | ~300–600 ms | Tunable by reducing `k` or using a faster reranker |

---

## 🧰 10. Dependencies

| Library | Purpose | Recommended Version |
|----------|----------|---------------------|
| `langchain_community.vectorstores.Chroma` | Vector store interface | ≥ 0.1.0 |
| `sentence_transformers.CrossEncoder` | Re-ranking model | ≥ 2.2.2 |
| `openai` or `langchain_openai` | Embedding generation | ≥ 1.0.0 |

---

## 🔐 11. Security & Access

- No external network calls beyond model loading.  
- Ensure Chroma directory access is restricted to authorized service roles.  
- Embedding model API keys (if OpenAI) must be securely stored via environment variables or AWS Secrets Manager.  

---

## 🧭 12. Versioning & Compatibility

| Version | Date | Change Summary |
|----------|------|----------------|
| `1.0.0` | 24-Oct-2025 | Initial implementation supporting Chroma retrieval and cross-encoder re-ranking |

---

## 🧑‍💻 13. Extension Guidelines

If enhancements are needed:
- Add new parameters as **optional** (maintains backward compatibility)
- For breaking changes:
  - Create `retrieve_top_docs_v2()` in a `v2` submodule  
  - Increment major version to `2.0.0`
- Always update:
  - Function docstring  
  - This Markdown doc  
  - Changelog  
  - API & MCP tool descriptions
