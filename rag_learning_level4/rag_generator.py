# rag_generator.py
import logging
from retriever_with_reranker import retrieve_with_rerank
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# === CONFIGURATION ===
TOP_K_FINAL = 5
MODEL_NAME = "gpt-4o-mini"
MAX_CONTEXT_LENGTH = 2500

# === LLM INITIALIZATION ===
llm = ChatOpenAI(model=MODEL_NAME, temperature=0.3)

# === PROMPT TEMPLATE ===
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert chemistry tutor. Use the provided context (including equations and images) to answer clearly and factually. Mention figures when relevant."),
    ("human", """Question: {question}

Context:
{context}

Provide a concise and scientifically accurate answer. If figures or diagrams are relevant, mention them explicitly (e.g., "as shown in Figure on page 12").""")
])

# === CONTEXT BUILDER ===
def build_context(results):
    """Combine retrieved chunks (text, equations, images) into a single string."""
    context_parts = []
    for item in results:
        snippet = item["content"].strip()
        entry = f"[Page {item.get('page', '?')} | Type: {item.get('type', 'text').upper()}]\n{snippet}"
        if item["type"] == "image" and item.get("local_path"):
            entry += f"\n(Image file: {item['local_path']})"
        context_parts.append(entry)
    context = "\n\n".join(context_parts)
    if len(context) > MAX_CONTEXT_LENGTH:
        context = context[:MAX_CONTEXT_LENGTH] + "..."
    return context

# === GENERATOR FUNCTION ===
def generate_answer(query):
    logging.info(f"🧠 Generating answer for query: {query}")

    results = retrieve_with_rerank(query, top_k_final=TOP_K_FINAL)
    if not results:
        return {"answer": None, "citations": [], "images": []}

    # Build text context
    context = build_context(results)

    # Query LLM
    messages = prompt_template.format_messages(question=query, context=context)
    response = llm.invoke(messages)
    answer = response.content.strip()

    # Collect citation info
    citations = []
    images = []
    for item in results:
        citations.append({
            "page": item.get("page"),
            "type": item.get("type"),
            "score": item.get("score"),
            "content": item.get("content"),
            "local_path": item.get("local_path")
        })
        if item["type"] == "image" and item.get("local_path"):
            images.append({
                "page": item.get("page"),
                "path": item["local_path"],
                "caption": item["content"],
                "score": item["score"]
            })

    return {"answer": answer, "citations": citations, "images": images}
