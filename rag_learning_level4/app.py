# rag_learning_level4/app.py
import streamlit as st
from rag_generator import generate_answer

st.set_page_config(page_title="📘 Chemistry RAG QA System", layout="wide")

st.title("🧠 Chemistry RAG-Based Question Answering System")
st.write("Ask any question related to your chemistry document! 🔬")

# --- Text input ---
query = st.text_area("Enter your question:", placeholder="e.g. Explain the process of distillation")

# --- Ask button ---
if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking... 🔍"):
            result = generate_answer(query)

        if result and result.get("answer"):
            st.subheader("💡 Answer")
            st.write(result["answer"])

            # --- Show Citations ---
            if "citations" in result and result["citations"]:
                st.markdown("---")
                st.subheader("📚 Citations (Retrieved Chunks)")
                for idx, cite in enumerate(result["citations"], start=1):
                    with st.expander(f"🔹 Citation {idx} — Page {cite.get('page', '?')} | {cite.get('type', 'text').upper()}"):
                        st.markdown(f"**Score:** {cite.get('score', 'N/A'):.3f}")
                        st.write(cite.get("content", ""))
                        if cite.get("type") == "image" and cite.get("local_path"):
                            st.image(cite["local_path"], caption=f"Image from Page {cite.get('page', '?')}")
            else:
                st.info("No supporting citations found.")
        else:
            st.error("No answer could be generated.")
