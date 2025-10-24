import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import pandas as pd
import pandas as pd
import chromadb

chunks_df=pd.read_csv('I:\\Sridhar\\rag_learning\\chunked_docs.csv')
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
'''
base_dir = "./chroma_langchain_db"
os.makedirs(base_dir, exist_ok=True)

for source, group_df in chunks_df.groupby("source"):
    print(f"\n📘 Creating collection for source: {source}")

    # Each source becomes its own collection
    collection_dir = os.path.join(base_dir, source)

    vectorstore = Chroma(
        collection_name=source,                # ✅ dynamic collection name
        embedding_function=embeddings,
        persist_directory=collection_dir       # store separately for clarity
    )

    # Add all text entries for this source
    texts = group_df["chunk"].tolist()
    metadatas = [{"source": source} for _ in texts]

    vectorstore.add_texts(texts=texts, metadatas=metadatas)    
    print(f"✅ Added {len(texts)} documents to collection '{source}'")

print("\n🎉 All collections created and persisted successfully!")
'''
#query='Explain benefits of artificial intelligence'
vectorstore = Chroma(
    collection_name="Europa_Commission",               # 👈 collection name
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db/Europa_Commission"  # 👈 path to that collection
)

# Step 3: Perform semantic search
results = vectorstore.similarity_search('Explain benefits of artificial intelligence', k=3)
