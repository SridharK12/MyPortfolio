#https://www.kaggle.com/datasets/samuelmatsuoharris/single-topic-rag-evaluation-dataset
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain.schema import Document
from langchain_core.documents import Document
df=pd.read_csv("I:\\sridhar\\rag_learning\\documents.csv")
#print(df.shape)
#print(df.head())
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # adjust for your embedding model
    chunk_overlap=200
)

docs = []

for idx, row in df.iterrows():
    text = str(row.get("text", ""))         # Initialize to empty string if no text present
    source = str(row.get("source_url", "")) # Initialize to empty string if no text present
    chunks=splitter.split_text(text)
    print(len(chunks))
    for i, chunk in enumerate(chunks):
        docs.append({
            "row_index":  idx,
            "chunk_index": i,
            "source": row.get("source_url"),
            "chunk": chunk
            })
chunks_df=pd.DataFrame(docs)
print(chunks_df.shape)

'''        
docs = [
    Document(page_content=str(row["text"]), metadata={"source": str(row["source_url"])})
    for _, row in df.iterrows()
]

chunked_docs = splitter.split_documents(docs)

chunk_data = []
for i, doc in enumerate(chunked_docs):
    chunk_data.append({
        "source": doc.metadata["source"],
        "chunk_index": i,
        "chunk": doc.page_content
    })

chunk_df = pd.DataFrame(chunk_data)
'''
print(chunks_df.head(10))  # first 10 chunks
chunks_df.to_csv("I:\\Sridhar\\rag_learning\\chunked_docs.csv")