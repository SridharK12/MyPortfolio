#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv("I:\\sridhar\\rag_learning\\students_table_with_unique_remarks.csv")
print(df.shape)


# In[2]:


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


# In[3]:


df.head()


# In[4]:


docs = [
    Document(page_content=str(row["remarks"]), metadata={"student": str(row["student_name"])})
    for _, row in df.iterrows()
]


# In[5]:


### Practice for iterating through a pandas data frame
for index, row in df.iterrows():
    print(index, row['roll_number'])


# In[6]:


df.iloc[0]


# In[7]:


for i, col in enumerate(df.columns):
    print(i, col)


# In[8]:


'''
Document(
    page_content="The actual text of the document",
    metadata={"source": "example.csv", "chunk": 1}
)
'''


# In[9]:


'''
from langchain_core.documents import Document

for i, chunk in enumerate(chunks):
        docs.append(
            Document(
                page_content=chunk,
                metadata={
                    "source_url": source,
                    "row_index": idx,   # keep track of original row
                    "chunk": i
                }
            )
        )
'''


# In[10]:


import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document   # or from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings  # or HuggingFaceEmbeddings


# In[11]:


df = pd.read_csv("I:\\Sridhar\\rag_learning\\documents.csv")


# In[12]:


df.shape


# In[13]:


docs = [
    Document(page_content=str(row["text"]), metadata={"source": str(row["source_url"])})
    for _, row in df.iterrows()
]


# In[14]:


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)


# In[15]:


'''
split_documents takes a list of Document objects as returns a list of Document objects
the document object consists of page_content, metadata
split_documents(
    documents: List[Document]
) -> List[Document]
'''


# In[16]:


chunked_docs = splitter.split_documents(docs)


# In[17]:


'''
We are creating a dictionary of chunks with meta data and preparing a dataframe
'''
chunk_data = []
for i, doc in enumerate(chunked_docs):
    chunk_data.append({
        "source": doc.metadata["source"],
        "chunk_index": i,
        "chunk": doc.page_content
    })

chunk_df = pd.DataFrame(chunk_data)
print(chunk_df.head(10))  # first 10 chunks


# In[18]:


chunk_df.to_csv("I:\\Sridhar\\rag_learning\\chunked_docs.csv")


# In[22]:


#import getpass
#import os

#if not os.environ.get("OPENAI_API_KEY"):
#  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

#from langchain_openai import OpenAIEmbeddings

#embeddings = OpenAIEmbeddings(model="text-embedding-3-small")




# In[3]:


from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# In[24]:


#get_ipython().system('pip  install Chromadb')


# In[30]:


import chromadb
from chromadb.utils import embedding_functions

embedding_functions.SentenceTransformerEmbeddingFunction
client = chromadb.PersistentClient(path="./chroma_store")  # persists to disk
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"  # fast & small
)


# In[29]:


#get_ipython().system('pip install sentence-transformers')


# In[31]:


# Create or load collection
collection = client.get_or_create_collection(
    name="my_collection",
    embedding_function=embedding_fn
)


# In[32]:


import uuid

# Generate UUIDs for each row
chunk_df["id"] = [str(uuid.uuid4()) for _ in range(len(chunk_df))]

# Only use "source" column as metadata
collection.add(
    documents=chunk_df["chunk"].tolist(),             # text chunks
    ids=chunk_df["id"].tolist(),                     # unique UUIDs
    metadatas=chunk_df[["source"]].to_dict("records")  # only 'source'
)

print("✅ Added chunks with only 'source' as metadata")


# In[57]:


#query = "Tell me about the first chunk"
#results = collection.query(
#    query_texts=[query],
#    n_results=3,
#    where={"source": "https://enterthegungeon.fandom.com/wiki/Bullet_Kin"}   # ✅ filter by metadata
#)
query = "What do keybullet kin drop?"
results = collection.query(
    query_texts=[query],
    n_results=3
)

print(results)


# In[58]:


contexts = results["documents"][0]


# In[59]:


context_text = "\n\n".join(contexts)


# In[60]:


print(context_text)


# In[61]:


from langchain.chat_models import init_chat_model
model = init_chat_model("gpt-4o-mini", model_provider="openai")


# In[62]:


#question="Please let me know what is a unix shell script"


# In[63]:


#context="Please refer Medication section"
from langchain.prompts import ChatPromptTemplate

# Define a chat prompt template with context and user question
template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant to answer my questions. Answer the question only from context. If answer is not found in context say I dont know"),
    ("user", "Context: {context_text}\n\nQuestion: {question}")
])


# In[65]:


chain = template | model


# In[68]:


response = chain.invoke({
    "context_text": context_text,
    "question": query
})


# In[69]:


print(response.content)

