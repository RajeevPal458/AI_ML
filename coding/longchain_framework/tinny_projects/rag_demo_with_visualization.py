# -------------------------
# 📌 LangChain RAG Example with Chart Visualization
# -------------------------

# --- 1. Imports ---
# LangChain model & embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# For prompts
from langchain.prompts import ChatPromptTemplate
# For document loading & splitting
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Vector database
from langchain_community.vectorstores import Chroma
# LCEL (LangChain Expression Language) utilities
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
# Chart plotting
import matplotlib.pyplot as plt
# To store API key securely
import os
import util

# --- 2. Set API key ---
# Replace 'your-openai-api-key' with your real key or set via environment variable
#this step is done in util file

# --- 3. Load the model ---
# ChatOpenAI: a chat-optimized LLM (using GPT-4o-mini here for cost/speed)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=50  # limit ~40 wordss
)
# --- 4. Create a prompt template ---
# This template has two placeholders: {question} and {context}
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use ONLY the provided context to answer."),
    ("human", "{question}\n\nContext:\n{context}")
])

# --- 5. Load the document ---
loader = TextLoader("testFile.txt")
docs = loader.load()  # List[Document]


# --- 6. Split into chunks ---
# RecursiveCharacterTextSplitter tries to break text at paragraph/sentence boundaries
splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
chunks = splitter.split_documents(docs)

# --- 📊 Chart 1: Chunk length distribution ---
chunk_lengths = [len(doc.page_content) for doc in chunks]
plt.bar(range(len(chunk_lengths)), chunk_lengths)
plt.title("Chunk Lengths")
plt.xlabel("Chunk Index")
plt.ylabel("Number of Characters")
plt.show()

# --- 7. Create embeddings & store in vector DB ---
embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(chunks, embeddings)

# --- 8. Create retriever ---
# Retriever finds top-k similar chunks for a given query
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# --- 9. Define a helper to format context ---
def format_context(query):
    # Get relevant docs
    relevant_docs = retriever.get_relevant_documents(query)
    # Join their content into one string
    return "\n\n".join(d.page_content for d in relevant_docs)

# --- 📊 Chart 2: Query relevance heatmap ---
# For visualization, let's embed each chunk and query, then plot similarities
import numpy as np
query = "whatv is vector search"
query_vec = embeddings.embed_query(query)
chunk_vecs = [embeddings.embed_query(doc.page_content) for doc in chunks]
similarities = [np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec)) for vec in chunk_vecs]
plt.bar(range(len(similarities)), similarities)
plt.title("Query-Chunks Similarity")
plt.xlabel("Chunk Index")
plt.ylabel("Cosine Similarity")
plt.show()

# --- 10. Build RAG chain ---
rag_chain = (
    {"question": RunnablePassthrough(), "context": RunnableLambda(format_context)}
    | prompt
    | llm
)

# --- 11. Run a query ---
question = "what is semantic search"
answer = rag_chain.invoke(question)

# --- 12. Print the answer ---
print("\nQUESTION:", question)
print("ANSWER:", answer.content)
