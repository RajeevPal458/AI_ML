# -------------------------
# 📌 LangChain RAG Chatbot with Memory & Similarity Timeline
# -------------------------

# --- 1. Imports ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, AgentExecutor, create_tool_calling_agent
import matplotlib.pyplot as plt
import numpy as np
import os
import util

# --- 2. Set API key ---
# done with util file

# --- 3. Load the model ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- 4. Create a prompt template ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use ONLY the provided context to answer."),
    ("human", "{question}\n\nContext:\n{context}")
])

# --- 5. Load and split documents ---
loader = TextLoader("testFile.txt")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
chunks = splitter.split_documents(docs)

# --- 📊 Chart 1: Chunk length distribution ---
chunk_lengths = [len(doc.page_content) for doc in chunks]
plt.bar(range(len(chunk_lengths)), chunk_lengths)
plt.title("Chunk Lengths")
plt.xlabel("Chunk Index")
plt.ylabel("Number of Characters")
plt.show()

# --- 6. Create embeddings & store in vector DB ---
embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# --- 7. Helper for retrieving context ---
def format_context(query):
    relevant_docs = retriever.get_relevant_documents(query)
    return "\n\n".join(d.page_content for d in relevant_docs)

# --- 8. Memory setup ---
memory = ConversationBufferMemory(return_messages=True)

# --- 9. Track similarity changes over chat ---
similarity_timeline = []  # list of lists (per question)

def log_similarity(query):
    query_vec = embeddings.embed_query(query)
    chunk_vecs = [embeddings.embed_query(doc.page_content) for doc in chunks]
    similarities = [np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec)) for vec in chunk_vecs]
    similarity_timeline.append(similarities)

# --- 10. Build RAG chain ---
rag_chain = (
    {"question": RunnablePassthrough(), "context": RunnableLambda(format_context)}
    | prompt
    | llm
)

# --- 11. Chat loop ---
questions = [
    "semantic search?",
    "vector search?",
    "what is the education of rajeev?"
]

for q in questions:
    log_similarity(q)  # log similarity for chart
    answer = rag_chain.invoke(q)
    memory.chat_memory.add_user_message(q)
    memory.chat_memory.add_ai_message(answer.content)
    print(f"\nUSER: {q}")
    print(f"BOT: {answer.content}")

# --- 📊 Chart 2: Similarity timeline ---
similarity_array = np.array(similarity_timeline)  # shape: (num_questions, num_chunks)
plt.figure(figsize=(8, 5))
for chunk_idx in range(similarity_array.shape[1]):
    plt.plot(similarity_array[:, chunk_idx], marker='o', label=f"Chunk {chunk_idx}")
plt.title("Chunk Similarity Over Conversation")
plt.xlabel("Question Number")
plt.ylabel("Cosine Similarity")
plt.legend()
plt.show()
