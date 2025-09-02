"""
RAG Chatbot (interactive) with Conversation Memory + Live Similarity Timeline
Run: python rag_chat_interactive.py
"""

# -------------------------
# 1) Imports
# -------------------------
# langchain imports for LLMs, embeddings, chains, memory, loaders, splitters, vectorstore
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# plotting and numeric tools
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import util
# -------------------------
# Explanation (Imports)
# -------------------------
# ChatOpenAI    -> Chat-optimized LLM wrapper (calls OpenAI chat models)
# OpenAIEmbeddings -> turn text into numeric vectors (embeddings)
# TextLoader    -> loads plain text files into LangChain Document objects
# RecursiveCharacterTextSplitter -> splits long text into chunks
# Chroma        -> local vector DB (stores embeddings + metadata)
# ConversationalRetrievalChain -> ready-made chain: retrieval + LLM + memory
# ConversationBufferMemory -> keeps chat history (in RAM)
# matplotlib/numpy -> used for charts and similarity calculations

# -------------------------
# 2) API Key (expects env var)
# -------------------------
# The script uses the OPENAI_API_KEY environment variable.
# You can set it in your shell BEFORE running the script:
# export OPENAI_API_KEY="sk-..."  (Linux/macOS)
# setx OPENAI_API_KEY "sk-..."   (Windows)

# -------------------------
# 3) Load & split documents
# -------------------------
# Load the sample text file into a Document object
loader = TextLoader("testFile.txt")           # loads file -> list of Document(s)
docs = loader.load()                          # docs is a list; each item has .page_content

# Split long documents into smaller chunks suitable for embeddings + retrieval
splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,      # aim: ~120 characters per chunk (adjust to your needs)
    chunk_overlap=20     # keep some overlap so context isn't lost at boundaries
)
chunks = splitter.split_documents(docs)       # returns a list of smaller Document objects

# -------------------------
# 4) Plot chunk length distribution (one-time)
# -------------------------
# Show how documents were chunked (useful to inspect chunk sizes)
chunk_lengths = [len(d.page_content) for d in chunks]
# plt.figure(figsize=(8, 3))
# plt.bar(range(len(chunk_lengths)), chunk_lengths)
# plt.title("Chunk Lengths (characters)")
# plt.xlabel("Chunk index")
# plt.ylabel("Length (chars)")
# plt.tight_layout()
# plt.show(block=False)   # non-blocking show so script can continue
# time.sleep(1)           # short pause so the window renders on some platforms

# -------------------------
# 5) Create embeddings and vectorstore (Chroma)
# -------------------------
# Create an embeddings object (uses your OpenAI key)
embeddings = OpenAIEmbeddings()

# Create a Chroma vectorstore from the chunks: this computes embeddings and saves them locally
# If you re-run often during development, you might want to set persist_directory to keep data.
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

# Create a retriever from the vectorstore (this is what the chain will call)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # return top 3 chunks by default

# -------------------------
# 6) Memory & Conversational RAG chain
# -------------------------
# ConversationBufferMemory saves all user and AI messages for the session.
memory = ConversationBufferMemory(
                                  memory_key="chat_history",
                                  return_messages=True,
                                  output_key="answer"  # ✅ Tell memory which output to store
                                  )

# ConversationalRetrievalChain wires up:
# - retriever (vectorstore)
# - chat LLM
# - and optional memory automatically (it will include previous user/assistant messages)
llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")  # pick model you have access to
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True   # we will also show which docs were used
)

# -------------------------
# Explanation (Chain + Memory)
# -------------------------
# ConversationalRetrievalChain expects a dict like {"question": "..."} when invoked.
# It will:
# 1) use the retriever to fetch context chunks for the question
# 2) combine question + retrieved context + chat history (from memory)
# 3) call the LLM to produce an answer
# 4) store the turn in memory automatically

# -------------------------
# 7) Helpers: similarity logging + plotting setup
# -------------------------
# We'll compute cosine similarity between a query and each chunk's embedding so we can chart "which chunk matters"
def cosine_similarity(a, b):
    # a and b are 1D numpy vectors
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

# Precompute embeddings for chunks once to speed up similarity checks
chunk_vecs = [np.array(embeddings.embed_query(d.page_content)) for d in chunks]

# Data structure to store similarity timeline: list of arrays; one entry per question
similarity_timeline = []

# Setup the live plot for similarity timeline
#plt.ion()  # interactive mode on so we can update plots live
fig, ax = plt.subplots(figsize=(9, 5))

def update_similarity_plot(similarity_timeline):
    ax.clear()
    if len(similarity_timeline) == 0:
        ax.set_title("No queries yet - ask a question to see similarities")
        ax.set_xlabel("Query index")
        ax.set_ylabel("Cosine similarity")
        plt.pause(0.01)
        return

    sim_array = np.array(similarity_timeline)  # shape: (num_queries, num_chunks)
    num_queries, num_chunks = sim_array.shape

    # plot each chunk's similarity across queries as a line
    for chunk_idx in range(num_chunks):
        ax.plot(range(num_queries), sim_array[:, chunk_idx], marker='o', label=f"chunk {chunk_idx}")
    ax.set_title("Chunk Similarity Over Conversation")
    ax.set_xlabel("Query number (0 = first question)")
    ax.set_ylabel("Cosine similarity")
    ax.set_ylim(0, 1)  # cosine similarity range for normalized vectors is -1..1 but embeddings typically non-negative in practice
    ax.legend(loc='upper right', fontsize='small', ncol=2)
    fig.tight_layout()
    plt.draw()
    plt.pause(0.01)

# show initial empty plot
#update_similarity_plot(similarity_timeline)

# -------------------------
# 8) Interactive chat loop
# -------------------------
print("\n==== RAG Chat (type 'exit' or 'quit' to stop) ====\n")
while True:
    user_q = input("You: ").strip()
    if user_q.lower() in ("exit", "quit"):
        print("Goodbye!")
        break
    if user_q == "":
        continue

    # 1) Compute similarity between query and each chunk (for plotting)
    q_vec = np.array(embeddings.embed_query(user_q))
    similarities = [cosine_similarity(q_vec, cv) for cv in chunk_vecs]
    similarity_timeline.append(similarities)

    # 2) Update the live similarity plot
    #update_similarity_plot(similarity_timeline)

    # 3) Call the conversational retrieval chain
    # It accepts {"question": ..., "chat_history": ...} but memory is attached, so we just pass question
    result = chain({"question": user_q})

    # result is a dict like:
    # {"answer": "...", "source_documents": [...], "chat_history": [...] } (fields may vary)
    answer_text = result.get("answer") or result.get("output_text") or result
    source_docs = result.get("source_documents", [])

    # 4) Print the answer and which source docs were used
    print("\nBot:", answer_text, "\n")
    if source_docs:
        print("Top source chunks used:")
        for i, sd in enumerate(source_docs):
            snippet = sd.page_content.replace("\n", " ").strip()
            print(f"  [{i}] {snippet[:200]}{'...' if len(snippet) > 200 else ''}")
    else:
        print("No source documents were returned.")

    # 5) Show current conversation memory short summary (last few turns)
    # memory.chat_memory.return_messages() is available as memory.load_memory_variables({})
    mem_var = memory.load_memory_variables({})  # returns dict; key is "chat_history"
    # print last 4 messages (user or assistant)
    print("\nConversation (last messages):")
    chat_messages = mem_var.get("chat_history", [])
    # messages are objects; we will print type + text for clarity
    for i, msg in enumerate(chat_messages[-6:]):   # last 6 items
        # msg could be a ChatMessage object; msg.content has the text typically
        content = getattr(msg, "content", str(msg))
        role = getattr(msg, "type", None) or getattr(msg, "role", None) or "message"
        print(f"  [{i}] {role}: {content[:200]}{'...' if len(content) > 200 else ''}")

    print("\n--- Ready for next question ---\n")
