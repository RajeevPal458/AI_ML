from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

text = "Delhi is the capital of India"

vector = embedding.embed_query(text)

print(str(vector))


#========================

from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

vector = embedding.embed_documents(documents)


#################################################


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

# 1️⃣ Load Hugging Face embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2️⃣ Read text file
with open("testFile.txt", "r", encoding="utf-8") as f:
    text_data = f.read().splitlines()

# 3️⃣ Store embeddings in Chroma vector store
persist_dir = "chroma_db"
vector_store = Chroma.from_texts(text_data, embedding, persist_directory=persist_dir)
vector_store.persist()

# 4️⃣ Search relevant docs for a query
query = "What is the education of rajeev"
docs = vector_store.similarity_search(query, k=2)

# 5️⃣ Prepare context
context = "\n".join([doc.page_content for doc in docs])

# 6️⃣ Load a local model for answering
hf_pipe = pipeline("text-generation", model="distilgpt2", max_length=100)
llm = HuggingFacePipeline(pipeline=hf_pipe)

# 7️⃣ Ask the model using retrieved context
prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
answer = llm(prompt)

print("\n--- Retrieved Answer ---\n")
print(answer)
