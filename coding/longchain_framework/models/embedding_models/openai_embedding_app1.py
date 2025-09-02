from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS,Chroma,Pinecone
import util
# 1. Load your file
loader = TextLoader("testFile.txt")  # replace with your file path
documents = loader.load()

# 2. Create embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 3. Store embeddings in FAISS
#vectorstore = FAISS.from_documents(documents, embeddings)
#or 
croma_vector_store = Chroma.from_documents(documents, embeddings)

# 4. Your query
query = "what is the name of a person who did MCA education"

# 5. Search relevant docs
docs = croma_vector_store.similarity_search(query, k=3) # k - How many of the most relevant documents (based on vector similarity) you want to retrieve.

# 6. Pass retrieved docs to LLM
llm = OpenAI(temperature=0,max_tokens=40)
final_answer = llm.invoke("Answer this question based on the following text:\n\n"
                          + "\n\n".join([d.page_content for d in docs])
                          + f"\n\nQuestion: {query}")

# 7. Print answer
print("Answer:", final_answer)



####################################################################

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

result = embedding.embed_documents(documents)

print(str(result))
