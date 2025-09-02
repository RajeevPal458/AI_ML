from langchain_community.document_loaders import Docx2txtLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import util
# Load docs
loader = Docx2txtLoader(r"C:\DriveD\projects\gitprojects\codingInterviewPreparation\learning\AI_ML\completeCourse\MachineLearning\GenerativeAI(GenAI)\LangChain\5.Indexes_(RAG_plumbing).docx")
docs = loader.load()

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(docs, embeddings)

# Chat model with token + word limit
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=50  # ~40 words
)

# Restrict retriever output
retriever = db.as_retriever(search_kwargs={"k": 2})

# Create retrieval chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Run query
response = qa.run("Answer in under 40 words: How to read PDF file using OpenAI?")
print(response)
