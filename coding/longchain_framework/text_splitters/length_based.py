from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import util


loader = PyPDFLoader('dl-curriculum.pdf')
pdfFileDoc = loader.load()

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

docs = splitter.split_documents(documents=pdfFileDoc)

vector_store = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory='my_chroma_db',
    collection_name='sample'
)

indexes = vector_store.add_documents(docs)

print("indexes: ",indexes)

for doc in docs:
    print("================")
    print(doc)

print("========================")

