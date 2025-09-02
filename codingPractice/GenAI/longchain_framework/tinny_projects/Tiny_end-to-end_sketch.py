# MODELS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, AgentExecutor, create_tool_calling_agent
import util


# ---- LLM ----
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=50  # limit ~40 words
)

# ---- PROMPT for plain RAG ----
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that uses provided context."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# ---- INDEX: load → split → embed → store ----
loader = Docx2txtLoader(
    r"C:\DriveD\projects\gitprojects\codingInterviewPreparation\learning\AI_ML\completeCourse\MachineLearning\GenerativeAI(GenAI)\LangChain\testFile.docx"
)
docs = loader.load()

chunks = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
).split_documents(docs)

vec = Chroma.from_documents(chunks, OpenAIEmbeddings())
retriever = vec.as_retriever(search_kwargs={"k": 4})

# ---- RAG CHAIN ----
def format_context(q):
    ctx = retriever.get_relevant_documents(q)
    return "\n\n".join(d.page_content for d in ctx)

rag_chain = (
    {"question": RunnablePassthrough(), "context": RunnableLambda(format_context)}
    | prompt
    | llm
)

# ---- MEMORY ----
memory = ConversationBufferMemory(return_messages=True)

# ---- AGENT SETUP ----
tools = [
    Tool(
        name="doc_search",
        func=lambda q: format_context(q),
        description="Searches the project notes for relevant info"
    )
]

# FIX: must include {agent_scratchpad} for create_tool_calling_agent
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful. Use tools when needed."),
    ("human", "{input}\n\n{agent_scratchpad}")
])

agent = create_tool_calling_agent(llm, tools, agent_prompt)
agent_exec = AgentExecutor(agent=agent, tools=tools, verbose=False)

# ---- USAGE ----
answer = rag_chain.invoke("What did we decide about Q3 targets?")
print("RAG answer:", answer.content)

agent_answer = agent_exec.invoke({"input": "what is semantic search"})
print("Agent answer:", agent_answer["output"])
#If you want, I can turn this into a starter repo layout (env, file structure, requirements) or swap in Pinecone/FAISS and a summary memory for longer chats.
