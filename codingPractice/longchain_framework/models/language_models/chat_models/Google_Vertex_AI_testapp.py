from langchain_google_vertexai import ChatVertexAI
from langchain.schema import SystemMessage, HumanMessage
import util
# Requires GOOGLE_APPLICATION_CREDENTIALS for Vertex AI
chat = ChatVertexAI(model="gemini-1.5-flash", temperature=0.7)

messages = [
    SystemMessage(content="You are a friendly assistant."),
    HumanMessage(content="Give me a fun fact about space."),
]

response = chat.invoke(messages)
print("Google Vertex AI:", response.content)


##########################################################

from langchain.schema import SystemMessage, HumanMessage
from langchain.chat_models import ChatGoogleGenerativeAI
import os

# Optional: set project & location if not in env
os.environ["GOOGLE_CLOUD_PROJECT"] = "your-gcp-project-id"
os.environ["GOOGLE_CLOUD_REGION"] = "us-central1"  # change if needed

# Initialize Google Generative AI chat model
chat = ChatGoogleGenerativeAI(
    model="gemini-1.5-t",  # PaLM / Gemini model
    temperature=0.7,
)

# Messages using LangChain schema
messages = [
    SystemMessage(content="You are a helpful assistant that answers concisely."),
    HumanMessage(content="Explain the difference between supervised and unsupervised learning."),
]

# Invoke the model
response = chat.invoke(messages)

print("Google Generative AI:", response.content)
