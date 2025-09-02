# Local LLaMA or Mistral models

from langchain_ollama import ChatOllama
from langchain.schema import SystemMessage, HumanMessage
import util

# Requires Ollama running locally (ollama serve)
chat = ChatOllama(model="llama2", temperature=0)

messages = [
    SystemMessage(content="You are a concise assistant."),
    HumanMessage(content="Summarize the concept of blockchain."),
]

response = chat.invoke(messages)
print("Ollama:", response.content)
