# Some HF chat LLMs

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.schema import SystemMessage, HumanMessage
import util
# Set HUGGINGFACEHUB_API_TOKEN
hf_llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=128,
    do_sample=True,
)

chat = ChatHuggingFace(llm=hf_llm)

messages = [
    SystemMessage(content="You are an AI that answers briefly."),
    HumanMessage(content="What is quantum computing?"),
]

response = chat.invoke(messages)
print("HuggingFace:", response.content)
