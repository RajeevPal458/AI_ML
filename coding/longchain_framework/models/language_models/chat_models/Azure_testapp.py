# GPT models via Azure
from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import util
# Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT
chat = AzureChatOpenAI(
    model="gpt-4",
    temperature=0.7,
)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Write a haiku about the cloud."),
]

response = chat.invoke(messages)
print("Azure OpenAI:", response.content)
