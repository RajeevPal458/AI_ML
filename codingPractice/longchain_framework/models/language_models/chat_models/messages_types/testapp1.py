from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
import util
messages = [
    SystemMessage(content="You are a helpful assistant who explains topics simply."),
    HumanMessage(content="Explain Kafka consumer groups."),
    AIMessage(content="Kafka consumer groups allow multiple consumers to share..."),
    HumanMessage(content="Give me a Java example."),
]


chat = ChatOpenAI(model="gpt-4o")
response = chat(messages)
print(response.content)
