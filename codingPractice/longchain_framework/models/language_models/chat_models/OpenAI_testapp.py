from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import util
# Set OPENAI_API_KEY in environment variables
"""
1. temperature=0.7
Controls randomness / creativity of the output.

Range:

0.0 → deterministic (always the same output for the same input).

~0.7 → balanced mix of creativity and reliability.

1.0+ → more randomness and diversity, but can lose factual accuracy.

"""
chat = ChatOpenAI(model="gpt-4", temperature=0.7, max_completion_tokens=50)

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me a joke about programmers."),
]

response = chat.invoke(messages)
print("OpenAI:", response.content)

