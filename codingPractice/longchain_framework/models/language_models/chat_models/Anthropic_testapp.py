#2. Anthropic – ChatAnthropic
from langchain_anthropic import ChatAnthropic
from langchain.schema import SystemMessage, HumanMessage
import os
from dotenv import load_dotenv, find_dotenv
import anthropic
import util


client =anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
print("========================={}",client.models.list())

"""
========================={} SyncPage[ModelInfo](data=[
    ModelInfo(id='claude-opus-4-1-20250805', created_at=datetime.datetime(2025, 8, 5, 0, 0, tzinfo=datetime.timezone.utc), display_name='Claude Opus 4.1', type='model'), 
    ModelInfo(id='claude-opus-4-20250514', created_at=datetime.datetime(2025, 5, 22, 0, 0, tzinfo=datetime.timezone.utc), display_name='Claude Opus 4', type='model'), 
    ModelInfo(id='claude-sonnet-4-20250514', created_at=datetime.datetime(2025, 5, 22, 0, 0, tzinfo=datetime.timezone.utc), display_name='Claude Sonnet 4', type='model'), 
    ModelInfo(id='claude-3-7-sonnet-20250219', created_at=datetime.datetime(2025, 2, 24, 0, 0, tzinfo=datetime.timezone.utc), display_name='Claude Sonnet 3.7', type='model'), 
    ModelInfo(id='claude-3-5-haiku-20241022', created_at=datetime.datetime(2024, 10, 22, 0, 0, tzinfo=datetime.timezone.utc), display_name='Claude Haiku 3.5', type='model'), 
    ModelInfo(id='claude-3-haiku-20240307', created_at=datetime.datetime(2024, 3, 7, 0, 0, tzinfo=datetime.timezone.utc), display_name='Claude Haiku 3', type='model')], 
    has_more=False, first_id='claude-opus-4-1-20250805', last_id='claude-3-haiku-20240307')

"""
chat = ChatAnthropic(model="claude-3-7-sonnet-20250219", temperature=0.7,max_tokens_to_sample=50)

messages = [
    SystemMessage(content="You are a witty assistant."),
    HumanMessage(content="Explain recursion in simple terms."),
]

response = chat.invoke(messages)
print("Anthropic:", response.content)
