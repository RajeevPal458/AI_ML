from datetime import date
#### Chat models with Prompts ####
import util
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate

today = date.today()

prompt_template = PromptTemplate.from_template(
    f'Todays Date: {today}: Tell me a {{adjective}} joke about {{content}}',
    template_format='f-string'
)
print(prompt_template.format(adjective="funny", content="chickens"))



######################################

template = f'''You are a joke generating chatbot.
Provide funny jokes based on the themes requested.

Question: Tell me a {{adjective}} joke about {{content}}'

Answer: '''

prompt_template = PromptTemplate.from_template(template)
print(prompt_template.format(adjective='funny', content='chickens'))

#####################################

prompt_template = PromptTemplate.from_template(
    'Tell me a {adjective} joke about {content}'
)

openai =  ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=50  # ~40 words
)

chain = LLMChain(llm=openai, prompt=prompt_template)

response = chain.invoke(
    input={'adjective': 'scary', 'content': 'French'}
)

print(response['text'])
