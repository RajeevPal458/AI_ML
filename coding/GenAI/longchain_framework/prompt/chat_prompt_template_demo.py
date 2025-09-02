from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage, HumanMessage

# 1️⃣ Create prompt template
system_prompt = SystemMessagePromptTemplate.from_template("You are a helpful assistant")
human_prompt = HumanMessagePromptTemplate.from_template("Tell me a joke about {topic}")

chat_prompt_template = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

# 2️⃣ Invoke with variables
prompt_value = chat_prompt_template.format_prompt(topic="cats")

# 3️⃣ Access messages
for msg in prompt_value.to_messages():
    print(f"{type(msg).__name__}: {msg.content}")
