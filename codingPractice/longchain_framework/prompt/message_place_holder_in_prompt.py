from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.schema import HumanMessage

# 1️⃣ Create system prompt
system_prompt = SystemMessagePromptTemplate.from_template("You are a helpful assistant")

# 2️⃣ Create messages placeholder
msgs_placeholder = MessagesPlaceholder(variable_name="msgs")

# 3️⃣ Create ChatPromptTemplate
chat_prompt_template = ChatPromptTemplate.from_messages([system_prompt, msgs_placeholder])

# 4️⃣ Invoke with a list of HumanMessage
prompt_value = chat_prompt_template.format_prompt(msgs=[HumanMessage(content="hi!")])

# 5️⃣ Access messages
for msg in prompt_value.to_messages():
    print(f"{type(msg).__name__}: {msg.content}")


