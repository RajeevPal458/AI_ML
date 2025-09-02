import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# -----------------------------
# 1️⃣ Set OpenAI API Key
# -----------------------------
import util
# -----------------------------
# 2️⃣ Initialize the Chat Model
# -----------------------------
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# -----------------------------
# 3️⃣ Few-shot examples
# -----------------------------
examples = [
    {"input": "2 🦜 2", "output": "4"},
    {"input": "2 🦜 3", "output": "5"},
]

# Create example prompt template
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

# Create few-shot prompt wrapper
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# Optional: see how few-shot messages look
print("Few-shot messages:")
for msg in few_shot_prompt.invoke({}).to_messages():
    print(f"{msg.type}: {msg.content}")

# -----------------------------
# 4️⃣ Final Prompt with few-shot
# -----------------------------
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a wondrous wizard of math."),
        few_shot_prompt,           # insert few-shot examples
        ("human", "{input}"),      # the user question
    ]
)

# -----------------------------
# 5️⃣ Combine prompt and model into a simple chain
# -----------------------------
chain = final_prompt | model

# -----------------------------
# 6️⃣ Invoke the chain with a new question
# -----------------------------
response = chain.invoke({"input": "What is 2 🦜 9?"})
print("\nModel Response:")
print(response.content)
