from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool
import  util

# -------------------------
# 1. Define Tools
# -------------------------
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


@tool
def reverse_text(text: str) -> str:
    """Reverse the given text."""
    return text[::-1]


# -------------------------
# 2. Define Schema (Pydantic)
# -------------------------
class MySchema(BaseModel):
    result: str = Field(description="The final result after tool use")
    reasoning: str = Field(description="Explanation of how the result was derived")


# -------------------------
# 3. Create Model + Bind Tools
# -------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ✅ Correct Order: First bind tools, then structured output
model_with_tools = llm.bind_tools([add_numbers, reverse_text])
structured_model = model_with_tools.with_structured_output(MySchema)


# -------------------------
# 4. Create Prompt
# -------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that must always use tools if required."),
        ("user", "{input}")
    ]
)


# -------------------------
# 5. Build Chain
# -------------------------
chain = prompt | structured_model


# -------------------------
# 6. Run Chain
# -------------------------
user_query = "Please add 15 and 30, then reverse the result text."
output = chain.invoke({"input": user_query})

print("Structured Output:")
print(output)
