from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
import util
# ----------------------------
# 1. Define Schema with Pydantic
# ----------------------------
class ResponseFormatter(BaseModel):
    answer: str = Field(description="The direct answer to the question.")
    followup_question: Optional[str] = Field(
        description="A related follow-up question to continue the conversation."
    )

# ----------------------------
# 2. Initialize Model
# ----------------------------
# Use GPT-4o (fast multimodal model) but you can replace with "gpt-4o-mini" or "gpt-3.5-turbo"
model = ChatOpenAI(model="gpt-4o", temperature=0)

# ----------------------------
# 3. Bind Schema as Tool
# ----------------------------
# Correct way: bind tool first, then model can call it
model_with_tools = model.bind_tools([ResponseFormatter])

# ----------------------------
# 4. Invoke Model with a Prompt
# ----------------------------
prompt = "What is the powerhouse of the cell?"
ai_msg = model_with_tools.invoke(prompt)

# ----------------------------
# 5. Extract Tool Call Arguments (Dictionary)
# ----------------------------
tool_args = ai_msg.tool_calls[0]["args"]
print("Tool Call Arguments (Dict):")
print(tool_args)

# ----------------------------
# 6. Parse into Pydantic Object
# ----------------------------
pydantic_object = ResponseFormatter.parse_obj(tool_args)
print("\nParsed into Pydantic Object:")
print(pydantic_object)

# ----------------------------
# 7. Access Structured Output Fields
# ----------------------------
print("\nFinal Answer:", pydantic_object.answer)
print("Follow-up Question:", pydantic_object.followup_question)

# ----------------------------
# 8. (Optional) Use in a Simple Chain
# ----------------------------
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Define a chain that asks another follow-up
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful biology tutor."),
    ("human", "{question}")
])

chain = prompt_template | model_with_tools | StrOutputParser()

print("\n--- Chain Execution ---")
response = chain.invoke({"question": "Explain ATP in simple terms."})
print(response)
