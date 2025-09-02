from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts import ChatPromptTemplate
import util
# 1. Define schema using Pydantic
class RandomInts(BaseModel):
    random_ints: list[int] = Field(..., description="A list of 10 random integers between 0 and 99")

# 2. Initialize model with JSON mode enabled
model = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(
    schema=RandomInts, method="json_mode"
)

# 3. Create a simple prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that generates random integers."),
    ("human", "Return a JSON object with key 'random_ints' and a value of 10 random ints in [0-99].")
])

# 4. Create a chain (Prompt → Model → Output)
chain = prompt | model

# 5. Invoke the chain
result = chain.invoke({})

# 6. Print result
print(result)
print(type(result))
