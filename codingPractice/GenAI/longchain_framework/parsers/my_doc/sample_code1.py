from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from pydantic import BaseModel, Field
import util

# 1. Define schema with Pydantic (recommended way)
class ReviewSummary(BaseModel):
    key_themes: list[str] = Field(..., description="Main themes of the review")
    summary: str = Field(..., description="Short summary of the review")


# 2. Initialize the LLM
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 3. Bind schema to model (structured output parsing will be auto-applied)
model_with_structure = model.with_structured_output(ReviewSummary)

# 4. Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that extracts structured insights from reviews."),
    ("user", "Analyze the following review and extract main themes and a short summary:\n\n{review}")
])

# 5. Chain prompt -> model_with_structure
chain = prompt | model_with_structure

# 6. Run chain with example input
user_input = {
    "review": "Nitish Singh's review of the Samsung Galaxy S24 Ultra highlights its powerful Snapdragon 8 Gen 3 processor, impressive battery life, and great camera quality. He also appreciates the S-Pen integration but feels the phone is a bit heavy, has too much bloatware, and is overpriced."
}

result: ReviewSummary = chain.invoke(user_input)

print(result)
print("Key themes:", result.key_themes)
print("Summary:", result.summary)
