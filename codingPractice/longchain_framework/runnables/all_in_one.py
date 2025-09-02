from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnableLambda,
    RunnableBranch,
    RunnablePassthrough,
)
from pydantic import BaseModel
import util
# ------------------------
# 1. Define a Pydantic model for structured sentiment output
# ------------------------
class SentimentResponse(BaseModel):
    sentiment: str  # "positive", "negative", "neutral"

# ------------------------
# 2. Create the LLM
# ------------------------
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ------------------------
# 3. Prompt to generate a story
# ------------------------
story_prompt = PromptTemplate(
    template="Write a short story about {topic} in 3-4 sentences.",
    input_variables=["topic"]
)

# ------------------------
# 4. Prompt for sentiment classification
# ------------------------
sentiment_prompt = PromptTemplate(
    template="Classify the sentiment of this text as positive, negative, or neutral:\n\n{text}",
    input_variables=["text"]
)

# ------------------------
# 5. Runnable sequence → story generation
# ------------------------
story_chain = RunnableSequence(
    story_prompt,
    model,
    StrOutputParser()
)

# ------------------------
# 6. Runnable sequence → sentiment detection (Pydantic v1 fix!)
# ------------------------
sentiment_chain = RunnableSequence(
    sentiment_prompt,
    model,
    StrOutputParser(),
    RunnableLambda(lambda x: SentimentResponse.parse_obj({"sentiment": x.strip().lower()}))
)

# ------------------------
# 7. Runnable Lambda → word count
# ------------------------
word_count = RunnableLambda(lambda story: {"word_count": len(story.split())})

# ------------------------
# 8. RunnableBranch → Regenerate if story is too short
# ------------------------
branch = RunnableBranch(
    (lambda story: len(story.split()) < 10,  # condition
     RunnableSequence(
         PromptTemplate(template="Regenerate a longer story about {topic}", input_variables=["topic"]),
         model,
         StrOutputParser()
     )),
    RunnablePassthrough()  # default (keep story as-is)
)

# ------------------------
# 9. RunnableParallel → Run multiple analyses in parallel
# ------------------------
parallel_tasks = RunnableParallel(
    sentiment=sentiment_chain,
    word_count=word_count
)

# ------------------------
# 10. Final chain with sequencing + branching + parallelism
# ------------------------
full_chain = RunnableSequence(
    story_chain,    # Step 1 → generate story
    branch,         # Step 2 → regenerate if too short
    parallel_tasks  # Step 3 → sentiment + word count in parallel
)

# ------------------------
# 11. Run the chain
# ------------------------
result = full_chain.invoke({"topic": "friendship"})

print("Final Result:")
print(result)
