from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import util

# Create LLM
llm = OpenAI(temperature=0.7)

# Create a prompt template
template = "You are a helpful assistant. Answer the following question:\n{question}"
prompt = PromptTemplate(input_variables=["question"], template=template)

# Create chain
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain
response = chain.run("What is the capital of France?")
print(response)
