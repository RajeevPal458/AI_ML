# -----------------------------
# 1️⃣ String Prompt Composition
# -----------------------------
from langchain_core.prompts import PromptTemplate

# Compose a string prompt with multiple parts
prompt = (
    PromptTemplate.from_template("Tell me a joke about {topic}")
    + ", make it funny"
    + "\n\nand in {language}"
)

# Show the prompt template
print("Prompt Template:", prompt)

# Format with actual variables
formatted_prompt = prompt.format(topic="sports", language="spanish")
print("Formatted Prompt:", formatted_prompt)


# -----------------------------
# 2️⃣ Chat Prompt Composition
# -----------------------------
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# Initialize a system message
system_msg = SystemMessage(content="You are a nice pirate")

# Compose chat prompt by concatenating messages and variables
chat_prompt = system_msg + HumanMessage(content="hi") + AIMessage(content="what?") + "{input}"

# Format the chat prompt with a user input
formatted_messages = chat_prompt.format_messages(input="i said hi")

print("\nFormatted Chat Messages:")
for msg in formatted_messages:
    print(f"{type(msg).__name__}: {msg.content}")


# -----------------------------
# 3️⃣ Pipeline Prompt (Deprecated)
# -----------------------------
from langchain_core.prompts import PipelinePromptTemplate

# Final combined template
full_template = """{introduction}

{example}

{start}"""
full_prompt = PromptTemplate.from_template(full_template)

# Individual templates
introduction_prompt = PromptTemplate.from_template("You are impersonating {person}.")
example_prompt = PromptTemplate.from_template("""Here's an example of an interaction:

Q: {example_q}
A: {example_a}""")
start_prompt = PromptTemplate.from_template("""Now, do this for real!

Q: {input}
A:""")

# Combine into a pipeline prompt
input_prompts = [
    ("introduction", introduction_prompt),
    ("example", example_prompt),
    ("start", start_prompt),
]

pipeline_prompt = PipelinePromptTemplate(
    final_prompt=full_prompt,
    pipeline_prompts=input_prompts
)

# Show required input variables
print("\nPipeline Input Variables:", pipeline_prompt.input_variables)

# Format the pipeline prompt with actual values
formatted_pipeline = pipeline_prompt.format(
    person="Elon Musk",
    example_q="What's your favorite car?",
    example_a="Tesla",
    input="What's your favorite social media site?"
)

print("\nFormatted Pipeline Prompt:\n", formatted_pipeline)
