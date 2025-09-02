# Most simplest way to use a local model is Ollama, here we can use all the models available in Ollama by just changing the model name.

from langchain_ollama import OllamaLLM

# Using mistral via Ollama (Local)
model = OllamaLLM(model = "mistral")

response = model.invoke("What is the capital of India")
print(response)
