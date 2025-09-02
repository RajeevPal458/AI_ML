from openai import OpenAI
import util

client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-3-small",
    input="FastAPI is a high-performance Python web framework."
)

vector = response.data[0].embedding
print(len(vector))  # length, first 5 values
print("################")
print(vector[:5])

