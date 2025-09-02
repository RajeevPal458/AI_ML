from fastapi import FastAPI
from openai import OpenAI
import numpy as np
import util

from fastapi import FastAPI
from openai import OpenAI
import numpy as np

app = FastAPI()
client = OpenAI()

# Step 1: Create sample restaurant data (without embeddings yet)
sample_restaurants = [
    {"name": "Luigi's Pasta", "description": "Authentic Italian pasta in the heart of the city"},
    {"name": "Roma's Pizzeria", "description": "Fresh wood-fired pizzas with mozzarella and basil"},
    {"name": "Tuscany Deli", "description": "Gourmet sandwiches and fine cheeses from Tuscany"},
]

# Step 2: Generate embeddings for the sample restaurants
restaurant_data = []
for r in sample_restaurants:
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=r["description"]
    ).data[0].embedding
    restaurant_data.append({
        "name": r["name"],
        "description": r["description"],
        "embedding": embedding
    })

# Step 3: Cosine similarity function
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Step 4: Search endpoint
@app.get("/search")
async def search_restaurants(query: str):
    # Create embedding for the query
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    # Rank restaurants by similarity
    ranked = sorted(
        restaurant_data,
        key=lambda r: cosine_similarity(query_embedding, r["embedding"]),
        reverse=True
    )

    # Remove embeddings from the response
    cleaned_results = [{"name": r["name"], "description": r["description"]} for r in ranked[:3]]

    return {"results": cleaned_results}

