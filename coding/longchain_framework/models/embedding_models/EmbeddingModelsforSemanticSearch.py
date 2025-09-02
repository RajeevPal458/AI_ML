from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import os
os.environ["USE_TF"] = "0"
# Load environment variables (API keys)


# Initialize Embedding Model
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Sample Documents (Support Articles)
documents = [
    "How to reset your password quickly and securely.",
    "Troubleshooting login issues on your account.",
    "Understanding account security and two-factor authentication.",
    "How to delete your account permanently.",
    "How to change your registered email address."
]

# User Query
query = "I forgot my password, what should I do?"

# Generate Document and Query Embeddings
doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

# Calculate Similarity using Cosine Similarity
scores = cosine_similarity([query_embedding], doc_embeddings)[0]
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

# Display the most similar document
print("User Query:", query)
print("Most Relevant Document:", documents[index])
print("Similarity Score:", score)
