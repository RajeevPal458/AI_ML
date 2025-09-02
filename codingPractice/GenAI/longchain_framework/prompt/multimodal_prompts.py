# -----------------------------
# 1️⃣ Imports
# -----------------------------
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
import base64
import httpx
import util
# -----------------------------
# 2️⃣ Prompt with image URL
# -----------------------------
prompt_url = ChatPromptTemplate(
    [
        {
            "role": "system",
            "content": "Describe the image provided.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source_type": "url",
                    "url": "{image_url}",
                },
            ],
        },
    ]
)

# Initialize chat model (example with Claude)
llm = init_chat_model("claude-3-7-sonnet-20250219")

# Example image URL
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

# Combine prompt and model in a chain
chain_url = prompt_url | llm

# Invoke with image URL
response_url = chain_url.invoke({"image_url": image_url})
print("Response for Image URL:")
print(response_url.text())


# -----------------------------
# 3️⃣ Prompt with base64 image
# -----------------------------
prompt_base64 = ChatPromptTemplate(
    [
        {
            "role": "system",
            "content": "Describe the image provided.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source_type": "base64",
                    "mime_type": "{image_mime_type}",
                    "data": "{image_data}",
                    "cache_control": {"type": "{cache_type}"},
                },
            ],
        },
    ]
)

# Convert image from URL to base64
image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")

# Combine prompt and model in a chain
chain_base64 = prompt_base64 | llm

# Invoke with base64 image
response_base64 = chain_base64.invoke(
    {
        "image_data": image_data,
        "image_mime_type": "image/jpeg",
        "cache_type": "ephemeral",
    }
)
print("\nResponse for Base64 Image:")
print(response_base64.text())
