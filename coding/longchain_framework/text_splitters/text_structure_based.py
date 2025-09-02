from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """
Space exploration has led to incredible scientific discoveries. From landing on the Moon to exploring Mars, humanity continues to push the boundaries of what’s possible beyond our planet.

    These missions have not only expanded our knowledge of the universe but have also contributed to advancements in technology here on Earth. Satellite communications, GPS, and even certain medical imaging techniques trace their roots back to innovations driven by space programs.
"""

# Initialize the splitter
splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", " ", ""],  # Paragraph → Line → Sentence → Word → Character
    chunk_size=100,
    chunk_overlap=5

)

# Perform the split
chunks = splitter.create_documents([text])

print(len(chunks))
print(chunks)