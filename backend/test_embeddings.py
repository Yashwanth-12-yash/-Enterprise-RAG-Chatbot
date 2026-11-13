from embedding_service import generate_embeddings

# Sample company document chunks
chunks = [
    "Employees receive 20 days of annual leave.",
    "Employees must apply for leave through the HR portal.",
    "The company provides health insurance to employees."
]

# Generate embeddings
embeddings = generate_embeddings(chunks)

# Display results
print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)

print("\nFirst embedding:")
print(embeddings[0])

print("\nEmbedding generation successful!")