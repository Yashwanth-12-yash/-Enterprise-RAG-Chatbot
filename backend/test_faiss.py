
from vector_store import (
    create_vector_store,
    search_documents
)

# Sample company documents
chunks = [
    "Employees receive 20 days of annual leave.",
    "Employees must apply for leave through the HR portal.",
    "The company provides health insurance to employees.",
    "Office working hours are from 9 AM to 6 PM.",
    "Employees receive performance bonuses annually."
]

# Store document embeddings
create_vector_store(chunks)

print("FAISS database created successfully!")

# Ask a question
query = "How many vacation days do employees get?"

results = search_documents(query, top_k=3)

print("\nSearch Results:")

for result in results:
    print("\nChunk:", result["chunk"])
    print("Similarity:", result["score"])
