from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load a pre-trained embedding model that converts text into numerical vectors
model = SentenceTransformer('all-MiniLM-L6-v2')

# Course summary sentences that will be embedded once and searched against
sentences = [
    "FastAPI automatically validates incoming request data using Pydantic models.",
    "JWT authentication allows secure login by issuing tokens that must be included in protected API requests.",
    "SQLAlchemy provides ORM tools for defining database models and performing CRUD operations.",
    "Dependency injection in FastAPI ensures clean, testable access to shared resources like database sessions.",
    "Streamlit reruns the script on every interaction, so session state is required to persist user data.",
    "PATCH requests allow partial updates to a resource without resending the entire object.",
    "Proper error handling in FastAPI returns clear HTTP status codes such as 401, 403, and 404.",
    "Semantic search uses embeddings to compare meaning rather than exact keyword matches.",
    "OpenAI’s streaming responses allow tokens to be displayed gradually for a more interactive chat experience.",
    "Mock data and fallback logic help keep frontend applications functional even when the backend is offline."
]

# Embed all sentences once at startup (this is the "document store")
sentence_embeddings = model.encode(sentences)

print("Semantic Course Search Ready!")
print("Type a query or 'quit' to exit.\n")

while True:
    # Ask the user for a search query
    query = input("Search (or 'quit'): ").strip()

    # Allow user to exit the loop
    if query.lower() == "quit":
        break

    # Prevent embedding empty strings (they produce meaningless results)
    if query == "":
        print("Please enter a search query.\n")
        continue

    # Encourage more descriptive queries (single-word queries are usually too vague)
    if len(query.split()) < 2:
        print("Try a more descriptive query.\n")
        continue

    # Convert the user's query into an embedding vector
    query_embedding = model.encode(query)

    # Compute cosine similarity between the query and all stored sentence embeddings
    # Higher scores mean stronger semantic similarity
    similarities = util.cos_sim(query_embedding, sentence_embeddings)[0]

    # Warn the user if the best match is very weak (still show results)
    # This helps users understand when their query doesn't align well with the course content
    if max(similarities) < 0.2:
        print("No strong semantic matches found, but here are the closest results:\n")

    # Sort similarity scores in descending order and take the top 3 matches
    top_results = np.argsort(-similarities)[:3]

    print("\nTop 3 results:")
    for idx in top_results:
        score = float(similarities[idx])  # Convert tensor value to float for printing
        print(f"  - [{score:.4f}] {sentences[idx]}")
    print()
