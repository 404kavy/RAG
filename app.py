from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import chromadb

# Initialize ChromaDB client
chroma_client = chromadb.Client()

# Create or retrieve collection
collection_name = "test_collection"

collection = chroma_client.get_or_create_collection(
    name=collection_name
)

# Define text documents
documents = [
    {"id": "doc1", "text": "Hello, world!"},
    {"id": "doc2", "text": "How are you today?"},
    {"id": "doc3", "text": "Goodbye, see you later!"}
]

# Insert or update documents
collection.upsert(
    ids=[doc["id"] for doc in documents],
    documents=[doc["text"] for doc in documents]
)

# Define query
query = "Hello, world!"

# Retrieve similar documents
results = collection.query(
    query_texts=[query],
    n_results=3
)

print(results)

embeddings_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)


def similarity_search_with_scores():
    vector_docs = [
        Document(page_content=doc["text"], metadata={"id": doc["id"]})
        for doc in documents
    ]

    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=vector_docs,
        embedding=embeddings_model,
        persist_directory="./chroma_db"
    )

    # Similarity search with scores
    query = "Explain vector stores."

    results_with_scores = vectorstore.similarity_search_with_score(
        query,
        k=3
    )

    print(f"Top 3 results with scores for query '{query}':")

    for i, (doc, score) in enumerate(results_with_scores):
        print(
            f"Result {i+1}: {doc.page_content} "
            f"(Score: {score:.4f}, Source: {doc.metadata.get('source', doc.metadata.get('id', 'unknown'))})"
        )


if __name__ == "__main__":
    similarity_search_with_scores()
