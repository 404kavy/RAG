from httpx2 import query
from langchain_ollama import ChatOllama, OllamaEmbeddings
from dotenv import load_dotenv
import numpy as np

load_dotenv()

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0)

def basic_embedding():
    text = "what is machine learning?"
    single_embedding = embeddings.embed_query(text)
    print(f"Embedding vector for '{text}': {single_embedding}")
    print(f"vector dimensions of embedding vector: {len(single_embedding)}")  # Print the dimensions of the embedding vector
    print(f"first 5 values of embedding vector: {single_embedding[:5]}")  # Print the first 5 values of the embedding vector
    print(f"vector norms of embedding vector: {np.linalg.norm(single_embedding)}")  # Print the vector norm of the embedding vector
    #vector norms means the length of the vector in the vector space. It is calculated as the square root of the sum of the squares of the vector's components. In this case, it gives an idea of how "long" or "large" the embedding vector is in its high-dimensional space.
#basic_embedding()  # Call the function to see the output                                                                                                                                                                             

def batch_embeddings():
    text = [
        "what is machine learning?",
        "what is deep learning?",
        "what is natural language processing?"]

    batch_embeddings = embeddings.embed_documents(text)
    import numpy as np

    batch_embedding = embeddings.embed_documents(text)

    for i, emb in enumerate(batch_embedding):

        print(f"Text {i+1} - Vector dimensions: {len(emb)}")

        print(f"Text {i+1} - First 5 values: {emb[:5]}")

        print(f"Text {i+1} - Vector norm: {np.linalg.norm(emb):.4f}")         

#batch_embeddings()

def similarity_search():
    docs = [
        "python is programming language",
        "javascript is use for web development",
        "machine learning enables computers to learn from data",
        "cats are popular pets",
        "deep learning uses neural network",
        "once upon a time there was an old man dancing",
        "some technical hard terms that this model shouldnt match wwith query ",
        "HaskellErlang Elixir Clojure OCaml",
        "Smalltalk Ada V Mojo APL"
    ]

    query = "what programming language exists?"


    

# embed documents and query 

# note that the embedding model for documents and query should be the same to ensure that they are in the same vector space for similarity comparison.

    doc_vector_embed = embeddings.embed_documents(docs)
    query_vector_embed = embeddings.embed_query(query)


    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    similarities = [cosine_similarity(query_vector_embed, doc_vec) for doc_vec in doc_vector_embed]
#here doc_vec is each document's embedding vector, and query_vector_embed is the embedding vector for the query. The cosine_similarity function computes the cosine similarity between the query and each document.
 #rank documents by similarity 
 
    ranked_docs = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)

    print(f"Query: {query}\n")
    print("Ranked documents by similarity:")
    for doc, sim in ranked_docs:
        print(f"Document: {doc} - Similarity: {sim:.4f}")

similarity_search()  # Call the function to see the output

