from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import ChatOllama, OllamaEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
import os
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

load_dotenv()

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

from practice_documents import documents

print(len(documents))
print(documents[0].page_content)

#now we will create a vectorstore because we need to store the documents and their embeddings for similarity search   
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name ='hybrid_test'
)

#now create a vector retreiver using the vectorstore
vector_retreiver = vectorstore.as_retriever( search_kwargs={"k": 5}) #the default search type is cosine similarity, but you can change it to dot product or euclidean distance if you want
print('vector retriever ready ')


#now we create a BM25 retriever using the documents
bm25_retriever = BM25Retriever.from_documents(documents=documents, k=5)
print('bm25 retriever ready ')


#now we create ensemble retrieever using the vector retriever and bm25 retriever
ensemble_retriever = EnsembleRetriever(retrievers=[vector_retreiver, bm25_retriever], weights=[0.4, 0.6])
print('ensemble retriever ready ')
#for tuning  the weights if we have codes/id heavy documents we can give more weight to bm25 retriever and if we have more text heavy documents we can give more weight to vector retriever.


#testing diff queries 
def test_query(query, name , retriver):
    results = retriver.invoke(query)
    print(f'\\n{name}-Query: {query}')
    for i, doc in enumerate(results[:3]):
        preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        print(f"Result {i+1}: {preview} | Metadata: {doc.metadata}")                                           
        return results

test_queries=[
    "SKU-7742X specifications",
    "e_conn_refused error troubleshooting",
    "how do i authenticate",
    "wcag compliance",
    "router configuration"
]


for query in test_queries:
    print('=' * 60)
    #vector only
    test_query(query, "Vector Retriever", vector_retreiver)
    #bm25 only
    test_query(query, "BM25 Retriever", bm25_retriever)
    #ensemble
    test_query(query, "Ensemble Retriever", ensemble_retriever)



#to run the code, you can simply execute this script. It will create the vector store, BM25 retriever, and ensemble retriever, and then test a set of queries against each retriever, printing the top results along with their metadata.