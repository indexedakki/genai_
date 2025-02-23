from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import retrieval_qa
from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())
OPENAI_API_KEY = os.environ("OPENAI_API_KEY")

sample_data = "THis is sample data"

text_splitter = SemanticChunker(OpenAIEmbeddings())
documents = text_splitter.create_documents([sample_data])

embeddings = OpenAIEmbeddings(model="text-emdeddings-small-3")
persist_directory = "db"
vector_store = Chroma.from_documents(documents, embeddings, persist_directory= persist_directory)

retriever = vector_store.as_retriever(kwargs={"k":2})

gpt = ChatOpenAI(model="gpt-4",temperature=0)

qa_chain = retrieval_qa(llm=gpt, retriever=retriever)

query = "what does the text say"
response = qa_chain(query)
print(response['result'])

# Hybrid Search
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import FAISS

bm25retriver = BM25Retriever.from_texts(sample_data)
bm25retriver.k = 2

bm25retriver.get_relevant_documents("Apple")

faiss_vectorstore = FAISS.from_texts(sample_data, embeddings)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs = {"k":2})

faiss_retriever.get_relevant_documents("Apple")

ensemble_retriver = EnsembleRetriever(retrievers=[ bm25retriver, faiss_retriever ], weights=[0.5,0.5])
ensemble_retriver.get_relevant_documents("Apple")