# -*- coding: utf-8 -*-
"""YT LangChain RAG tips and Tricks 03 - BM25 + Ensemble = Hybrid Search.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1lsT1V_U1Gq-jv09wv0ok5QHdyRjJyNxm
"""

# !pip -q install langchain huggingface_hub openai google-search-results tiktoken chromadb rank_bm25 faiss-cpu

import os

os.environ["OPENAI_API_KEY"] = ""

# !pip show langchain

"""# Hybrid Search

## BM25 Retriever - Sparse retriever
"""

from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema import Document

from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS

from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()

doc_list = [
    "I like apples",
    "I like oranges",
    "Apples and oranges are fruits",
    "I like computers by Apple",
    "I love fruit juice"
]

# initialize the bm25 retriever and faiss retriever
bm25_retriever = BM25Retriever.from_texts(doc_list)
bm25_retriever.k = 2

bm25_retriever.get_relevant_documents("Apple")

bm25_retriever.get_relevant_documents("a green fruit")

bm25_retriever.dict

"""## Embeddings - Dense retrievers FAISS"""

faiss_vectorstore = FAISS.from_texts(doc_list, embedding)
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 2})

faiss_retriever.get_relevant_documents("A green fruit")

"""## Ensemble Retriever"""

# initialize the ensemble retriever
ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever],
                                       weights=[0.5, 0.5])

docs = ensemble_retriever.get_relevant_documents("A green fruit")
docs

docs = ensemble_retriever.get_relevant_documents("Apple Phones")
docs

