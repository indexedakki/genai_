def fibonacci(n):
    a = 0
    b = 1

    for i in range(2, n):
        a, b = b, a+b
        yield b

# Example usage:
for num in fibonacci(8):
    print(num)
    
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import retrieval_qa
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

sample_data ="this is sample data which would be tokenized and stored in vector store"

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 2)
documents = text_splitter.create_documents([sample_data])
print(documents)

embeddings = OpenAIEmbeddings( model="text-embedding-3-large" )
persist_directory = "db"
vector_store = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)

retriever = vector_store.as_retriever(kwargs={"k":2})

gpt = ChatOpenAI(
    model="gpt-4",
    temperature=0
)

qa_chain = retrieval_qa(llm= gpt,
                        retriever = retriever)

query = "what does the text say"
fetched = qa_chain(query)
print(fetched["result"])
