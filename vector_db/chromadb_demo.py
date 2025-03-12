import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI clients
embedding = OpenAIEmbeddings()
llm = OpenAI()

# Constants
PERSIST_DIRECTORY = 'db'
ARTICLES_DIRECTORY = "new_articles/"

def load_articles():
    """Load and split articles from directory"""
    loader = DirectoryLoader(
        ARTICLES_DIRECTORY, 
        glob="*.txt", 
        loader_cls=TextLoader
    )
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    return text_splitter.split_documents(documents)

def add_articles_to_db():
    """Add processed articles to vector database"""
    texts = load_articles()
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=PERSIST_DIRECTORY
    )
    vectordb.persist()

def process_llm_response(llm_response):
    """Process and print LLM response with sources"""
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

def search_articles(query: str):
    """Search articles using vector database and LLM"""
    vectordb = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 2})
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    llm_response = qa_chain.invoke(query)
    process_llm_response(llm_response)

def main():
    query = "How much money did Microsoft raise?"
    search_articles(query)

if __name__ == "__main__":
    main()