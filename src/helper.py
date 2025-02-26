from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()  

pinecone_api_key = os.environ.get("PINECONE_API_KEY")

#Extract the data from pdf
def load_pdf_files(directory_path):
    loader = PyPDFDirectoryLoader(directory_path)
    document = loader.load()
    return document

# Split the data into chunks

def text_splitter(extractd_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    texts_chunks = text_splitter.split_documents(extractd_data)
    return texts_chunks


# Download the embeddings from hugging face 
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name= 'sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

