from src.helper import load_pdf_files, text_splitter, download_hugging_face_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Pinecone
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = pinecone_api_key


extractd_data = load_pdf_files("/home/muhammed-shafeeh/AI_ML/medical_chatbot/data")
text_chunks = text_splitter(extractd_data)
embeddings = download_hugging_face_embeddings()

pc = Pinecone(pinecone_api_key)
index_name = 'medicalchatbot'

pc.create_index(
    name=index_name,
    dimension=384,
    metric='cosine',
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

docsearch = PineconeVectorStore(
    index=pc.Index(index_name),  
    embedding=embeddings,
)
docsearch.add_documents(text_chunks) 