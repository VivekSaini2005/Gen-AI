from pathlib import Path
pdf_path = Path(__file__).parent / "COMPILER DESIGN.pdf"


# Load pdf file into python program
from langchain_community.document_loaders import PyPDFLoader  # pip install -U langchain-text-splitters
loader = PyPDFLoader(file_path=pdf_path)
pages = loader.load()
# print(pages[5])


# Split the document into smaller chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter  # pip install -U langchain-text-splitters

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500, 
    chunk_overlap=100
)
chunks = text_splitter.split_documents(pages)


# Create a vector store and add the chunks to it, embeddings are used to convert text into vector format which can be easily searched and retrieved based on similarity
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

from langchain_google_genai import GoogleGenerativeAIEmbeddings # pip install -U langchain-google-genai

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    api_key=api_key,
    batch_size=10
)
# vector = embedding_model.embed_query("what is a compiler?")
# print(vector[:5])


# Create a vector store and add the chunks to it
from langchain_qdrant import QdrantVectorStore # pip install -U langchain-qdrant

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url = "http://localhost:6333", # go to website http://localhost:6333/dashboard to monitor `qdrant` database
    collection_name="learning-rag"
)
print("Indexing of chunks done successfully!")