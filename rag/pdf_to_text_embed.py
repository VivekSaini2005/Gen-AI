from pathlib import Path

# =========================
# STEP 1: Load PDF
# =========================
pdf_path = Path(__file__).parent / "COMPILER DESIGN.pdf"

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(file_path=pdf_path)
pages = loader.load()

print(f"Loaded {len(pages)} pages")


# =========================
# STEP 2: Split into chunks
# =========================
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(pages)

print(f"Created {len(chunks)} chunks")


# =========================
# STEP 3: Local Embedding Model (NO API)
# =========================
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded...")


# =========================
# STEP 4: Store in Qdrant
# =========================
from langchain_qdrant import QdrantVectorStore

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning-rag-v2"
)

print("✅ Indexing completed successfully!")