# from pathlib import Path

# from langchain_huggingface import HuggingFaceEmbeddings
# from sentence_transformers import SentenceTransformer


# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# LOCAL_MODEL_DIR = PROJECT_ROOT / "models" / "embeddinggemma"

# if not LOCAL_MODEL_DIR.exists():
#     raise FileNotFoundError(f"Embedding model not found at: {LOCAL_MODEL_DIR}")


# # Sentence-Transformers usage (direct)
# model = SentenceTransformer(str(LOCAL_MODEL_DIR))

# sentences = [
#     "That is a happy person",
#     "That is a happy dog",
#     "That is a very happy person",
#     "Today is a sunny day",
# ]

# embeddings = model.encode(sentences)
# similarities = model.similarity(embeddings, embeddings)

# print(f"Loaded model from: {LOCAL_MODEL_DIR}")
# print(f"Embedding matrix shape: {embeddings.shape}")
# print(f"Similarity matrix shape: {similarities.shape}")


# # LangChain usage (for Qdrant / RAG pipelines)
# langchain_embedding_model = HuggingFaceEmbeddings(
#     model_name=str(LOCAL_MODEL_DIR)
# )




"""
FULL RAG PIPELINE:
PDF → Load → Chunk → Embed (local model) → Store (Qdrant) → Query
"""



# -----------------------------
# 📌 1. Load Required Libraries
# -----------------------------






# -----------------------------
# 📌 2. Paths Configuration
# -----------------------------
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent

PDF_PATH = Path(__file__).parent / "CppBook.pdf"
MODEL_PATH = REPO_ROOT / "models" / "embeddinggemma"

# Backward-compatible fallback if models are kept inside rag/
if not MODEL_PATH.exists():
    MODEL_PATH = PROJECT_ROOT / "models" / "embeddinggemma"

# Safety check
if not PDF_PATH.exists():
    raise FileNotFoundError(f"PDF not found at: {PDF_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")






# -----------------------------
# 📌 3. Load PDF
# -----------------------------
from langchain_community.document_loaders import PyPDFLoader
print("\n📄 Loading PDF...")

loader = PyPDFLoader(str(PDF_PATH))


documents = loader.load() # Each page becomes one document

print(f"✅ Loaded {len(documents)} pages")





# -----------------------------
# 📌 4. Chunking (Important Step)
# -----------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("\n✂️ Splitting into chunks...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Max characters per chunk
    chunk_overlap=200      # Overlap to preserve context
)

chunks = text_splitter.split_documents(documents) # Split pages into smaller chunks

print(f"✅ Created {len(chunks)} chunks")





# -----------------------------
# 📌 5. Load Embedding Model (LOCAL)
# -----------------------------
from langchain_huggingface import HuggingFaceEmbeddings
print("\n🤖 Loading local embedding model...")

embedding_model = HuggingFaceEmbeddings(
    model_name=str(MODEL_PATH)
)

print("✅ Embedding model loaded")




# -----------------------------
# 📌 6. Create Vector Store (Qdrant)
# -----------------------------
from langchain_qdrant import QdrantVectorStore
print("\n📊 Creating vector store and indexing chunks...")
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="CppBook-collection",
    force_recreate=True  # WARNING: This will delete existing collection with the same name
)

print("✅ Indexing completed successfully!")




# -----------------------------
# 📌 7. Query Function
# -----------------------------
def ask_query(query: str, k: int = 3):
    """
    Search similar chunks from vector DB
    """
    print(f"\n🔍 Query: {query}")

    # Retrieve top-k similar chunks
    results = vector_store.similarity_search(query, k=k)

    print("\n📚 Top Results:\n")

    for i, doc in enumerate(results):
        print(f"--- Result {i+1} ---")
        print(f"Page: {doc.metadata.get('page', 'N/A')}")
        print(doc.page_content[:500])  # print first 500 chars
        print()


# -----------------------------
# 📌 8. Run Sample Query
# -----------------------------
if __name__ == "__main__":
    while True:
        user_query = input("\n💬 Enter your question (or 'exit'): ")

        if user_query.lower() == "exit":
            break

        ask_query(user_query)