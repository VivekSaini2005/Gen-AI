from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODEL_DIR = PROJECT_ROOT / "models" / "embeddinggemma"

if not LOCAL_MODEL_DIR.exists():
    raise FileNotFoundError(f"Embedding model not found at: {LOCAL_MODEL_DIR}")


# Sentence-Transformers usage (direct)
model = SentenceTransformer(str(LOCAL_MODEL_DIR))

sentences = [
    "That is a happy person",
    "That is a happy dog",
    "That is a very happy person",
    "Today is a sunny day",
]

embeddings = model.encode(sentences)
similarities = model.similarity(embeddings, embeddings)

print(f"Loaded model from: {LOCAL_MODEL_DIR}")
print(f"Embedding matrix shape: {embeddings.shape}")
print(f"Similarity matrix shape: {similarities.shape}")


# LangChain usage (for Qdrant / RAG pipelines)
langchain_embedding_model = HuggingFaceEmbeddings(
    model_name=str(LOCAL_MODEL_DIR)
)