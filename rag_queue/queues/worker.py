from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# Load embedding model
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent

MODEL_PATH = Path("/home/vivek-saini/Gen-AI/models/embeddinggemma")

embedding_model = HuggingFaceEmbeddings(
    model_name=str(MODEL_PATH)
)

# Load existing DB (IMPORTANT)
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="CppBook-collection"
)



def process_query(query: str):
    print('Searching Chunks',query)
    search_result = vector_store.similarity_search(query, k=3)


    context = "\n\n\n".join([f"Page Content: {result.page_content}\nPage Number:{result.metadata['page_label']}\nFile Location: {result.metadata['source']}" for result in search_result])

    system_prompt = f"""You are helpful AI assistant who answers the user query based on the available context retrieved 
                        from a PDF file along with page_contents and page number,You should only ans the user based on the
                        following context and navigate the user to open the right page number to know more.
                        context: {context}"""
    
    from dotenv import load_dotenv
    load_dotenv()
    from google import genai
    from google.genai import types

    client = genai.Client()

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        config=types.GenerateContentConfig(
            system_instruction={system_prompt}),
        contents=[query]
    )

    # print(response.text)
    return response.text