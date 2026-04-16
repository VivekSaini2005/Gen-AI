from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

from langchain_google_genai import GoogleGenerativeAIEmbeddings # pip install -U langchain-google-genai

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    api_key=api_key
)
# vector = embedding_model.embed_query("what is a compiler?")
# print(vector[:5])


from langchain_qdrant import QdrantVectorStore
vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    url = "http://localhost:6333", # go to website http://localhost:6333/dashboard to monitor `qdrant` database
    collection_name="learning-rag"
)


# take user input collection_name=''
user_query = input("Ask Something : ")

# Relevent chunks from database 
search_result = vector_db.similarity_search(user_query)

context = "\n\n\n".join([f"Page Content: {result.page_content}\nPage Number:{result.metadata['page_label']}\nFile Location: {result.metadata['source']}" for result in search_result])

system_prompt = f"""You are helpful AI assistant who answers the user query based on the available context retrieved 
                    from a PDF file along with page_contents and page number,You should only ans the user based on the
                     following context and navigate the user to open the right page number to know more.
                     context: {context}"""



from google import genai
from google.genai import types

client = genai.Client()

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    config=types.GenerateContentConfig(
        system_instruction={system_prompt}),
    contents=[user_query]
)

print(response.text)