from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from typer.cli import app  # ✅ updated import

# Load embedding model
# embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent

MODEL_PATH = REPO_ROOT / "models" / "embeddinggemma"

embedding_model = HuggingFaceEmbeddings(
    model_name=str(MODEL_PATH)
)


# Load existing DB (IMPORTANT)
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="CppBook-collection"
)

# Query
# query = "What is a compiler?"
user_query = input("Ask Something : ")

search_result = vector_store.similarity_search(user_query, k=3)

# for i, doc in enumerate(results):
#     print(f"\nResult {i+1}:")
#     print(doc.page_content)

context = "\n\n\n".join([f"Page Content: {result.page_content}\nPage Number:{result.metadata['page_label']}\nFile Location: {result.metadata['source']}" for result in search_result])

system_prompt = f"""You are helpful AI assistant who answers the user query based on the available context retrieved 
                    from a PDF file along with page_contents and page number,You should only ans the user based on the
                     following context and navigate the user to open the right page number to know more.
                     context: {context}"""


# from dotenv import load_dotenv
# import os

# load_dotenv()
# api_key = os.getenv("GOOGLE_API_KEY")
# from google import genai
# from google.genai import types

# client = genai.Client()

# response = client.models.generate_content(
#     model="gemini-3-flash-preview",
#     config=types.GenerateContentConfig(
#         system_instruction={system_prompt}),
#     contents=[user_query]
# )

# print(response.text)




# from fastapi import FastAPI, Body
# app = FastAPI()

# from ollama import chat
# from ollama import Client

# client = Client(
#     host='http://localhost:11434/',
# )

# @app.post("/askme")
# def askme(message: str = Body(..., description="The message to send to the model")):
#     response = client.chat(model='gemma2:2b', messages=[
#         {
#             "role": "system",
#             "content": system_prompt
#         },
#         {
#             'role': 'user',
#             'content': message,
#         },
#     ])
#     return response.message.content

# print(askme(user_query))




from transformers import AutoTokenizer, AutoModelForCausalLM
path = "/home/vivek-saini/models/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(
    path,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    path,
    local_files_only=True,
    device_map="auto"
)

print("✅ Model loaded successfully")
messages = [
    {"role": "user", "content": system_prompt + "\n\n" + user_query},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=5000,      # ✅ longer answer
    temperature=0.7,         # creativity
    top_p=0.9,               # better sampling
    do_sample=True           # enable sampling
)
response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True
)

print(response)