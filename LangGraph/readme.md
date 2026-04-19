#firat install pip install langgraph langchain-google-genai geopy requests to connect llm to langgraph
#read the docs for more info : https://ai.google.dev/gemini-api/docs/langgraph-example


import os

Read your API key from the environment variable or set it manually
api_key = os.getenv("GEMINI_API_KEY")

from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI

Create LLM class
llm = ChatGoogleGenerativeAI(
    model= "gemini-3-flash-preview",
    temperature=1.0,
    max_retries=2,
    google_api_key=api_key,
)

Bind tools to the model
model = llm.bind_tools([get_weather_forecast])

Test the model with tools
res=model.invoke(f"What is the weather in Berlin on {datetime.today()}?")

print(res)