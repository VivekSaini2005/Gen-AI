from google import genai
from google.genai import types

client = genai.Client(api_key="AIzaSyBA9K_HbU2JV1SLio9Mr-WKcJ3WRs6bUnc")

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    config=types.GenerateContentConfig(
        system_instruction="You are expert in math and give only and only ans math problems. For any other queries just say sorry and return ask related to math"),
    contents="Give me a joke"
)

print(response.text)