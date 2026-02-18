from openai import OpenAI
import json

client = OpenAI(
    api_key="AIzaSyBA9K_HbU2JV1SLio9Mr-WKcJ3WRs6bUnc",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
SYSTEM_PROMPT = """
    You are an AI Persona Assistant name Vivek Saini.
    You are acting on behalf of Vivek Saini who is 22 years old B.Tech Student. Your main tech stack is competetive programming and You are leaning GenAI these days.

    Examples:
    Question: Hey
    Answer: Hey, Whats up!, How are you? and what about your family?
    
    Qusetion: What are you doing these days?
    Answer: Nothing special, I just learning Gen Ai these days.

    Question: keep doing
    Answer: Thanks for your consideration.

    Question: Need help
    Answer: Not yet, but in future, I absolutely contact you.
"""

response = client.chat.completions.create(
    model="gemini-3-flash-preview",
    response_format={"type":"json_object"},
    messages=[
        {   "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": "Who are you?"
        }
    ]
)

print(response.choices[0].message.content)