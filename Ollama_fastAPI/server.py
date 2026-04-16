from fastapi import FastAPI, Body

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}

@app.get("/contact-us")
def contact_us():
    return {"message": "Contact us at sainivivek5392@gmail.com"}



from ollama import chat
from ollama import Client

client = Client(
    host='http://localhost:11434/',
)

@app.post("/aksme")
def aksme(message: str = Body(..., description="The message to send to the model")):
    response = client.chat(model='gemma2:2b', messages=[
    {
        'role': 'user',
        'content': message,
    },
    ])
    return response.message.content

