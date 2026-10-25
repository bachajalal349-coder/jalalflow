import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq

app = FastAPI(title="Jalal Flow AI API")

# Initialize Groq with your secret key
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"status": "AI API is live! Go to /docs to test it."}

@app.post("/ask")
def ask_ai(data: PromptRequest):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": data.prompt}
            ],
            model="llama-3.1-8b-instant",
        )
        return {"response": chat_completion.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
