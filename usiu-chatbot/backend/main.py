import os
from dotenv import load_dotenv
load_dotenv()
from extract_text import chatbot_response  # Assumes chatbot setup is already safely initialized inside extract_text
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple

app = FastAPI()

# Enable CORS for frontend connection (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with your React domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageRequest(BaseModel):
    user_input: str
    chat_history: List[Tuple[str, str]]

@app.post("/chat")
def chat_endpoint(req: MessageRequest):
    updated_chat_history, _ = chatbot_response(req.user_input, req.chat_history)
    return {"chat_history": updated_chat_history}