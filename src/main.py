from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from myRAG import RAG
rag = RAG()
rag.ready()
print("RAG is ready.")

app = FastAPI()

origins = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["*"],
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    text: str

@app.post("/chat")
async def chat(message: Message):
    # Just echo back for demo
    print(f"Send message: {message.text}")
    aws = rag.ask(message.text)
    print(f"Got answer: {aws}")
    return {"reply": aws}
    # return {"reply": f"You said: {message.text}"} #For debug
