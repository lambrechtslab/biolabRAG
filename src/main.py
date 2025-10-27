from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from myRAG import RAG
from fastapi.responses import StreamingResponse
import os
from dotenv import load_dotenv
from pathlib import Path

print("Start backend")
env_path = Path(__file__).resolve().parent.parent / '.env' #find absolute path of ../
load_dotenv(dotenv_path=env_path)

rag = RAG(main_model_name = os.environ["MAIN_MODEL_NAME"],
          num_ctx = int(os.environ["NUM_CTX"]),
          vectordb_path = os.environ["VECTORDB_PATH"],
          minor_models_device = os.environ["MINOR_MODELS_DEVICE"],
          raw_documents_path = os.environ["RAW_DOCUMENTS_PATH"])
rag.ready()
print("RAG is ready.")

app = FastAPI()

# origins = [
#     "http://127.0.0.1:5173",
#     "http://localhost:5173",
# ]
origins = [x.strip() for x in os.environ["ALLOW_ORIGINS"].split(",")]

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
    option: str

@app.post("/chat")
async def chat(message: Message):
    print(f"Send message: {message.text}")

    def stream_with_logging():
        collected = []
        for chunk in rag.ask_stream(message.text, retrival_option=message.option):
            collected.append(chunk)
            yield chunk
        print(f"Got answer: {''.join(collected)}")

    return StreamingResponse(stream_with_logging(), media_type="text/plain")

@app.post("/reset")
async def reset_dialog():
    """Endpoint to allow the frontend to signal a dialog reset."""
    print("Reset dialog requested from frontend.")
    rag.clear_chat_history()
    # Hook for any backend-side cleanup when conversations reset.
    return {"status": "reset received"}
