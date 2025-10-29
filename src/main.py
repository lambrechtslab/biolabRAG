import asyncio
from typing import List

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

origins = [x.strip() for x in os.environ["ALLOW_ORIGINS"].split(",")]

_custom_event_listeners: List[asyncio.Queue[str]] = []
def send_custom_event(message: str, attach:bool=False, bgprint:bool=True) -> None:
    """Send a custom message to any connected frontend listeners."""
    if bgprint:
        if attach:
            print(message, end="")
        else:
            print(message)
    last_message = getattr(send_custom_event, "last_message", "")
    if attach:
        message = last_message + message
    send_custom_event.last_message = message
    for queue in list(_custom_event_listeners):
        queue.put_nowait(message)
        
send_custom_event("Starting RAG...")
rag = RAG(main_model_name = os.environ["MAIN_MODEL_NAME"],
          fast_model_name = os.environ["MINOR_MODEL_NAME"],
          num_ctx = int(os.environ["NUM_CTX"]),
          vectordb_path = os.environ["VECTORDB_PATH"],
          minor_models_device = os.environ["MINOR_MODELS_DEVICE"],
          raw_documents_path = os.environ["RAW_DOCUMENTS_PATH"],
          msgfun=send_custom_event)
rag.ready()
send_custom_event("RAG is ready.")
        
app = FastAPI()

# origins = [
#     "http://127.0.0.1:5173",
#     "http://localhost:5173",
# ]

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


@app.get("/custom-events")
async def custom_events_stream():
    """Stream server-sent events containing custom backend messages."""

    queue: asyncio.Queue[str] = asyncio.Queue()
    _custom_event_listeners.append(queue)

    async def event_generator():
        try:
            while True:
                message = await queue.get()
                yield f"data: {message}\n\n"
        finally:
            if queue in _custom_event_listeners:
                _custom_event_listeners.remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.post("/chat")
async def chat(message: Message):
    print(f"Send message: {message.text}")
    if message.text[0]==':':
        send_custom_event(message.text[1:])
        return StreamingResponse("", media_type="text/plain")
    elif message.text[0]=='+':
        send_custom_event(message.text[1:], attach=True)
        return StreamingResponse("", media_type="text/plain")

    def stream_with_logging():
        collected = []
        for chunk in rag.ask_stream(message.text, retrival_option=message.option):
            collected.append(chunk)
            yield chunk
        print(f"Answer finished. Total {len(''.join(collected))} chars.")

    return StreamingResponse(stream_with_logging(), media_type="text/plain")

@app.post("/reset")
async def reset_dialog():
    """Endpoint to allow the frontend to signal a dialog reset."""
    print("Reset dialog requested from frontend.")
    rag.clear_chat_history()
    # Hook for any backend-side cleanup when conversations reset.
    return {"status": "reset received"}
