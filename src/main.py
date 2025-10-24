import asyncio
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from myRAG import RAG
from fastapi.responses import StreamingResponse
rag = RAG()
rag.ready()
print("RAG is ready.")

app = FastAPI()

origins = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
]

_custom_event_listeners: List[asyncio.Queue[str]] = []


def send_custom_event(message: str) -> None:
    """Send a custom message to any connected frontend listeners."""

    for queue in list(_custom_event_listeners):
        queue.put_nowait(message)

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
