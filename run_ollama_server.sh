OLLAMA_HOST=127.0.0.1:11434 CUDA_VISIBLE_DEVICES=0 ollama serve >& ollama_server1.log &
OLLAMA_HOST=127.0.0.1:11435 CUDA_VISIBLE_DEVICES=1 ollama serve >& ollama_server2.log &
uvicorn main:app --reload --port 8000
# npm run dev
