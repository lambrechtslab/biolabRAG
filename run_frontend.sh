echo "Run in local: ssh -L 5173:localhost:5173 -L 8000:localhost:8000 -J vsc $USER@$HOSTNAME"
cd chat-frontend
npm run dev
