#!/bin/bash

# Navigate to the project directory
cd /Users/suryeshpandey/Downloads/chatbot_project/

# Activate the virtual environment
echo "Activating virtual environment..."
source chatbot_env/bin/activate

# Start the llama-server in the background
echo "Starting llama-server..."
llama-server --model /Users/suryeshpandey/Downloads/chatbot_project/models/mistral-7b-instruct-v0.3-q4_k_m.gguf -c 2048 --n-predict 100 &

# Wait for the llama-server to start
echo "Waiting for llama-server to start..."
sleep 10

# Check if llama-server is running
if ! ps aux | grep -q '[l]lama-server'; then
    echo "Error: llama-server failed to start."
    exit 1
fi

# Find first available port between 5000–5010
PORT=$(comm -23 <(seq 5000 5010) <(lsof -i -P -n | grep LISTEN | awk '{print $9}' | cut -d: -f2 | sort -u | grep -E '^[0-9]+$') | head -n 1)

# Start the Flask chatbot server in the background
echo "Starting Flask chatbot server on port $PORT..."
FLASK_APP=chatbot.py flask run --port=$PORT &

# Wait for the Flask server to start
echo "Waiting for Flask server to start..."
sleep 10

# Check if Flask server is running
if ! ps aux | grep -q '[f]lask run'; then
    echo "Error: Flask server failed to start."
    exit 1
fi

# Keep the script running to keep the servers alive
echo "Chatbot is running on http://127.0.0.1:$PORT. Press Ctrl+C to stop."
wait