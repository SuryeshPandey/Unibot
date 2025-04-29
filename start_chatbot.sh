#!/bin/bash

# Step 1: Check if container exists
if [ "$(docker ps -a -q -f name=chatbot_container)" ]; then
    echo "✅ Container exists. Starting chatbot_container..."
    docker start -ai chatbot_container
else
    echo "🚀 Container not found. Running new container from image..."
    docker run -it -p 5000:5000 --name chatbot_container -v $(pwd):/app chatbot-neuralcoref /bin/bash
fi
