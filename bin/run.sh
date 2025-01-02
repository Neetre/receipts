#!/bin/bash

cd /sites/receipts/

cd ..
pwd
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# da rimuovere in caso
# if ! command -v docker &> /dev/null; then
#     echo "Docker not found. Installing Docker..."
#     curl -fsSL https://get.docker.com -o get-docker.sh
#     sh get-docker.sh
#     rm get-docker.sh
#     echo "Docker installed successfully."
# fi

if ! command -v tesseract &> /dev/null; then
    echo "Tesseract-OCR not found. Installing Tesseract-OCR..."
    sudo apt-get update
    sudo apt-get install -y tesseract-ocr
    sudo apt install libtesseract-dev
    echo "Tesseract-OCR installed successfully."
fi

if [ "$(docker ps -q -f name=qdrant)" ]; then
    echo "Docker container is already running."
else
    echo "Docker container is not running. Starting container..."
    docker run -d -p 6333:6333 -p 6334:6334 --name qdrant -v "$(pwd)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant 
fi

cd bin

echo "Installing Python dependencies..."
pip install -r ../requirements.txt

echo "Running the main application..."
uvicorn api:app --reload --host 0.0.0.0 --port 8000
