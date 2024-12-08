#!/bin/bash

cd ..
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# da rimuovere in caso
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
    echo "Docker installed successfully."
fi

if ! command -v tesseract &> /dev/null; then
    echo "Tesseract-OCR not found. Installing Tesseract-OCR..."
    sudo apt-get update
    sudo apt-get install -y tesseract-ocr
    sudo apt install libtesseract-dev
    echo "Tesseract-OCR installed successfully."
fi

echo "Running Docker container in detached mode..."
docker run -d -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant

cd bin
echo "Running the main application..."
python3 ./api.py --host 0.0.0.0 --port 5000
