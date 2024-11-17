# Receipts

[![GitHub issues](https://img.shields.io/github/issues/Neetre/receipts)](https://github.com/Neetre/receipts/issues)
[![GitHub forks](https://img.shields.io/github/forks/Neetre/receipts)](https://github.com/Neetre/receipts/network)
[![GitHub stars](https://img.shields.io/github/stars/Neetre/receipts)](https://github.com/Neetre/receipts/stargazers)
[![GitHub license](https://img.shields.io/github/license/Neetre/receipts)](https://github.com/Neetre/receipts/blob/main/LICENSE)

## Description

This project consists of a receipt management system. It allows users to upload receipts, view them, and download them. The receipts are stored in a Qdrant database, which allows for fast search and retrieval of receipts. The project is built using the FastAPI framework and the Qdrant database.
It also uses the Groq API to extract,analyze and format text from the receipts.

[!IMPORTANT]
The web application is not yet implemented. The current implementation is a normal python script that interacts with the Qdrant database and the Groq API.

## Installation

### Requirements

- Python > 3.9
- Docker

### Environment setup

1. Create and activate a virtual environment:

   **Linux/macOS:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   **Windows:**

    ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### API Configuration

Create a `.env` file in the project root with the following:

```bash
GROQ_API_KEY="your_groq_api_key"
```

- Get your Groq API key from the [Groq Console](https://console.groq.com/playground)

### Start the Qdrant database

Start the Qdrant database using Docker:

```bash
docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant
```

This will start the Qdrant database on `localhost:6333`.
And the Qdrant dashboard on `localhost:6334`.

## Usage

### Upload a receipt

Add a receipt in the `data` folder and run the following command:

```bash
cd bin
python receipts.py
```

This will upload the receipt to the Qdrant database.

### Search for a receipt

Cooming soon... :thumbsup:

## Licence

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

[Neetre](https://github.com/Neetre)