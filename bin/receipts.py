import os
import json
import io
import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models
import pytesseract
from PIL import Image
import numpy as np
from functools import wraps
import time

from transformers import AutoTokenizer, AutoModel
from groq import Groq
import torch
import google.generativeai as genai

from processing import ReceiptProcessor

from dotenv import load_dotenv
load_dotenv()
import logging
from icecream import ic
ic.enable()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

logger = logging.getLogger("receipts")
logging.basicConfig(level=logging.INFO)

class Receipts(BaseModel):
    id: str
    date: Optional[datetime.datetime]
    total_amount: Optional[float]
    merchant: Optional[str]
    items: List[dict]

class ReceiptResponse(BaseModel):
    receipts: List[dict]
    total_count: int

model_gemini = genai.GenerativeModel(model_name="gemini-1.5-flash")
prompt = "Extract the text from the receipt."

class ReceiptProcessingError(Exception):
    pass

class ReceiptPipeline:
    def __init__(self, max_attempts=3, base_delay=1):
        self.max_attempts = max_attempts
        self.base_delay = base_delay

    def run(self, func, *args, **kwargs):
        attempt = 0
        while attempt < self.max_attempts:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                attempt += 1
                logger.warning(f"Attempt {attempt} failed: {e}")
                if attempt == self.max_attempts:
                    logger.error(f"Max attempts reached for {func.__name__}")
                    raise ReceiptProcessingError(f"Failed after {self.max_attempts} attempts: {e}")
                delay = self.base_delay * (2 ** attempt)
                time.sleep(delay)
        return None

class AnalyzeReceipts:
    def __init__(
        self,
        qdrant_client: Optional[QdrantClient] = None,
        tokenizer: Optional[Any] = None,
        model: Optional[Any] = None,
        client_groq: Optional[Groq] = None,
        processor: Optional[ReceiptProcessor] = None,
        embedding_size: int = 384,
        retry_max_attempts: int = 3,
        retry_base_delay: int = 1
    ):
        self.qdrant_client = qdrant_client or QdrantClient(host="localhost", port=6333)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = model or AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.client_groq = client_groq or Groq(api_key=GROQ_API_KEY)
        self.processor = processor or ReceiptProcessor()
        self.embedding_size = embedding_size
        self.pipeline = ReceiptPipeline(max_attempts=retry_max_attempts, base_delay=retry_base_delay)
        self.init_collection()

    def init_collection(self):
        if self.qdrant_client.collection_exists("receipts"):
            return
        self.qdrant_client.recreate_collection(
            collection_name="receipts",
            vectors_config=models.VectorParams(
                size=self.embedding_size,
                distance=models.Distance.COSINE
            )
        )

    def generate_with_groq(self, messages: List[Dict[str, str]]) -> str:
        completion = self.client_groq.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
            temperature=0.7,
        )
        return completion.choices[0].message.content

    def identify_data(self, receipt_text):
        prompt = [
            {"role": "system", "content": "Given the receipt text, identify the date, 'total_amount', merchant, and items. If you can't identify any of these, leave it blank."},
            {"role": "user", "content": receipt_text}
        ]
        return self.generate_with_groq(prompt)

    def generate_embedding(self, text: str) -> list:
        inputs = self.tokenizer(text=text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings[0].numpy().tolist()

    def clean_date(self, date_str: str) -> Optional[datetime.datetime]:
        if not date_str:
            return None
        try:
            # Try parsing common formats
            for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
                try:
                    return datetime.datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            logger.warning(f"Unrecognized date format: {date_str}")
            return None
        except Exception as e:
            logger.error(f"Error parsing date: {e}")
            return None

    def clean_total_amount(self, total_amount: Any) -> Optional[float]:
        if total_amount is None:
            return None
        try:
            if isinstance(total_amount, float):
                return total_amount
            if isinstance(total_amount, int):
                return float(total_amount)
            if isinstance(total_amount, str):
                cleaned = total_amount.replace("€", "").replace(",", ".").strip()
                return float(cleaned)
        except Exception as e:
            logger.error(f"Error cleaning total_amount: {e}")
            return None
        return None

    def generate_uuid_from_text(self, text):
        namespace_uuid = uuid.NAMESPACE_DNS
        return str(uuid.uuid5(namespace_uuid, text))

    def define_data_dict(self, raw_text, text) -> Receipts:
        unique_id = self.generate_uuid_from_text(raw_text)
        try:
            text = text.strip()
            if not text.startswith('{'):
                text = text[text.find('{'):]
            if not text.endswith('}'):
                text = text[:text.rfind('}')+1]
            text_json = json.loads(text)
            logger.info(f"Parsed JSON: {text_json}")
            date = self.clean_date(text_json.get("date"))
            total_amount = self.clean_total_amount(text_json.get("total_amount"))
            merchant = text_json.get("merchant")
            items = text_json.get("items", [])
            return Receipts(
                id=unique_id,
                date=date,
                total_amount=total_amount,
                merchant=merchant,
                items=items
            )
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Could not parse JSON response. Error: {e}")
            return Receipts(
                id=unique_id,
                date=None,
                total_amount=None,
                merchant=None,
                items=[]
            )

    def text_to_dict(self, text: str) -> Receipts:
        prompt = [
            {"role": "system", "content": '''
Given a transaction or receipt input, create a JSON object with the following structure:
{
    "date": "",
    "total_amount": "",
    "merchant": "",
    "items": [
        {
            "name": "",
            "number": "",
            "price": ""
        }
    ]
}

Rules:
- If any field is missing or appears incorrect, leave it as an empty string
- Do not include any comments in the JSON
- Ensure the JSON is valid and well-formatted
- Respond ONLY with the JSON object
'''},
            {"role": "user", "content": text}
        ]
        result = self.generate_with_groq(prompt)
        return self.define_data_dict(text, result)

    def store_receipt(self, receipt_data: Receipts, embedding: np.array, receipt_text: str):
        try:
            self.qdrant_client.upsert(
                collection_name="receipts",
                points=[
                    models.PointStruct(
                        id=receipt_data.id,
                        vector=embedding,
                        payload={
                            "date": receipt_data.date.isoformat() if receipt_data.date else None,
                            "total_amount": receipt_data.total_amount,
                            "merchant": receipt_data.merchant,
                            "items": receipt_data.items,
                            "text": receipt_text
                        }
                    )
                ]
            )
            return receipt_data.id
        except Exception as e:
            logger.error(f"Error storing receipt: {e}")
            raise ReceiptProcessingError(f"Failed to store receipt: {e}")

    def update_receipt(self, receipt_id: str, receipt_data: dict):
        try:
            self.qdrant_client.update(
                collection_name="receipts",
                points=[
                    {
                        "id": receipt_id,
                        "payload": receipt_data
                    }
                ]
            )
        except Exception as e:
            logger.error(f"Error updating receipt: {e}")
            raise ReceiptProcessingError(f"Failed to update receipt: {e}")

    def extract_text(self, image_bytes, is_file: bool):
        try:
            if not is_file:
                image = Image.open(io.BytesIO(image_bytes))
            else:
                logger.info(f"Nome foto: {image_bytes}")
                image = Image.open(image_bytes)
            image = self.processor.preprocess_receipt(image)
            text = pytesseract.image_to_string(image, lang="ita")
            return text
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            raise ReceiptProcessingError(f"Failed to extract text: {e}")

    def scan_receipt(self, file: str):
        text = self.pipeline.run(self.extract_text, file, True)
        identified_data = self.pipeline.run(self.identify_data, text)
        embedding = self.pipeline.run(self.generate_embedding, identified_data)
        receipt_data = self.pipeline.run(self.text_to_dict, identified_data)
        logger.info(f"ID: {receipt_data.id}")
        self.pipeline.run(self.store_receipt, receipt_data, embedding, identified_data)

    @staticmethod
    def analyze_with_gemini(image: Image) -> Optional[str]:
        try:
            response = model_gemini.generate_content([image, prompt])
            return str(response.text)
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return None


def get_files(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                yield os.path.join(root, file)


def main():
    analyze_receipts = AnalyzeReceipts()
    file = input("Insert the name of the receipt file---> ").strip()
    if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
        file_path = f"../data/{file}"
        try:
            analyze_receipts.scan_receipt(file_path)
        except ReceiptProcessingError as e:
            logger.error(f"Processing failed: {e}")
    else:
        logger.error(f"File extension invalid: {file.split('.')[-1]}")

if __name__ == "__main__":
    main()
