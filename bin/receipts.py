import os
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models
import pytesseract
from PIL import Image
import numpy as np
from transformers import AutoTokenizer, AutoModel
from groq import Groq
import torch
import logging
from typing import List, Optional, Dict
import datetime
from dotenv import load_dotenv
load_dotenv()
from icecream import ic
ic.enable()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

logging.basicConfig(
    filename="../log/receipts.log",
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class Receipts(BaseModel):
    id: str
    date: datetime.datetime
    total_amount: float
    merchant: str
    items: List[dict]
    image_path: Optional[str]


class AnalyzeReceipts:
    
    def __init__(self):
        self.qdrant_client = QdrantClient(host="localhost", port=6333)
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self. model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.client_groq = Groq(api_key=GROQ_API_KEY)
        self.init_collection()

    def init_collection(self):
        self.qdrant_client.recreate_collection(
            collection_name="receipts",
            vectors_config=models.VectorParams(
                size=384,
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
            {"role": "system", "content": "Given the receipt text, identify the date, total amount, merchant, and items."},
            {"role": "user", "content": receipt_text}
        ]
        return self.generate_with_groq(prompt)
    
    def embed_text(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.pooler_output
    
    def text_to_dict(self, text: str):
        prompt = [
            {"role": "system", "content": "Given the identified data, convert it to a dictionary."},
            {"role": "user", "content": text}
        ]
        return self.generate_with_groq(prompt)
    
    def store_receipt(self, receipt_data: Receipts, embedding: np.array, receipt_text: str):
        self.qdrant_client.upsert(
            collection_name="receipts",
            points=[
                models.PointStruct(
                    id=receipt_data.id,
                    vector=embedding,
                    payload={
                        "date": receipt_data.date.isoformat(),
                        "total_amount": receipt_data.total_amount,
                        "merchant": receipt_data.merchant,
                        "items": receipt_data.items,
                        "text": receipt_text
                    }
                )
            ]
        )

    def scan_receipt(self, file: str):
        image = Image.open(file)
        text = pytesseract.image_to_string(image)
        ic(text)
        # identified_data = self.identify_data(text)
        # embedding = self.embed_text(identified_data)
        # receipt_data = self.text_to_dict(identified_data)
        # ic(receipt_data)
        # self.store_receipt(receipt_data, embedding, identified_data)


def main():
    analyze_receipts = AnalyzeReceipts()
    analyze_receipts.scan_receipt("../data/receipt2.jpg")


if __name__ == "__main__":
    main()