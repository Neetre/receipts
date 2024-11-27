import os
import json
import io
import datetime
from typing import List, Optional, Dict
from pydantic import BaseModel
import uuid

from fastapi import FastAPI, UploadFile, File
import uvicorn

from qdrant_client import QdrantClient
from qdrant_client.http import models
import pytesseract
from PIL import Image
import numpy as np

from transformers import AutoTokenizer, AutoModel
from groq import Groq
import torch

from processing import ReceiptProcessor

from dotenv import load_dotenv
load_dotenv()
import logging
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
    
# image_path: Optional[str]


class ReceiptResponse(BaseModel):
    receipts: List[dict]
    total_count: int


class AnalyzeReceipts:
    def __init__(self):
        self.qdrant_client = QdrantClient(host="localhost", port=6333)
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.client_groq = Groq(api_key=GROQ_API_KEY)
        self.processor = ReceiptProcessor()
        self.init_collection()

    def init_collection(self):
        # if self.qdrant_client.collection_exists("receipts"):
        #     self.qdrant_client.delete_collection("receipts")

        self.qdrant_client.recreate_collection(
            collection_name="receipts",
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            )
        )

    def read_saved_receipts(self):
        collection_info = self.qdrant_client.get_collection("receipts")
        total_points = collection_info.points_count
        # da finire

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
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings[0].numpy().tolist()
    
    def clean_date(self, data_json: str):
        date_parts = data_json.replace("/", "-").split("-") if data_json else None
        if date_parts and len(date_parts) == 3:
            date_parts[0], date_parts[2] = date_parts[2], date_parts[0]
            date = "-".join(date_parts)
        print("\nDATE: ", date)
        return date
    
    def clean_total_amount(self, total_amount: str):
        if type(total_amount) is not float:
            total_amount = float(total_amount.replace("€", ""))
            print("\nTOTAL AMOUNT: ", total_amount)
        return total_amount
    
    def generate_uuid_from_text(self, text):
        namespace_uuid = uuid.NAMESPACE_DNS
        return str(uuid.uuid5(namespace_uuid, text))

    def define_data_dict(self, raw_text, text):
        try:
            text = text.strip()
            if not text.startswith('{'):
                text = text[text.find('{'):]
            if not text.endswith('}'):
                text = text[:text.rfind('}')+1]
            text_json = json.loads(text)
            print(text_json)

            date = self.clean_date(text_json.get("date"))
            self.clean_total_amount(text_json.get("total_amount"))
            # print("DATE: ", date)

            unique_id = self.generate_uuid_from_text(raw_text)
            print(unique_id)  # old: str(abs(hash(raw_text)))[:8]

            return Receipts(
            id=unique_id,
            date=date,
            total_amount=text_json.get("total_amount"),
            merchant=text_json.get("merchant"),
            items=text_json.get("items", [])
            )
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Could not parse JSON response. Error: {e}")
            return Receipts(
            id=unique_id,
            date=None,
            total_amount=None,
            merchant=None,
            items=[]
            )

    def text_to_dict(self, text: str) -> Receipts:
        # old prompt: Given the identified data, return a JSON object with the date, total amount, merchant, and items fields (dictionary with name, number and price). If any of these fields are missing or they seeem incorrect, leave them blank. And dont add comments in the json part.
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
        # ic(result)
        return self.define_data_dict(text, result)
    
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
    
    def extract_text(self, image_bytes, is_file: bool):
        if not is_file:
            image = Image.open(io.BytesIO(image_bytes))
        else:
            print("Nome foto: ", image_bytes)
            image = Image.open(image_bytes)
        image = self.processor.preprocess_receipt(image)
        text = pytesseract.image_to_string(image, lang="ita")
        return text

    def scan_receipt(self, file: str):
        text = self.extract_text(file, True)
        identified_data = self.identify_data(text)
        embedding = self.generate_embedding(identified_data)
        
        receipt_data = self.text_to_dict(identified_data)
        print("ID: ", receipt_data.id)
        self.store_receipt(receipt_data, embedding, identified_data)

app = FastAPI()
analyze_receipts = AnalyzeReceipts()

@app.post("/upload_receipt/")
async def upload_receipt(file: UploadFile = File(...)):
    contents = await file.read()
    receipt_text = analyze_receipts.extract_text(contents, False)
    embedding = analyze_receipts.generate_embedding(receipt_text)

    identified_data = analyze_receipts.identify_data(receipt_text)
    receipt_data = analyze_receipts.text_to_dict(identified_data)
    
    analyze_receipts.store_receipt(receipt_data, embedding, receipt_text)
    return {"message": "Receipt processed successfully", "receipt_id": receipt_data.id}


@app.get("/search_similar_receipts/{receipt_id}")
async def search_similar_receipts(receipt_id: str):
    search_results = analyze_receipts.qdrant_client.search(
        collection_name="receipts",
        query_vector=analyze_receipts.generate_embedding(receipt_id),
        limit=5
    )
    return {"similar_receipts": search_results}


@app.get("/get_receipt/{receipt_id}")
async def get_receipt(receipt_id: str):
    receipt = analyze_receipts.qdrant_client.retrieve(
        collection_name="receipts",
        ids=[receipt_id]
    )
    return receipt[0] if receipt else {"error": "Receipt not found"}
    

def get_files(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"):
                yield os.path.join(root, file)


def main():
    c = 0
    analyze_receipts = AnalyzeReceipts()
    # for file in get_files("../data/Route/Cibo"):
    #    if c == 2:
    #        break
#
    #    print(file)
    #    analyze_receipts.scan_receipt(file)
    #    c += 1
    analyze_receipts.scan_receipt("../data/clan/Route/Cibo/2024-08-07Lidl2-1.png")
    

if __name__ == "__main__":
    main()
    analyze_receipts = AnalyzeReceipts()
    # uvicorn.run(app, host="0.0.0.0", port=8000)
