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
import cv2
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


class ReceiptResponse(BaseModel):
    receipts: List[dict]
    total_count: int


class ReceiptProcessor:
    def __init__(self):
        self.debug_mode = False

    def detect_receipt(self, image: Image) -> Image:
        # identify the receipt in the image and crop it
        image_np = np.array(image)  # image is already grayscale
        blurred = cv2.GaussianBlur(image_np, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)

        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        screenCnt = None

        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                screenCnt = approx
                break

        if screenCnt is None:
            raise ValueError("No receipt found in the image")

        rect = cv2.minAreaRect(screenCnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        width = int(rect[1][0])
        height = int(rect[1][1])

        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height-1],
                    [0, 0],
                    [width-1, 0],
                    [width-1, height-1]], dtype="float32")

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image_np, M, (width, height))

        return Image.fromarray(warped)

    def detect_brightness_contrast(self, image: Image):
        image_np = np.array(image)
        # Convert to grayscale if the image is in color
        if len(image_np.shape) == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        # Calculate brightness as the mean pixel value
        brightness = np.mean(image_np)
        # Calculate contrast as the standard deviation of pixel values
        contrast = np.std(image_np)
        return brightness, contrast

    def preprocess_receipt(self, image: Image) -> Image:
        brightness, contrast = self.detect_brightness_contrast(image)
        print(f"Brightness: {brightness}, Contrast: {contrast}")
        
        image_np = np.array(image)
        brightness = -(brightness-100)
        ic(brightness)
        contrast = 1.5
        adjusted_image = cv2.addWeighted(image_np, contrast, np.zeros(image_np.shape, image_np.dtype), 0, brightness)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        adjusted_image = cv2.filter2D(adjusted_image, -1, kernel) 
        image = Image.fromarray(adjusted_image)
        brightness, contrast = self.detect_brightness_contrast(image)
        print(f"Adjusted Brightness: {brightness}, Adjusted Contrast: {contrast}")
        adjusted_image = self.detect_receipt(image)
        cv2.imwrite("../data/adjusted_image.png", adjusted_image)
        return image

class AnalyzeReceipts:
    
    def __init__(self):
        self.qdrant_client = QdrantClient(host="localhost", port=6333)
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self. model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.client_groq = Groq(api_key=GROQ_API_KEY)
        self.processor = ReceiptProcessor()
        self.init_collection()

    def init_collection(self):
        if self.qdrant_client.collection_exists("receipts"):
            self.qdrant_client.delete_collection("receipts")

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
            {"role": "system", "content": "Given the receipt text, identify the date, total amount, merchant, and items. If you can't identify any of these, leave it blank."},
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
            {"role": "system", "content": "Given the identified data, convert it to a dictionary. If you can't identify any of these, leave it blank."},
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

    def is_receipt_valid(self, receipt_data: Receipts):
        if receipt_data.date is None or receipt_data.total_amount is None or receipt_data.merchant is None or receipt_data.items is None:
            return False
        return True

    def scan_receipt(self, file: str):
        print(file)
        image = Image.open(file)
        image = self.processor.preprocess_receipt(image)
        text = pytesseract.image_to_string(image, lang="ita")
        ic(text)
        identified_data = self.identify_data(text)
        ic(identified_data)
        # embedding = self.embed_text(identified_data)
        receipt_data = self.text_to_dict(identified_data)
        ic(receipt_data)
        # if self.is_receipt_valid(receipt_data):
        # self.store_receipt(receipt_data, embedding, identified_data)


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