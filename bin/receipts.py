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

    def detect_receipt(self, image_np: np.array) -> np.array:
        # identify the receipt in the image and crop it
        # image_np = np.array(image)  # image is already grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if len(image_np.shape) == 3 else image_np
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        edged = cv2.Canny(thresh, 50, 200, apertureSize=3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edged = cv2.dilate(edged, kernel, iterations=1)

        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found in the image")

        # Filter contours by area and aspect ratio
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000:  # Filter out small contours
                continue

            # Check aspect ratio
            rect = cv2.minAreaRect(cnt)
            width, height = rect[1]
            if width == 0 or height == 0:
                continue

            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > 5:  # Filter out extremely elongated contours
                continue

            valid_contours.append(cnt)

        if not valid_contours:
            raise ValueError("No valid receipt contours found")

        receipt_contour = max(valid_contours, key=cv2.contourArea)  # largest valid

        peri = cv2.arcLength(receipt_contour, True)
        approx = cv2.approxPolyDP(receipt_contour, 0.02 * peri, True)

        if len(approx) != 4:  # no 4 corners
            rect = cv2.minAreaRect(receipt_contour)
            approx = cv2.boxPoints(rect)

        approx = self._sort_corners(approx)

        src_pts = approx.astype("float32")
        width = int(max(
            np.linalg.norm(src_pts[0] - src_pts[1]),
            np.linalg.norm(src_pts[2] - src_pts[3])
        ))
        height = int(max(
            np.linalg.norm(src_pts[0] - src_pts[3]),
            np.linalg.norm(src_pts[1] - src_pts[2])
        ))

        dst_pts = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
            ], dtype="float32")

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image_np, M, (width, height))

        return warped
    
    def _sort_corners(self, pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def detect_brightness_contrast(self, image: np.array):
        # Convert to grayscale if the image is in color
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Calculate brightness as the mean pixel value
        brightness = np.mean(image)
        # Calculate contrast as the standard deviation of pixel values
        contrast = np.std(image)
        return brightness, contrast

    def preprocess_receipt(self, image: Image) -> np.array:
        image_np = np.array(image)
        
        brightness, contrast = self.detect_brightness_contrast(image_np)
        #print(f"Brightness: {brightness}, Contrast: {contrast}")
        brightness = -(brightness-100)
        #ic(brightness)
        contrast = 1.5
        adjusted_image = cv2.addWeighted(image_np, contrast, np.zeros(image_np.shape, image_np.dtype), 0, brightness)

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        adjusted_image = cv2.filter2D(adjusted_image, -1, kernel)

        adjusted_image = self.detect_receipt(adjusted_image)
        # cv2.imwrite("../data/adjusted_image.png", adjusted_image)
        return adjusted_image

class AnalyzeReceipts:
    def __init__(self):
        self.qdrant_client = QdrantClient(host="localhost", port=6333)
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
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
        print("Nome foto: ", file)
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