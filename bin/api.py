import os
from typing import Optional, Dict, Any
import argparse
from pathlib import Path
from datetime import date

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

if os.path.exists("../log") is False:
    os.mkdir("../log")

if os.path.exists("../log/receipts.log") is False:
    with open("../log/receipts.log", "w") as f:
        f.write("")


import logging
logging.basicConfig(
    filename="../log/receipts.log",
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

Path("../log").mkdir(exist_ok=True)
Path("../log/receipts.log").touch(exist_ok=True)

from receipts import AnalyzeReceipts

class Config:
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

app = FastAPI(
    title="Receipts API",
    description="API for processing receipts",
    docs_url='/docs',
    redoc_url='/redoc',
    openapi_url='/openapi.json'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=["*"],
    allow_headers=["*"],
)

analyze_receipts = AnalyzeReceipts()

# Static files and templates setup
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.middleware("http")
async def file_size_middleware(request: Request, call_next):
    if request.method == "POST" and "multipart/form-data" in request.headers.get("content-type", ""):
        content_length = int(request.headers.get("content-length", 0))
        if content_length > Config.MAX_FILE_SIZE:
            return JSONResponse(
                status_code=413,
                content={"detail": f"File too large. Maximum size is {Config.MAX_FILE_SIZE // (1024 * 1024)} MB"}
            )
    return await call_next(request)


@app.get("/", response_class=HTMLResponse)
async def load_template_page():
    try:
        template_path = Path("templates/receipts.html")
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found at {template_path}")
        return FileResponse(template_path)
    except Exception as e:
        logger.error(f"Error loading template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/receipt", response_class=HTMLResponse)
async def view_receipt(request: Request, id: str = Query(None, description="Receipt ID")):
    try:
        return templates.TemplateResponse(
            "receipt_view.html", 
            {"request": request, "id": id}
        )
    except Exception as e:
        logger.error(f"Error loading receipt view: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/receipts")
async def get_all_receipts(
    offset: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    sort_by: str = Query("date", regex="^(date|merchant|total_amount)$"),
    sort_order: str = Query("desc", regex="^(asc|desc)$")
) -> Dict[str, Any]:
    try:
        collection_info = analyze_receipts.qdrant_client.get_collection("receipts")
        scroll_response = analyze_receipts.qdrant_client.scroll(
            collection_name="receipts",
            limit=limit,
            offset=offset,
            with_payload=True,
        )
        
        receipts = [{**point.payload, "id": point.id} for point in scroll_response[0]]
        receipts.sort(
            key=lambda x: x[sort_by],
            reverse=(sort_order.lower() == "desc")
        )
        
        return {
            "total": collection_info.points_count,
            "receipts": receipts
        }
    except Exception as e:
        logger.error(f"Error getting receipts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_receipt")
async def upload_receipt(file: UploadFile = File(...)) -> Dict[str, str]:
    try:
        contents = await file.read()
        receipt_text = analyze_receipts.extract_text(contents, False)
        logger.info(f"Extracted text from receipt: {receipt_text}")

        embedding = analyze_receipts.generate_embedding(receipt_text)
        identified_data = analyze_receipts.identify_data(receipt_text)
        receipt_data = analyze_receipts.text_to_dict(identified_data)
        
        receipt_id = analyze_receipts.store_receipt(receipt_data, embedding, receipt_text)
        return {"message": "Receipt processed successfully", "receipt_id": receipt_id}
    except Exception as e:
        logger.error(f"Error processing receipt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search_receipts")
async def search_receipts(
    query: str = Query(..., min_length=1),
    limit: int = Query(5, ge=1, le=100)
) -> Dict[str, list]:
    try:
        embedding = analyze_receipts.generate_embedding(query)
        search_results = analyze_receipts.qdrant_client.search(
            collection_name="receipts",
            query_vector=embedding,
            limit=limit
        )
        return {"results": [{**point.payload, "id": point.id} for point in search_results]}
    except Exception as e:
        logger.error(f"Error searching receipts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search_similar_receipts/{receipt_id}")
async def search_similar_receipts(receipt_id: str):
    search_results = analyze_receipts.qdrant_client.search(
        collection_name="receipts",
        query_vector=analyze_receipts.generate_embedding(receipt_id),
        limit=5
    )
    receipts = []
    for point in search_results:
        receipt = point.payload
        receipt["id"] = point.id
        receipts.append(receipt)

    return {"receipts": receipts}


@app.get("/get_receipt")
async def get_receipt(id: str = Query(None, description="Receipt ID")):
    if not id:
        raise HTTPException(status_code=400, detail="Receipt ID is required")
        
    receipt = analyze_receipts.qdrant_client.retrieve(
        collection_name="receipts",
        ids=[id]
    )
    return {"receipt": receipt[0].payload} if receipt and len(receipt) > 0 else {"error": "Receipt not found"}

@app.get("/save_receipt/{receipt_id}")
async def save_receipt(
    receipt_id: str,
    receipt_data: dict
):
    try:
        receipt = analyze_receipts.qdrant_client.retrieve(
            collection_name="receipts",
            ids=[receipt_id]
        )
        if not receipt or len(receipt) == 0:
            raise HTTPException(status_code=404, detail="Receipt not found")

        receipt_data = receipt_data.dict()

        receipt[0].payload["date"] = receipt_data["date"].isoformat()
        receipt[0].payload["total_amount"] = receipt_data["totalAmount"]
        receipt[0].payload["merchant"] = receipt_data["merchant"]
        receipt[0].payload["items"] = receipt_data["items"]

        analyze_receipts.update_receipt(receipt_id, receipt[0].payload)
        return {"message": "Receipt updated successfully"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not update receipt: {str(e)}")


def parse_args():
    parser = argparse.ArgumentParser(description="API for receipt processing")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0.",
        help="Host for the API server"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="0.0.0.0",
        help='Dominio del server'
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8000,
        help="Port for the API server"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    uvicorn.run(app,
                host=args.ip_address if not args.domain else args.domain,
                port=args.port,
                ws_max_size=10 * 1024 * 1024,
                timeout_keep_alive=30,
                log_level="debug")


if __name__ == "__main__":
    main()
