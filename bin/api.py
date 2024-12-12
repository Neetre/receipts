from typing import Optional, List
import argparse

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
import uvicorn

from receipts import AnalyzeReceipts

app = FastAPI()

analyze_receipts = AnalyzeReceipts()


@app.get("/")
def load_template_page():
    return FileResponse("index.html")


@app.get("/receipts/")
async def get_all_receipts(
    offset: int = 0,
    limit: int = 10,
    sort_by: str = "date",
    sort_order: str = "desc"
):
    collection_info = analyze_receipts.qdrant_client.get_collection("receipts")
    total_points = collection_info.points_count
    
    scroll_response = analyze_receipts.qdrant_client.scroll(
        collection_name="receipts",
        limit=limit,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )
    
    receipts = []
    for point in scroll_response[0]:
        receipt = point.payload
        receipt["id"] = point.id
        receipts.append(receipt)
        
    receipts.sort(
        key=lambda x: x[sort_by],
        reverse=(sort_order.lower() == "desc")
    )
    
    return {
        "total_points": total_points,
        "receipts": receipts
    }

@app.post("/upload_receipt/")
async def upload_receipt(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        receipt_text = analyze_receipts.extract_text(contents, False)
        embedding = analyze_receipts.generate_embedding(receipt_text)

        identified_data = analyze_receipts.identify_data(receipt_text)
        receipt_data = analyze_receipts.text_to_dict(identified_data)
        
        analyze_receipts.store_receipt(receipt_data, embedding, receipt_text)
        return {"message": "Receipt processed successfully", "receipt_id": receipt_data.id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Receipt processing error: {str(e)}")


@app.get("/search_receipts/{receipt_id}")
async def search_receipts(
    receipt_id: Optional[str] = None,
    merchant: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    min_amount: Optional[float] = None,
    max_amount: Optional[float] = None,
    limit: int = Query(5, ge=1, le=100)
):
    query = {}


    if receipt_id:
        query["receipt_id"] = receipt_id
    if merchant:
        query["merchant"] = merchant
    if date_from and date_to:
        query["date"] = {"$gte": date_from, "$lte": date_to}
    if min_amount and max_amount:
        query["total_amount"] = {"$gte": min_amount, "$lte": max_amount}

    search_results = analyze_receipts.qdrant_client.search(
        collection_name="receipts",
        query=analyze_receipts.generate_embedding(query),
        limit=limit
    )

    return {"search_results": search_results}


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


@app.get("/get_receipt/{receipt_id}")
async def get_receipt(receipt_id: str):
    receipt = analyze_receipts.qdrant_client.retrieve(
        collection_name="receipts",
        ids=[receipt_id]
    )
    # print(receipt[0].payload)
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
        "--port",
        type=int,
        default=8000,
        help="Port for the API server"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    uvicorn.run(app, host=args.ip_address if not args.domain else args.domain, port=args.port)


if __name__ == "__main__":
    main()
