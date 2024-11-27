from fastapi import FastAPI, UploadFile, File
import uvicorn
from receipts import AnalyzeReceipts, ReceiptResponse

app = FastAPI()
analyze_receipts = AnalyzeReceipts()


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
    
    return ReceiptResponse(
        receipts=receipts,
        total_count=total_points
    )

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

           
def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
