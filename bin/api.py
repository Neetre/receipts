import os
from typing import Optional, Dict, Any
import argparse
from pathlib import Path
from datetime import date, datetime, timedelta

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request, BackgroundTasks, Depends
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import uvicorn
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt

from data_manager import SQLiteDBManager
from fastapi import status
from pydantic import EmailStr
import hashlib


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
    ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}

class ReceiptItem(BaseModel):
    name: str
    quantity: int = 1
    price: float

class ReceiptData(BaseModel):
    id: str = None
    date: str
    total_amount: float = Field(..., alias="totalAmount")
    merchant: str
    items: List[ReceiptItem]

class UploadReceiptResponse(BaseModel):
    message: str
    receipt_id: str

class SaveReceiptRequest(BaseModel):
    date: str
    totalAmount: float
    merchant: str
    items: List[ReceiptItem]

class UserRegisterRequest(BaseModel):
    username: str
    email: str
    password: str

class UserLoginRequest(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    username: str
    email: EmailStr

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user_id: int
    username: str

# JWT config
SECRET_KEY = "your_secret_key_here"  # Change this to a secure value
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

class Token(BaseModel):
    access_token: str
    token_type: str

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        user_id: int = payload.get("user_id")
        if username is None or user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    db_user = user_db.fetch_user(username)
    if not db_user or db_user[0] != user_id:
        raise credentials_exception
    return db_user

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
async def load_template_page(request: Request):
    token = request.cookies.get("access_token") or request.headers.get("Authorization")
    if not token:
        return RedirectResponse(url="/login-page")
    try:
        # Try to decode the token and get the user
        user = get_current_user(token.replace("Bearer ", ""))
        template_path = Path("templates/receipts.html")
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found at {template_path}")
        return FileResponse(template_path)
    except Exception:
        return RedirectResponse(url="/login-page")


@app.get("/receipt", response_class=HTMLResponse)
async def view_receipt(request: Request, id: str = Query(None, description="Receipt ID"), current_user: tuple = Depends(get_current_user)):
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
    sort_order: str = Query("desc", regex="^(asc|desc)$"),
    current_user: tuple = Depends(get_current_user)
) -> Dict[str, Any]:
    try:
        user_id = current_user[0]
        collection_info = analyze_receipts.qdrant_client.get_collection("receipts")
        scroll_response = analyze_receipts.qdrant_client.scroll(
            collection_name="receipts",
            limit=limit,
            offset=offset,
            with_payload=True,
        )
        
        receipts = [
            {**point.payload, "id": point.id} 
            for point in scroll_response[0] 
            if point.payload.get("user_id") == user_id
        ]
        receipts.sort(
            key=lambda x: x[sort_by],
            reverse=(sort_order.lower() == "desc")
        )
        
        return {
            "total": len(receipts),
            "receipts": receipts
        }
    except Exception as e:
        logger.error(f"Error getting receipts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from PIL import Image

@app.post("/upload_receipt", response_model=UploadReceiptResponse, status_code=201)
async def upload_receipt(file: UploadFile = File(...), background_tasks: BackgroundTasks = None, current_user: tuple = Depends(get_current_user)):
    ext = Path(file.filename).suffix.lower()
    if ext not in Config.ALLOWED_EXTENSIONS:
        logger.warning(f"Rejected file upload with extension: {ext}")
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    if file.spool_max_size and file.spool_max_size > Config.MAX_FILE_SIZE:
        logger.warning(f"Rejected file upload due to size: {file.spool_max_size}")
        raise HTTPException(status_code=413, detail="File too large.")
    contents = await file.read()
    if len(contents) > Config.MAX_FILE_SIZE:
        logger.warning(f"Rejected file upload due to size: {len(contents)}")
        raise HTTPException(status_code=413, detail="File too large.")
    file_path = Path("../data/input") / file.filename
    Path("../data/input").mkdir(exist_ok=True)    
    with open(file_path, "wb") as f:
        f.write(contents)

    def process_receipt():
        try:
            user_id = current_user[0]
            image = Image.open(file_path)
            receipt_text = analyze_receipts.analyze_with_gemini(image)
            logger.info(f"Extracted text from receipt: {receipt_text}")
            embedding = analyze_receipts.generate_embedding(receipt_text)
            identified_data = analyze_receipts.identify_data(receipt_text)
            receipt_data = analyze_receipts.text_to_dict(identified_data)
            # Add user_id to receipt data
            receipt_data_dict = receipt_data.dict() if hasattr(receipt_data, 'dict') else receipt_data
            receipt_data_dict["user_id"] = user_id
            logger.info(f"Generated receipt ID: {receipt_data.id}")
            analyze_receipts.store_receipt(receipt_data, embedding, receipt_text, user_id=user_id)
        except Exception as e:
            logger.error(f"Error processing receipt in background: {e}")

    if background_tasks is not None:
        background_tasks.add_task(process_receipt)
        return {"message": "Receipt processing started in background", "receipt_id": file.filename}
    else:
        process_receipt()
        return {"message": "Receipt processed successfully", "receipt_id": file.filename}


@app.get("/search_receipts")
async def search_receipts(
    query: str = Query(..., min_length=1),
    limit: int = Query(5, ge=1, le=100),
    current_user: tuple = Depends(get_current_user)
) -> Dict[str, list]:
    try:
        user_id = current_user[0]
        embedding = analyze_receipts.generate_embedding(query)
        search_results = analyze_receipts.qdrant_client.search(
            collection_name="receipts",
            query_vector=embedding,
            limit=limit
        )
        # Filter results by user_id
        filtered_results = [
            {**point.payload, "id": point.id} 
            for point in search_results 
            if point.payload.get("user_id") == user_id
        ]
        return {"results": filtered_results}
    except Exception as e:
        logger.error(f"Error searching receipts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/searchPage", response_class=HTMLResponse)
async def view_search_page(request: Request, id: str = Query(None, description="Receipt ID"), current_user: tuple = Depends(get_current_user)):
    try:
        return templates.TemplateResponse(
            "searchPage.html", 
            {"request": request, "id": id}
        )
    except Exception as e:
        logger.error(f"Error loading receipt view: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search_similar_receipts")
async def search_similar_receipts(id: str = Query(None, description="Receipt ID"), current_user: tuple = Depends(get_current_user)):
    user_id = current_user[0]
    search_results = analyze_receipts.qdrant_client.search(
        collection_name="receipts",
        query_vector=analyze_receipts.generate_embedding(id),
        limit=5
    )
    receipts = []
    for point in search_results:
        # Only include receipts that belong to the current user
        if point.payload.get("user_id") == user_id:
            receipt = point.payload
            receipt["id"] = point.id
            receipts.append(receipt)

    return {"receipts": receipts}


@app.get("/get_receipt")
async def get_receipt(id: str = Query(None, description="Receipt ID"), current_user: tuple = Depends(get_current_user)):
    if not id:
        raise HTTPException(status_code=400, detail="Receipt ID is required")
        
    user_id = current_user[0]
    receipt = analyze_receipts.qdrant_client.retrieve(
        collection_name="receipts",
        ids=[id]
    )
    
    if receipt and len(receipt) > 0:
        receipt_data = receipt[0].payload
        # Check if the receipt belongs to the current user
        if receipt_data.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="Access denied to this receipt")
        return {"receipt": receipt_data}
    else:
        return {"error": "Receipt not found"}

@app.post("/save_receipt/{receipt_id}", status_code=200)
async def save_receipt(receipt_id: str, receipt_data: SaveReceiptRequest, current_user: tuple = Depends(get_current_user)):
    try:
        user_id = current_user[0]
        receipt = analyze_receipts.qdrant_client.retrieve(
            collection_name="receipts",
            ids=[receipt_id]
        )
        if not receipt or len(receipt) == 0:
            raise HTTPException(status_code=404, detail="Receipt not found")
        
        payload = receipt[0].payload
        # Check if the receipt belongs to the current user
        if payload.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="Access denied to this receipt")
            
        payload["date"] = receipt_data.date
        payload["total_amount"] = receipt_data.totalAmount
        payload["merchant"] = receipt_data.merchant
        payload["items"] = [item.dict() for item in receipt_data.items]
        analyze_receipts.update_receipt(receipt_id, payload)
        return {"message": "Receipt updated successfully"}
    except Exception as e:
        logger.error(f"Could not update receipt: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Could not update receipt: {str(e)}")


# Initialize user DB manager and ensure table exists
user_db = SQLiteDBManager()
user_db.create_user_table()

# Helper for password hashing

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

@app.post("/register", status_code=status.HTTP_201_CREATED)
async def register_user(user: UserRegisterRequest):
    try:
        if user_db.fetch_user(user.username):
            raise HTTPException(status_code=400, detail="Username already exists.")
        hashed_pw = hash_password(user.password)
        user_db.insert_user(user.username, user.email, hashed_pw)
        db_user = user_db.fetch_user(user.username)
        user_id, username, email, _ = db_user
        access_token = create_access_token(data={"sub": username, "user_id": user_id})
        
        response = RedirectResponse(url="/", status_code=302)
        response.set_cookie(
            key="access_token",
            value=f"Bearer {access_token}",
            httponly=True,
            secure=False,  # Set to True in production with HTTPS
            samesite="lax"
        )
        return response
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=400, detail="Registration failed.")

@app.post("/login", response_model=TokenResponse)
async def login_user(user: UserLoginRequest):
    try:
        print(user)
        db_user = user_db.fetch_user(user.username)
        print(db_user)
        if not db_user:
            raise HTTPException(status_code=401, detail="Invalid username or password.")
        user_id, username, email, db_password = db_user
        if hash_password(user.password) != db_password:
            raise HTTPException(status_code=401, detail="Invalid username or password.")
        access_token = create_access_token(data={"sub": username, "user_id": user_id})
        
        # Return JSON response with token and user details
        return {"access_token": access_token, "token_type": "bearer", "user_id": user_id, "username": username}
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=401, detail="Login failed.")

@app.post("/token", response_model=TokenResponse)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    db_user = user_db.fetch_user(form_data.username)
    if not db_user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    user_id, username, email, db_password = db_user
    if hash_password(form_data.password) != db_password:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": username, "user_id": user_id})
    return {"access_token": access_token, "token_type": "bearer", "user_id": user_id, "username": username}

# API endpoints that return JSON (for programmatic access)
@app.post("/api/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register_user_api(user: UserRegisterRequest):
    try:
        if user_db.fetch_user(user.username):
            raise HTTPException(status_code=400, detail="Username already exists.")
        hashed_pw = hash_password(user.password)
        user_db.insert_user(user.username, user.email, hashed_pw)
        db_user = user_db.fetch_user(user.username)
        user_id, username, email, _ = db_user
        access_token = create_access_token(data={"sub": username, "user_id": user_id})
        return {"access_token": access_token, "token_type": "bearer", "user_id": user_id, "username": username}
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=400, detail="Registration failed.")

@app.post("/api/login", response_model=TokenResponse)
async def login_user_api(user: UserLoginRequest):
    try:
        db_user = user_db.fetch_user(user.username)
        if not db_user:
            raise HTTPException(status_code=401, detail="Invalid username or password.")
        user_id, username, email, db_password = db_user
        if hash_password(user.password) != db_password:
            raise HTTPException(status_code=401, detail="Invalid username or password.")
        access_token = create_access_token(data={"sub": username, "user_id": user_id})
        return {"access_token": access_token, "token_type": "bearer", "user_id": user_id, "username": username}
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=401, detail="Login failed.")

# Public endpoints (no authentication required)
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/login-page", response_class=HTMLResponse)
async def login_page(request: Request):
    try:
        return templates.TemplateResponse("login.html", {"request": request})
    except Exception as e:
        logger.error(f"Error loading login page: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/register-page", response_class=HTMLResponse)
async def register_page(request: Request):
    try:
        return templates.TemplateResponse("register.html", {"request": request})
    except Exception as e:
        logger.error(f"Error loading register page: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
                host=args.host if not args.domain else args.domain,
                port=args.port,
                ws_max_size=10 * 1024 * 1024,
                timeout_keep_alive=30,
                log_level="debug")


if __name__ == "__main__":
    main()
