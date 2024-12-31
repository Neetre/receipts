REM filepath: /run.bat
@echo off

REM Check if Tesseract is installed
where tesseract >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Tesseract-OCR...
    winget install UB-Mannheim.TesseractOCR
    echo Tesseract-OCR installed successfully.
) else (
    echo Tesseract-OCR is already installed.
)

REM Check if Docker container exists and start if needed
docker ps -q -f name=qdrant >nul 2>&1
if %errorlevel% equ 0 (
    echo Docker container is already running.
) else (
    echo Docker container is not running. Starting container...
    docker run -d -p 6333:6333 -p 6334:6334 --name qdrant -v "%cd%\qdrant_storage:/qdrant/storage:z" qdrant/qdrant
)

REM Change directory to bin
cd bin

REM Install Python dependencies
echo Installing Python dependencies...
pip install -r ..\requirements.txt

REM Run the Flask application
echo Running the main application...
python api.py --host 0.0.0.0 --port 8000

pause