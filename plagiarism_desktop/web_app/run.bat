@echo off
cd /d "%~dp0"
echo Installing dependencies...
pip install -r requirements-web.txt
echo Starting Plagiarism Detector Web App...
echo Access at: http://localhost:8000
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8081
pause
