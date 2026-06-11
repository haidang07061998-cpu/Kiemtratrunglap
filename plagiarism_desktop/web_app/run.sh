cd "$(dirname "$0")"
pip install -r requirements-web.txt
echo "Access at: http://localhost:8000"
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
