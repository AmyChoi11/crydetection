cd "$(dirname "$0")/app"
echo "Starting Baby Cry Detection System..."
python -m uvicorn main:app --host 0.0.0.0 --port 5000 --reload &
echo "Server started on http://localhost:5000"
echo "UI available at http://localhost:5000/ui"