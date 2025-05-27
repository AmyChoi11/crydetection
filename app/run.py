from fastapi import FastAPI
from nicegui import ui
import uvicorn
app = FastAPI()

@ui.page('/ui')
def test_page():
    ui.label('This is working!')

ui.run_with(app)

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=5001)