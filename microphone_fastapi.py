from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
from datetime import datetime
import wave
import logging

app = FastAPI()

# Configuration
UPLOAD_DIR = "audio_clips"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    try:
        # Validate file
        if file.content_type not in ["audio/wav", "application/octet-stream"]:
            raise HTTPException(400, "Only WAV files accepted")
        
        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"clip_{timestamp}.wav"
        filepath = os.path.join(UPLOAD_DIR, filename)
        
        # Save with WAV validation
        with open(filepath, "wb") as f:
            while content := await file.read(1024):
                f.write(content)
        
        # Verify WAV file
        try:
            with wave.open(filepath, 'rb') as wav:
                params = wav.getparams()
                logger.info(f"Received audio: {params}")
        except Exception as e:
            os.remove(filepath)
            raise HTTPException(400, f"Invalid WAV file: {str(e)}")
        
        return JSONResponse({
            "status": "success",
            "filename": filename,
            "path": filepath,
            "timestamp": timestamp
        })
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(500, "Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)