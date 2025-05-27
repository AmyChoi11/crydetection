from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
import librosa
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

# AI model configuration
class Config:
    CLASS_LABELS = [
        'Unwell', 'Sleeping', 'Cry',
        'Laugh', 'Tired', 'Silence'
    ]
    NUM_CLASSES = len(CLASS_LABELS)
    MAX_FRAMES = 50  # Make sure this matches your model
    N_MFCC = 10       # Make sure this matches your model
    SAMPLE_RATE = 22050
    N_FFT = 512
    HOP_LENGTH = 256

# Load the AI model
model = None
mean = None
std = None

def load_model():
    global model, mean, std
    try:
        model = tf.keras.models.load_model('newest_model.keras')
        mean = np.load('mean.npy')
        std = np.load('std.npy')
        logger.info("Model and normalization parameters loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def extract_mfcc(audio_path):
    """Extract MFCC features from audio file"""
    try:
        # Load audio file and resample to expected rate
        y, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE)
        
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=Config.N_MFCC,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH
        ).T  # Transpose to get time on first axis

        # Pad or truncate to expected length
        if mfcc.shape[0] < Config.MAX_FRAMES:
            pad_width = ((0, Config.MAX_FRAMES - mfcc.shape[0]), (0, 0))
            mfcc = np.pad(mfcc, pad_width, mode='constant')
        else:
            mfcc = mfcc[:Config.MAX_FRAMES, :]
        
        # Add batch and channel dimensions
        mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension
        mfcc = np.expand_dims(mfcc, axis=-1)  # Add channel dimension
        
        return mfcc
    
    except Exception as e:
        logger.error(f"Error extracting MFCC: {str(e)}")
        raise

def predict(audio_path):
    """Run prediction on audio file"""
    global model, mean, std
    
    # Ensure model is loaded
    if model is None:
        if not load_model():
            raise Exception("Model could not be loaded")
    
    # Extract features
    mfcc = extract_mfcc(audio_path)
    
    # Normalize
    mfcc_normalized = (mfcc - mean) / (std + 1e-8)
    
    # Predict
    prediction = model.predict(mfcc_normalized, verbose=0)[0]
    
    # Get result
    predicted_idx = np.argmax(prediction)
    predicted_class = Config.CLASS_LABELS[predicted_idx]
    confidence = float(prediction[predicted_idx] * 100)
    
    # Get all probabilities
    all_probs = {label: float(prob * 100) for label, prob in zip(Config.CLASS_LABELS, prediction)}
    
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": all_probs
    }

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
        
        # Save the file
        with open(filepath, "wb") as f:
            while content := await file.read(1024):
                f.write(content)
        
        # Verify WAV file
        try:
            with wave.open(filepath, 'rb') as wav_file:
                params = wav_file.getparams()
                logger.info(f"Received audio: {params}")
        except Exception as e:
            os.remove(filepath)
            raise HTTPException(400, f"Invalid WAV file: {str(e)}")
        
        # Run prediction
        try:
            prediction_result = predict(filepath)
            logger.info(f"Prediction: {prediction_result['predicted_class']} ({prediction_result['confidence']:.2f}%)")
            
            return JSONResponse({
                "status": "success",
                "filename": filename,
                "timestamp": timestamp,
                "prediction": prediction_result
            })
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise HTTPException(500, f"Prediction error: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(500, "Internal server error")

@app.get("/")
async def root():
    return {"status": "Cry Detection API is running"}

# Load model at startup
@app.on_event("startup")
async def startup_event():
    load_model()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)