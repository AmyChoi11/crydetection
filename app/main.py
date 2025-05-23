import numpy as np
import librosa
import tensorflow as tf
import logging
import os
import aiohttp
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("baby-cry-detection")

# Initialize the FastAPI application
app = FastAPI(
    title="Baby Cry Detection API",
    description="API for analyzing baby cry audio to determine the likely cause",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
mean_value = None
std_value = None

# Configuration
class Config:
    CLASS_LABELS = ['unwelltired', 'hungry', 'others', 'laugh', 'quiet']
    NUM_CLASSES = len(CLASS_LABELS)
    MAX_FRAMES = 48  # Match your model's expected input
    N_MFCC = 24      # Match your model's expected input
    SAMPLE_RATE = 16000  # Match ESP8266 sampling rate
    HOP_LENGTH = 512
    N_FFT = 1024

# Response models
class PredictionResponse(BaseModel):
    reason: str
    confidence: float
    class_probabilities: Dict[str, float]
    status: str = "success"

class StatusResponse(BaseModel):
    status: str
    message: str

# Load model at startup
@app.on_event("startup")
async def load_model():
    """Load the TensorFlow model when the application starts"""
    global model, mean_value, std_value
    try:
        # Build model architecture
        from tensorflow.keras.layers import (
            InputLayer, BatchNormalization, Conv2D, MaxPooling2D,
            Dense, Dropout, GlobalAveragePooling2D
        )
        
        model = tf.keras.Sequential([
            InputLayer(shape=(Config.MAX_FRAMES, Config.N_MFCC, 1)),
            BatchNormalization(),
            
            Conv2D(64, (3,3), activation='relu', padding='same'),
            MaxPooling2D((2,2)),
            Dropout(0.3),
            
            Conv2D(128, (3,3), activation='relu', padding='same'),
            GlobalAveragePooling2D(),
            Dropout(0.4),
            
            Dense(Config.NUM_CLASSES, activation='softmax')
        ])
        
        # Try to load saved model weights
        try:
            model.load_weights('best_model.keras')
            logger.info("Model weights loaded successfully")
        except:
            logger.warning("Could not load model weights. Using uninitialized model.")
        
        # Try to load normalization values
        try:
            mean_value = np.load('mean.npy')
            std_value = np.load('std.npy')
            logger.info("Normalization values loaded successfully")
        except:
            logger.warning("Could not load normalization values. Using defaults (0,1).")
            mean_value = 0
            std_value = 1
        
        logger.info("Model initialization complete")
    except Exception as e:
        logger.error(f"Error during model initialization: {str(e)}")
        model = None

# Process raw audio data
def process_raw_audio(audio_data, sample_rate=Config.SAMPLE_RATE):
    """Process raw audio data into MFCC features suitable for the model"""
    try:
        # Convert int16 audio to float
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio_float, 
            sr=sample_rate,
            n_mfcc=Config.N_MFCC,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH
        )
        
        # Transpose to time x features
        mfcc = mfcc.T
        
        # Handle time dimension (frames)
        if mfcc.shape[0] < Config.MAX_FRAMES:
            # Pad if too short
            pad_width = ((0, Config.MAX_FRAMES - mfcc.shape[0]), (0, 0))
            mfcc = np.pad(mfcc, pad_width, mode='constant')
        else:
            # Trim if too long
            mfcc = mfcc[:Config.MAX_FRAMES, :]
        
        # Add batch and channel dimensions
        mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension
        mfcc = np.expand_dims(mfcc, axis=-1)  # Add channel dimension
        
        # Normalize using stored values
        global mean_value, std_value
        mfcc_normalized = (mfcc - mean_value) / (std_value + 1e-8)
        
        return mfcc_normalized
        
    except Exception as e:
        logger.error(f"Error processing audio data: {str(e)}")
        raise Exception(f"Audio processing error: {str(e)}")

# Notify external devices
async def notify_devices(cry_reason):
    """Notify Watchy watch and mobile app about cry detection"""
    try:
        # Get Watchy IP from configuration
        watch_ip = os.getenv("WATCHY_IP", "192.168.1.X")  # Replace X with actual value
        
        # Only try to notify if the IP is set
        if watch_ip and "x" not in watch_ip.lower():
            logger.info(f"Attempting to notify Watchy at {watch_ip}")
            # Send to Watchy (vibrate endpoint)
            async with aiohttp.ClientSession() as session:
                watch_url = f"http://{watch_ip}/vibrate"
                try:
                    async with session.post(watch_url) as response:
                        logger.info(f"Notified watch: {response.status}")
                except Exception as e:
                    logger.error(f"Failed to notify watch: {str(e)}")
                    
    except Exception as e:
        logger.error(f"Failed to notify devices: {str(e)}")

# API Routes
@app.get("/", response_model=StatusResponse)
async def root():
    """Root endpoint to check if the API is running"""
    return {
        "status": "online", 
        "message": "Baby Cry Detection API is running"
    }

@app.get("/status", response_model=StatusResponse)
async def status():
    """Check if the API and model are running properly"""
    if model is None:
        return {"status": "error", "message": "Model not loaded"}
    return {"status": "ok", "message": "API is running and model is loaded"}

@app.post("/analyze-cry")
async def analyze_cry(request: Request):
    """
    Analyze raw audio data from ESP8266
    
    The ESP8266 sends raw audio data as binary
    """
    if model is None:
        return {"status": "error", "message": "Model not loaded"}
        
    try:
        # Read the binary audio data
        audio_bytes = await request.body()
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Handle empty data
        if len(audio_data) == 0:
            return {"status": "error", "message": "Empty audio data received"}
            
        logger.info(f"Received audio data of length: {len(audio_data)}")
        
        # Process the audio data
        mfcc_features = process_raw_audio(audio_data)
        
        # Make prediction
        prediction = model.predict(mfcc_features, verbose=0)[0]
        
        # Get predicted class and confidence
        predicted_idx = np.argmax(prediction)
        confidence = float(prediction[predicted_idx])
        predicted_class = Config.CLASS_LABELS[predicted_idx]
        
        logger.info(f"Predicted class: {predicted_class} with confidence: {confidence:.4f}")
        
        # Create class probabilities dictionary
        class_probs = {cls: float(prob) for cls, prob in zip(Config.CLASS_LABELS, prediction)}
        
        # Try to notify devices (don't wait for completion)
        import asyncio
        asyncio.create_task(notify_devices(predicted_class))
        
        # Return result to ESP8266
        return {
            "reason": predicted_class,
            "confidence": confidence,
            "class_probabilities": class_probs,
            "status": "success"
        }
            
    except Exception as e:
        logger.error(f"Error analyzing cry: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint to verify API functionality"""
    return {"status": "success", "message": "Test endpoint is working"}

# For running directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)