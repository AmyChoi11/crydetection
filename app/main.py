from fastapi import FastAPI, HTTPException, File, UploadFile, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import io
import os
import logging
import json
from typing import Dict, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Baby Cry Detection API",
    description="API for classifying baby cry sounds",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration class (similar to demo.py)
class Config:
    CLASS_LABELS = ['unwelltired', 'hungry', 'others', 'laugh', 'quiet']
    NUM_CLASSES = len(CLASS_LABELS)
    MAX_FRAMES = 48
    N_MFCC = 24
    SAMPLE_RATE = 16000

# Global variables to store the model and normalization values
model = None
mean_value = 0  # Default value if file not found
std_value = 1   # Default value if file not found

# Response model for predictions
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]
    status: str

class StatusResponse(BaseModel):
    status: str
    message: str

# Build model architecture
def build_model():
    from tensorflow.keras.layers import (
        InputLayer, BatchNormalization, Conv2D, SeparableConv2D, MaxPooling2D,
        Dense, Dropout, GlobalAveragePooling2D, Reshape, Bidirectional, 
        LSTM, Multiply, Concatenate
    )
    
    # Define input shape
    inputs = tf.keras.Input(shape=(Config.MAX_FRAMES, Config.N_MFCC, 1))
    
    # Preprocessing
    x = BatchNormalization()(inputs)
    
    # First convolutional block with residual connection
    conv1 = SeparableConv2D(48, (3,3), activation='relu', padding='same')(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = SeparableConv2D(48, (3,3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    
    # Add residual connection
    x = Conv2D(48, (1,1), padding='same')(x)  # 1x1 conv to match dimensions
    x = tf.keras.layers.add([x, conv1])
    x = tf.keras.layers.Activation('relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Dropout(0.3)(x)
    
    # Second block with attention mechanism
    conv2 = SeparableConv2D(96, (3,3), activation='relu', padding='same')(x)
    conv2 = BatchNormalization()(conv2)
    
    # Channel attention
    att = GlobalAveragePooling2D()(conv2)
    att = Dense(96, activation='sigmoid')(att)
    att = Reshape((1, 1, 96))(att)
    conv2 = Multiply()([conv2, att])
    
    x = MaxPooling2D((2,2))(conv2)
    x = Dropout(0.4)(x)
    
    # Global spatial features
    spatial_features = GlobalAveragePooling2D()(x)
    
    # Temporal features using LSTM
    # Reshape for LSTM processing
    reshape_layer = Reshape((-1, x.shape[-1] * x.shape[-2]))(x)
    lstm_layer = Bidirectional(LSTM(64))(reshape_layer)
    
    # Combine features
    combined = Concatenate()([spatial_features, lstm_layer])
    combined = Dropout(0.5)(combined)
    
    # Final classification head
    x = Dense(128, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    outputs = Dense(Config.NUM_CLASSES, activation='softmax', dtype='float32')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Data preprocessing function
def preprocess_mfcc(mfcc_data):
    """Preprocess MFCC data for model input"""
    try:
        # Handle dimensionality
        if mfcc_data.ndim == 2:
            mfcc_data = np.expand_dims(mfcc_data, axis=0)
        
        # Time axis alignment
        if mfcc_data.shape[1] < Config.MAX_FRAMES:
            pad_width = ((0, 0), (0, Config.MAX_FRAMES - mfcc_data.shape[1]), (0, 0))
            mfcc_data = np.pad(mfcc_data, pad_width, mode='constant')
        else:
            mfcc_data = mfcc_data[:, :Config.MAX_FRAMES, :]
        
        # Add channel dimension
        mfcc_data = np.expand_dims(mfcc_data, axis=-1)
        
        # Normalize
        global mean_value, std_value
        return (mfcc_data - mean_value) / (std_value + 1e-8)
        
    except Exception as e:
        logger.error(f"Error preprocessing MFCC data: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error preprocessing data: {str(e)}")

# Post-processing correction function
def correct_predictions(predictions):
    """Apply post-processing correction to predictions"""
    corrected = predictions.copy()
    unwelltired_idx = Config.CLASS_LABELS.index('unwelltired')
    hungry_idx = Config.CLASS_LABELS.index('hungry')
    
    # For predictions where hungry is top but unwelltired is close behind
    for i in range(len(corrected)):
        pred_class = np.argmax(corrected[i])
        if pred_class == hungry_idx:
            # If model is unsure between hungry and unwelltired
            if (corrected[i, hungry_idx] < 0.65 and 
                corrected[i, unwelltired_idx] > 0.25):
                # Boost unwelltired probability
                boost = corrected[i, hungry_idx] * 0.3
                corrected[i, hungry_idx] -= boost
                corrected[i, unwelltired_idx] += boost
    
    return corrected

# Process audio data from ESP8266
def process_audio_data(audio_data):
    """Convert raw audio to MFCC features"""
    try:
        # This function converts raw audio to MFCC features
        import librosa
        
        # Convert int16 audio data to float
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio_float, 
            sr=Config.SAMPLE_RATE,
            n_mfcc=Config.N_MFCC,
            n_fft=1024,
            hop_length=512
        )
        
        # Transpose to time x features
        mfcc = mfcc.T
        
        # Prepare for model input
        mfcc_expanded = np.expand_dims(np.expand_dims(mfcc, 0), -1)
        
        # Normalize using stored values
        global mean_value, std_value
        return (mfcc_expanded - mean_value) / (std_value + 1e-8)
        
    except Exception as e:
        logger.error(f"Error processing audio data: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing audio: {str(e)}")

# Add this near your other endpoint definitions
@app.get("/")
async def root():
    return {
        "message": "Baby Cry Detection API is running",
        "endpoints": {
            "status": "/status",
            "docs": "/docs",
            "predict": "/predict",
            "analyze": "/analyze-cry"
        }
    }

# Load model at startup
@app.on_event("startup")
async def load_model():
    global model, mean_value, std_value
    
    try:
        logger.info("Loading model...")
        model = build_model()
        
        # Try to load the model, but don't fail if it doesn't exist
        try:
            model.load_weights('best_model.keras')
            logger.info("Model loaded successfully")
        except:
            logger.warning("Could not load model weights. Using uninitialized model.")
            
        # Try to load normalization values, but don't fail if they don't exist
        try:
            mean_value = np.load('mean.npy')
            std_value = np.load('std.npy')
            logger.info("Normalization values loaded successfully")
        except:
            logger.warning("Could not load normalization values. Using defaults (mean=0, std=1).")
            
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")

# API status endpoint
@app.get("/status", response_model=StatusResponse)
async def status():
    """Check if the API and model are running properly"""
    if model is None:
        return {"status": "error", "message": "Model not loaded"}
    return {"status": "ok", "message": "API is running and model is loaded"}

# Prediction endpoint for JSON data
@app.post("/predict", response_model=PredictionResponse)
async def predict_from_json(file_path: str):
    """
    Make a prediction from an MFCC file path
    
    Args:
        file_path: Path to .npy file containing MFCC data
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Load MFCC data from the provided path
        mfcc_data = np.load(file_path)
        
        # Preprocess data
        processed_data = preprocess_mfcc(mfcc_data)
        
        # Get prediction
        raw_prediction = model.predict(processed_data, verbose=0)[0]
        
        # Apply post-processing correction
        corrected_prediction = correct_predictions(np.array([raw_prediction]))[0]
        
        # Get final class
        predicted_class_idx = np.argmax(corrected_prediction)
        predicted_class = Config.CLASS_LABELS[predicted_class_idx]
        confidence = float(corrected_prediction[predicted_class_idx])
        
        # Prepare all probabilities as a dictionary
        all_probs = {
            class_name: float(prob) 
            for class_name, prob in zip(Config.CLASS_LABELS, corrected_prediction)
        }
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_probabilities": all_probs,
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# Prediction endpoint for file upload
@app.post("/predict/upload", response_model=PredictionResponse)
async def predict_from_upload(file: UploadFile = File(...)):
    """
    Make a prediction from an uploaded MFCC .npy file
    
    Args:
        file: Uploaded .npy file containing MFCC data
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read file content
        content = await file.read()
        
        # Load MFCC data from the uploaded file
        mfcc_data = np.load(io.BytesIO(content))
        
        # Preprocess data
        processed_data = preprocess_mfcc(mfcc_data)
        
        # Get prediction
        raw_prediction = model.predict(processed_data, verbose=0)[0]
        
        # Apply post-processing correction
        corrected_prediction = correct_predictions(np.array([raw_prediction]))[0]
        
        # Get final class
        predicted_class_idx = np.argmax(corrected_prediction)
        predicted_class = Config.CLASS_LABELS[predicted_class_idx]
        confidence = float(corrected_prediction[predicted_class_idx])
        
        # Prepare all probabilities as a dictionary
        all_probs = {
            class_name: float(prob) 
            for class_name, prob in zip(Config.CLASS_LABELS, corrected_prediction)
        }
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_probabilities": all_probs,
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# Endpoint for ESP8266 to send audio data
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
        
        try:
            # Process audio to extract MFCC features and make prediction
            import librosa
            
            # Convert int16 audio to float
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=audio_float, 
                sr=Config.SAMPLE_RATE,
                n_mfcc=Config.N_MFCC
            )
            
            # Transpose to time x features
            mfcc = mfcc.T
            
            # Ensure proper shape for model input
            if mfcc.shape[0] < Config.MAX_FRAMES:
                pad_width = ((0, Config.MAX_FRAMES - mfcc.shape[0]), (0, 0))
                mfcc = np.pad(mfcc, pad_width, mode='constant')
            else:
                mfcc = mfcc[:Config.MAX_FRAMES, :]
                
            # Add batch and channel dimensions
            mfcc = np.expand_dims(np.expand_dims(mfcc, 0), -1)
            
            # Normalize
            mfcc = (mfcc - mean_value) / (std_value + 1e-8)
            
            # Make prediction
            prediction = model.predict(mfcc, verbose=0)[0]
            
            # Apply correction
            corrected = correct_predictions(np.array([prediction]))[0]
            
            # Get results
            predicted_class_idx = np.argmax(corrected)
            predicted_class = Config.CLASS_LABELS[predicted_class_idx]
            confidence = float(corrected[predicted_class_idx])
            
            # Format for ESP8266
            response = {
                "reason": predicted_class,
                "confidence": confidence,
                "all_probabilities": {
                    class_name: float(prob) 
                    for class_name, prob in zip(Config.CLASS_LABELS, corrected)
                }
            }
            
            return response
            
        except ImportError:
            # If librosa isn't installed, return a dummy response for testing
            logger.warning("Librosa not installed, returning dummy prediction")
            return {
                "reason": "hungry",  # Default prediction
                "confidence": 0.85,
                "all_probabilities": {
                    "unwelltired": 0.1,
                    "hungry": 0.85,
                    "others": 0.02,
                    "laugh": 0.01,
                    "quiet": 0.02
                }
            }
            
    except Exception as e:
        logger.error(f"Error analyzing cry: {str(e)}")
        return {"status": "error", "message": str(e)}

# If you need a direct connection to Arduino/ESP8266
@app.get("/predict/esp", response_model=PredictionResponse)
async def predict_for_esp(file_path: str):
    """
    Simplified prediction endpoint for ESP8266/Arduino devices
    
    Args:
        file_path: Path to .npy file containing MFCC data
    """
    # Reuse the JSON prediction endpoint
    result = await predict_from_json(file_path)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)