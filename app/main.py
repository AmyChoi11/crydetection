import numpy as np
import librosa
import tensorflow as tf
import logging
import os
import aiohttp
import asyncio
import sqlite3
import json
import scipy.stats  # Import needed for audio processing
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
import scipy.stats  # Needed for entropy calculation
import socket
from fastapi import WebSocket
from fastapi.websockets import WebSocketDisconnect
import wave
import asyncio
import threading
import queue
import time
import sounddevice as sd

# Configure environment variables
os.environ["WATCHY_IP"] = "192.168.4.1"  # Replace with actual Watchy IP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("baby-cry-detection")

# Global variables
model = None
mean_value = None
std_value = None
config = None

# Configuration
class Config:
    # Original labels used for model training - DO NOT CHANGE THESE
    ORIGINAL_LABELS = ['Unwell', 'Sleeping', 'Cry', 'Laugh', 'Tired', 'Silence']
    
    # New presentation labels for API responses
    CLASS_LABELS = ['uncomfortable', 'sleeping', 'crying', 'laughing', 'tired', 'silent']
    
    # Mapping from original index to new label (this preserves the model's original output)
    LABEL_MAPPING = {
        0: 0,  # 'Unwell' → 'uncomfortable'
        1: 1,  # 'Sleeping' → 'sleeping'
        2: 2,  # 'Cry' → 'crying'
        3: 3,  # 'Laugh' → 'laughing'
        4: 4,  # 'Tired' → 'tired'
        5: 5,  # 'Silence' → 'silent'
    }
    
    NUM_CLASSES = len(ORIGINAL_LABELS)
    MAX_FRAMES = 50  # Match training value
    N_MFCC = 13  # Match training value
    SAMPLE_RATE = 22050  # Match training value
    HOP_LENGTH = 256  # Match training value
    N_FFT = 512  # Match training value
    UPLOAD_DIR = "audio_clips"
    DB_PATH = "db/cry_history.db"
    
    # System configuration parameters
    DETECTION_THRESHOLD = 500  # Amplitude threshold for cry detection
    NOTIFICATION_COOLDOWN = 60  # Seconds between notifications
    CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence to trigger notification

class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass
prediction_counters = {label: 0 for label in Config.CLASS_LABELS}
current_prediction = "silent"
MIN_DETECTION_VOLUME = 2000  # Minimum volume to consider non-ambient noise
COUNTER_THRESHOLD = 3  # Number of consecutive detections needed
# Create an instance of the connection manager
manager = ConnectionManager()

# Audio processing queue and thread state
audio_queue = queue.Queue()
stop_audio_thread = threading.Event()

# Audio capture configuration
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 3  # 3-second clips as requested

async def load_model():
    """Load the TensorFlow model and initialize system when the application starts"""
    global model, mean_value, std_value, config
    
    # Initialize database
    initialize_database()
    
    # Initialize configuration
    config = AppConfiguration()
    
    try:
        # Load the trained model
        model_path = "newest_model.keras"
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
        else:
            # Fallback to placeholder if model file not found
            logger.warning(f"Model file {model_path} not found, using placeholder")
            inputs = tf.keras.layers.Input(shape=(169,))
            x = tf.keras.layers.Dense(128, activation='relu')(inputs)
            x = tf.keras.layers.Dropout(0.3)(x)
            x = tf.keras.layers.Dense(64, activation='relu')(x)
            outputs = tf.keras.layers.Dense(Config.NUM_CLASSES, activation='softmax')(x)
            model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', loss='categorical_crossentropy')
            logger.warning("Using placeholder model (weights not loaded)")
        
        # Load normalization values
        try:
            mean_value = np.load('mean.npy')
            std_value = np.load('std.npy')
            logger.info(f"Normalization values loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load normalization values: {str(e)}. Using defaults (0,1).")
            mean_value = 0
            std_value = 1
        
        logger.info(f"Model initialization complete")
    except Exception as e:
        logger.error(f"Error during model initialization: {str(e)}")
        model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model
    await load_model()
    
    # List audio devices - will help identify if there's a device issue
    list_audio_devices()
    
    # Check audio levels before starting the monitoring thread
    check_audio_levels()
    
    # Start audio thread
    audio_thread = threading.Thread(target=process_audio_task)
    audio_thread.daemon = True
    audio_thread.start()
    asyncio.create_task(process_audio_queue())
    
    yield
    
    # Cleanup
    stop_audio_thread.set()

# Initialize the FastAPI application
app = FastAPI(
    title="Baby Cry Detection API",
    description="API for analyzing baby cry audio to determine the likely cause",
    version="1.0.0",
    lifespan=lifespan
)

# Create directories if they don't exist
os.makedirs("audio_clips", exist_ok=True)
os.makedirs("db", exist_ok=True)

# Response models
class PredictionResponse(BaseModel):
    reason: str
    confidence: float
    class_probabilities: Dict[str, float]
    status: str = "success"

class StatusResponse(BaseModel):
    status: str
    message: str

class ConfigModel(BaseModel):
    threshold: int = Config.DETECTION_THRESHOLD
    notification_cooldown: int = Config.NOTIFICATION_COOLDOWN
    confidence_threshold: float = Config.CONFIDENCE_THRESHOLD
    watchy_ip: str = os.environ["WATCHY_IP"]
    notify_parent: bool = True
    notify_watchy: bool = True
    camera_ip: str = "172.20.10.2"

def get_local_ip():
    """Get the local IP address of the machine"""
    try:
        # Create a socket connection to an external server
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        logger.warning(f"Could not determine IP address: {str(e)}")
        return "localhost"
    
def list_audio_devices():
    """List available audio input devices"""
    devices = sd.query_devices()
    input_devices = []
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append((i, device['name']))
    return input_devices

def map_prediction_to_new_labels(prediction):
    """Convert model prediction using original labels to new presentation labels"""
    # Get original prediction index and confidence
    original_idx = np.argmax(prediction)
    confidence = float(prediction[original_idx])
    
    # Map to the new label
    new_idx = Config.LABEL_MAPPING[original_idx]
    new_label = Config.CLASS_LABELS[new_idx]
    
    # Create presentation probabilities dictionary
    probs = {}
    for orig_idx, prob in enumerate(prediction):
        new_idx = Config.LABEL_MAPPING[orig_idx]
        new_label = Config.CLASS_LABELS[new_idx]
        probs[new_label] = float(prob)
    
    return new_label, confidence, probs

# Database functions
def initialize_database():
    """Create database for storing cry history"""
    try:
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS cry_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            reason TEXT,
            confidence REAL,
            audio_file TEXT
        )
        ''')
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")

def store_cry_event(reason, confidence, audio_path=None):
    """Store cry event in database for pattern analysis"""
    try:
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO cry_events (timestamp, reason, confidence, audio_file) VALUES (?, ?, ?, ?)",
            (timestamp, reason, confidence, audio_path)
        )
        conn.commit()
        conn.close()
        logger.info(f"Cry event stored: {reason} at {timestamp}")
        return True
    except Exception as e:
        logger.error(f"Failed to store cry event: {str(e)}")
        return False

def get_cry_history(days=7, limit=100):
    """Retrieve cry history for the specified number of days"""
    try:
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        query = """
        SELECT timestamp, reason, confidence, audio_file 
        FROM cry_events 
        WHERE datetime(timestamp) >= datetime('now', ?) 
        ORDER BY timestamp DESC LIMIT ?
        """
        cursor.execute(query, (f"-{days} days", limit))
        results = cursor.fetchall()
        conn.close()
        
        # Format results as list of dictionaries
        history = []
        for row in results:
            history.append({
                "timestamp": row[0],
                "reason": row[1],
                "confidence": row[2],
                "audio_file": row[3]
            })
        return history
    except Exception as e:
        logger.error(f"Failed to get cry history: {str(e)}")
        return []

# Configuration management
class AppConfiguration:
    def __init__(self, config_file='config.json'):
        self.config_file = config_file
        self.load_config()
    
    def load_config(self):
        """Load configuration from file or use defaults"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
                logger.info("Configuration loaded from file")
            else:
                # Initialize with default values
                self.config = {
                    "threshold": Config.DETECTION_THRESHOLD,
                    "notification_cooldown": Config.NOTIFICATION_COOLDOWN,
                    "confidence_threshold": Config.CONFIDENCE_THRESHOLD,
                    "watchy_ip": os.environ["WATCHY_IP"],
                    "notify_parent": True,
                    "notify_watchy": True
                }
                self.save_config()
                logger.info("Default configuration created")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            # Fallback to defaults
            self.config = {
                "threshold": Config.DETECTION_THRESHOLD,
                "notification_cooldown": Config.NOTIFICATION_COOLDOWN,
                "confidence_threshold": Config.CONFIDENCE_THRESHOLD,
                "watchy_ip": os.environ["WATCHY_IP"],
                "notify_parent": True,
                "notify_watchy": True
            }
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info("Configuration saved to file")
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
    
    def get(self, key, default=None):
        """Get configuration value with fallback"""
        return self.config.get(key, default)
    
    def update(self, new_config):
        """Update configuration with new values"""
        self.config.update(new_config)
        self.save_config()
        return self.config

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Replace the existing process_raw_audio function with this improved version

def process_raw_audio(audio_data, sample_rate=Config.SAMPLE_RATE):
    """Process raw audio data into features suitable for the model"""
    try:
        # Convert int16 audio to float
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        # Resample if needed
        if sample_rate != Config.SAMPLE_RATE:
            audio_float = librosa.resample(
                y=audio_float, 
                orig_sr=sample_rate, 
                target_sr=Config.SAMPLE_RATE
            )
        
        # Extract MFCC features - using parameters that match the model training
        mfcc = librosa.feature.mfcc(
            y=audio_float, 
            sr=Config.SAMPLE_RATE,
            n_mfcc=10,  # We need 10 MFCCs based on the expected shape
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH
        )
        
        logger.info(f"MFCC shape: {mfcc.shape}")
        
        # Ensure we have exactly 50 frames (time steps)
        if mfcc.shape[1] > 50:
            # Truncate to 50 frames
            mfcc = mfcc[:, :50]
        elif mfcc.shape[1] < 50:
            # Pad with zeros to 50 frames
            padding_width = 50 - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, padding_width)), mode='constant')
        
        # Reshape to (50, 10) and add channel dimension for CNN
        mfcc = mfcc.T  # Transpose to (time_steps, features)
        
        # Normalize features using loaded mean and std
        normalized_features = (mfcc - mean_value) / (std_value + 1e-8)
        
        # Add batch and channel dimensions: (1, 50, 10, 1)
        normalized_features = normalized_features.reshape(1, 50, 10, 1)
        
        logger.info(f"Final feature shape: {normalized_features.shape}")
        if normalized_features is not None:
            # Log shape before returning
            logger.info(f"Final feature shape: {normalized_features.shape}")
            return normalized_features
        return normalized_features
        
    except Exception as e:
        logger.error(f"Error processing audio data: {str(e)}")
        raise Exception(f"Audio processing error: {str(e)}")

# Notify external devices
async def notify_devices(cry_reason, confidence):
    """Notify Watchy watch and mobile app about cry detection"""
    try:
        # Check if notifications should be sent based on configuration
        if not config.get("notify_watchy") and not config.get("notify_parent"):
            logger.info("Notifications disabled in configuration")
            return
        
        # Only notify if confidence exceeds threshold
        if confidence < config.get("confidence_threshold", 0.7):
            logger.info(f"Confidence {confidence:.2f} below threshold - not sending notification")
            return
            
        # Get Watchy IP from configuration or try to discover it
        watch_ip = config.get("watchy_ip")
        
        # Notify Watchy if enabled
        if config.get("notify_watchy") and watch_ip:
            logger.info(f"Attempting to notify Watchy at {watch_ip}")
            
            # Try multiple ways to reach the Watchy
            success = False
            
            # Try direct IP first
            if not success:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            f"http://{watch_ip}/vibrate", 
                            json={"reason": cry_reason}, 
                            timeout=2
                        ) as response:
                            if response.status == 200:
                                success = True
                                logger.info("Notified Watchy via direct IP")
                except:
                    pass
            
            # Try mDNS if direct IP failed
            if not success:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            "http://watchy.local/vibrate", 
                            json={"reason": cry_reason}, 
                            timeout=2
                        ) as response:
                            if response.status == 200:
                                success = True
                                logger.info("Notified Watchy via mDNS")
                except:
                    pass
            
            if not success:
                logger.warning("Failed to notify Watchy - could not reach device")
                
    except Exception as e:
        logger.error(f"Failed to notify devices: {str(e)}")
        
        # Notify mobile app if enabled
        if config.get("notify_parent"):
            logger.info("Would notify parent mobile app here")
            # In a real implementation, you would:
            # 1. Send push notification via Firebase Cloud Messaging
            # 2. Update mobile app via websocket connection
            # 3. Trigger other integrations as needed
def check_audio_levels():
    """Check microphone audio levels and print diagnostics"""

    
    duration = 5  # seconds
    sample_rate = RATE
    
    logger.info("Recording 5 seconds to check audio levels...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=CHANNELS, dtype='int16')
    sd.wait()  # Wait until recording is finished
    
    # Calculate diagnostics
    abs_data = np.abs(recording)
    logger.info(f"Audio diagnostics:")
    logger.info(f"  Shape: {recording.shape}")
    logger.info(f"  Min: {np.min(recording)}")
    logger.info(f"  Max: {np.max(recording)}")
    logger.info(f"  Mean: {np.mean(abs_data)}")
    logger.info(f"  RMS: {np.sqrt(np.mean(recording**2))}")
    logger.info(f"  Non-zero samples: {np.count_nonzero(recording)}/{recording.size}")
    
    # Check if the audio is too quiet
    if np.max(abs_data) < 100:
        logger.warning("Audio levels extremely low - microphone may not be working!")
    elif np.max(abs_data) < 500:
        logger.warning("Audio levels very low - check microphone sensitivity")
    else:
        logger.info("Audio levels appear normal")
    
    return recording

# API Routes
@app.get("/routes")
async def list_routes():
    return [
        {"path": route.path, "name": route.name, "methods": route.methods}
        for route in app.routes
    ]
# Add this route handler after your other @app.get routes

@app.get("/ui", response_class=HTMLResponse)
async def ui():
    """Serve the same UI at /ui path for compatibility"""
    return await root()  # Reuse the same HTML from the root handler
# Update the root() function with this modified HTML

@app.get("/test-microphone")
async def test_microphone():
    """Test if microphone is working"""
    try:
        devices = sd.query_devices()
        default_device = sd.default.device[0]
        device_name = devices[default_device]['name']
        return {
            "status": "success",
            "message": f"Microphone detected: {device_name}",
            "device_id": default_device
        }
    except Exception as e:
        return {"status": "error", "message": f"Microphone error: {str(e)}"}
    
# Add this WebSocket endpoint after your other endpoints
@app.get("/test-microphone-levels")
async def test_microphone_levels():
    """Test endpoint to check microphone audio levels"""
    recording = check_audio_levels()
    
    # Return diagnostics as JSON
    return {
        "status": "success",
        "diagnostics": {
            "shape": recording.shape,
            "min": float(np.min(recording)),
            "max": float(np.max(recording)),
            "mean": float(np.mean(np.abs(recording))),
            "rms": float(np.sqrt(np.mean(recording**2))),
            "non_zero": int(np.count_nonzero(recording))
        }
    }
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Just keep the connection alive - data is sent via broadcast
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.on_event("startup")
async def startup_event():
    # Load configuration and other startup tasks
    # ...
    
    # List audio devices
    list_audio_devices()

def list_audio_devices():
    """List available audio input devices"""
    
    devices = sd.query_devices()
    logger.info("Available audio devices:")
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            logger.info(f"  {i}: {device['name']} (Inputs: {device['max_input_channels']})")
    
    # Log the default device
    default_device = sd.default.device[0]
    logger.info(f"Default input device: {default_device}: {devices[default_device]['name']}")
    
    return devices

# Background task to process audio
def process_audio_task():
    
    try:
        logger.info("Microphone monitoring started")
        
        # Run an initial test to check levels
        check_audio_levels()
        
        while not stop_audio_thread.is_set():
            # Record audio for RECORD_SECONDS
            audio_data = sd.rec(int(RECORD_SECONDS * RATE), 
                              samplerate=RATE, 
                              channels=CHANNELS, 
                              dtype='int16')
            sd.wait()  # Wait until recording is finished
            
            # Add volume diagnostic
            max_vol = np.max(np.abs(audio_data))
            logger.info(f"Audio recorded - max volume: {max_vol}")
            
            # Skip processing if audio is too quiet (likely silence)
            if max_vol < 50:  # Adjust this threshold as needed
                logger.warning("Audio too quiet, skipping processing")
                time.sleep(0.5)  # Short sleep before next recording
                continue
                
            # Flatten the array for processing if needed
            if audio_data.ndim > 1:
                audio_data = audio_data.flatten()
            
            # Push to processing queue
            audio_queue.put(audio_data)
            
            # Small break between recordings
            time.sleep(0.1)
    
    except Exception as e:
        logger.error(f"Error in audio recording: {str(e)}")
    finally:
        logger.info("Microphone monitoring stopped")

async def process_audio_queue():
    # Define class multipliers first
    class_multipliers = {
        0: 1.0,  # Boost "uncomfortable" 
        1: 1.5,  # Boost "sleeping"
        2: 1.0,  # Boost "crying" slightly
        3: 2.1,  # Major boost for "laughing" which seems under-predicted
        4: 0.8,  # Slightly reduce "tired" which seems over-predicted
        5: 1.0   # Keep "silent" as-is
    }
    

    # Create threshold values for each class
    class_thresholds = {
        "uncomfortable": 0.40,  # Unwell
        "sleeping": 0.45,
        "crying": 0.45,
        "laughing": 0.29,
        "tired": 0.65,
        "silent": 0.70
    }
    
    # Add these for temporal smoothing
    global prediction_ema
    if 'prediction_ema' not in globals():
        prediction_ema = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.7])  # Initially biased to silence
    
    # Add counter for periodic updates even when state doesn't change
    update_counter = 0

    while True:
        try:
            # Process items if available, otherwise sleep
            if not audio_queue.empty():
                audio_data = audio_queue.get()
                update_counter += 1
                
                # Get max volume for audio level detection
                max_volume = np.max(np.abs(audio_data))
                logger.info(f"Audio recorded - max volume: {max_volume}")
                
                # Process audio if volume is high enough (to avoid processing silence)
                if max_volume >= MIN_DETECTION_VOLUME:
                    logger.info(f"Processing audio with volume: {max_volume}")
                    
                    # Process audio with your existing function
                    features = process_raw_audio(audio_data, sample_rate=RATE)
                    
                    # Make prediction
                    full_prediction = model.predict(features, verbose=0)[0]
                    
                    # TRUNCATE to 6 classes - ignore the 7th class
                    prediction = full_prediction[:6]
                    
                    # Re-normalize to ensure probabilities sum to 1
                    prediction = prediction / np.sum(prediction)
                    
                    # Debug the prediction output
                    logger.info(f"Raw prediction shape: {full_prediction.shape}, truncated to 6 classes")
                    logger.info(f"Class probabilities: {[round(float(p), 2) for p in prediction]}")
                    
                    # Apply temporal smoothing with exponential moving average
                    alpha = 0.3
                    prediction_ema = alpha * prediction + (1 - alpha) * prediction_ema[:6]
                    prediction_ema = prediction_ema / np.sum(prediction_ema)  # Renormalize
                    
                    # Apply class multipliers to adjust for model bias
                    adjusted_prediction = np.array([prediction_ema[i] * class_multipliers[i] for i in range(len(prediction_ema))])
                    # Re-normalize to ensure probabilities still sum to 1
                    adjusted_prediction = adjusted_prediction / np.sum(adjusted_prediction)
                    # Add this right after your adjusted_prediction calculation
                    if prediction_ema[3] > 0.1:  # If laughing probability is significant
                        logger.info(f"Laughing probability before adjustment: {prediction[3]:.4f}")
                        logger.info(f"Laughing probability after EMA: {prediction_ema[3]:.4f}")
                        logger.info(f"Laughing probability after multiplier: {adjusted_prediction[3]:.4f}")
                        logger.info(f"Laughing threshold: {dynamic_thresholds['laughing']:.4f}")
                    # Log both original and adjusted predictions
                    logger.info(f"Original probabilities: {[round(float(p), 2) for p in prediction_ema]}")
                    logger.info(f"Adjusted probabilities: {[round(float(p), 2) for p in adjusted_prediction]}")
                    
                    # Use adjusted prediction for the rest of your logic
                    prediction_ema = adjusted_prediction
                    
                    # ADVANCED DECISION LOGIC
                    volume_factor = min(1.0, max_volume / 30000)  # Normalize to 0-1
                    
                    # Audio volume is significant - use smoothed predictions
                    # Add spectral feature to help identify crying
                    spectral_variety = np.std(features.reshape(-1)) 
                    
                    # Reset all counters first to avoid confusion
                    highest_prob_idx = np.argmax(prediction_ema)
                    highest_prob_class = Config.CLASS_LABELS[highest_prob_idx]
                    highest_prob_value = float(prediction_ema[highest_prob_idx])
                    
                    logger.info(f"Highest probability class: {highest_prob_class} ({highest_prob_value:.2f})")
                    
                    # Modify thresholds based on volume
                    dynamic_thresholds = {}
                    for class_name, base_threshold in class_thresholds.items():
                        # Lower volume = higher threshold to avoid false positives
                        dynamic_thresholds[class_name] = base_threshold + (1 - volume_factor) * 0.05
                    
                    # Special adjustment for crying (uses spectral variety)
                    crying_modifier = 1.0 if spectral_variety > 0.4 else 1.3
                    dynamic_thresholds["crying"] *= crying_modifier
                    # Add this right after your crying_modifier line
                    laughing_spectral_signature = np.std(features.reshape(-1)) * np.max(np.abs(features.reshape(-1)))
                    laughing_modifier = 1.0 if laughing_spectral_signature > 0.6 else 1.5
                    dynamic_thresholds["laughing"] *= laughing_modifier
                    # Check each class against its threshold
                    detected_class = None
                    for idx, class_name in enumerate(Config.CLASS_LABELS):
                        if idx != 5:  # Skip silent class in this check
                            if prediction_ema[idx] > dynamic_thresholds[class_name]:
                                detected_class = class_name
                                prediction_counters[class_name] += 1
                                # Reset other counters
                                for other_class in Config.CLASS_LABELS:
                                    if other_class != class_name and other_class != "silent":
                                        prediction_counters[other_class] = 0
                                break
                    
                    # If no class detected above threshold, decay all counters
                    if not detected_class:
                        for class_name in Config.CLASS_LABELS:
                            if class_name != "silent":
                                prediction_counters[class_name] = max(0, prediction_counters[class_name] - 1)
                    
                    # Log current counter state for debugging
                    logger.info(f"Current counters: {prediction_counters}")
                    
                    # Determine final prediction based on counters
                    predicted_class = "silent"  # Default
                    confidence = float(prediction_ema[5])
                    
                    # Check all non-silent classes
                    for class_name in Config.CLASS_LABELS:
                        if class_name != "silent":
                            class_idx = Config.CLASS_LABELS.index(class_name)
                            if prediction_counters[class_name] >= COUNTER_THRESHOLD:
                                predicted_class = class_name
                                confidence = float(prediction_ema[class_idx])
                                # Add this in the section where you determine the final prediction
                                if predicted_class == "laughing":
                                    # Double-check laughing prediction - it should have significant volume
                                    if max_volume < 5000:
                                        # If volume is too low, it's probably not laughter
                                        prediction_counters["laughing"] = 0
                                        predicted_class = "silent"
                                        confidence = float(prediction_ema[5])
                                break
                    
                    # Generate timestamp
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    
                    # MODIFIED LOGIC: Always send updates in these cases:
                    # 1. When prediction changes
                    # 2. Every 5th frame regardless of change (for live feedback)
                    # 3. When we detect high volume (> 15000)
                    global current_prediction
                    should_update = (
                        predicted_class != current_prediction or  # State changed
                        update_counter % 5 == 0 or              # Periodic update
                        max_volume > 20000                      # High volume detected
                    )
                    
                    if should_update:
                        # Save to history only when state changes to avoid filling the DB
                        if predicted_class != current_prediction:
                            store_cry_event(predicted_class, confidence, "live_audio")
                            current_prediction = predicted_class
                            logger.info(f"Live prediction changed to: {predicted_class} ({confidence:.2f})")
                        
                        # Always broadcast for UI updates
                        await manager.broadcast({
                            "status": "success",
                            "reason": predicted_class,
                            "confidence": float(confidence),
                            "timestamp": timestamp,
                            "volume": int(max_volume),  # Add volume info for UI
                            "counters": prediction_counters  # Send all counters for debug info
                        })
                else:
                    # Low volume = ambient noise, default to silent
                    predicted_class = "silent"
                    confidence = 1.0
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    
                    # Only send silent updates occasionally to reduce traffic
                    if update_counter % 10 == 0:
                        await manager.broadcast({
                            "status": "success",
                            "reason": "silent",
                            "confidence": 1.0,
                            "timestamp": timestamp,
                            "volume": int(max_volume),
                            "counters": prediction_counters
                        })
            
        except Exception as e:
            logger.error(f"Error processing live audio: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            try:
                await manager.broadcast({
                    "status": "error",
                    "message": str(e)
                })
            except:
                pass
        
        # Sleep to prevent CPU overuse
        await asyncio.sleep(0.1)

# Update your root() function to include WebSocket support

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root route that displays a simple UI with embedded video stream"""
    # Get camera IP from config or use default
    camera_ip = config.get("camera_ip", "172.20.10.2")
    
    return f"""
    <html>
    <head>
        <title>Baby Cry Detection</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f8f9fa;
            }}
            h1, h2, h3, h4 {{
                color: #3b5998;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            }}
            .header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 1px solid #eee;
            }}
            .status-indicator {{
                display: inline-block;
                padding: 5px 10px;
                border-radius: 4px;
                background-color: #d1e7dd;
                color: #0f5132;
                font-weight: bold;
            }}
            .grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }}
            .camera-container {{
                border: 1px solid #ddd;
                border-radius: 8px;
                overflow: hidden;
                position: relative;
            }}
            .camera-stream {{
                width: 100%;
                height: 400px;
                background-color: #eee;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #666;
            }}
            .baby-status {{
                padding: 10px;
                margin-top: 10px;
                background-color: #e8f4f8;
                border-radius: 4px;
                text-align: center;
                font-size: 1.2em;
            }}
            .config-section {{
                margin-top: 20px;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 4px;
            }}
            .prediction-list {{
                max-height: 300px;
                overflow-y: auto;
                margin-top: 20px;
                padding: 10px;
                background-color: #f8f9fa;
                border-radius: 4px;
            }}
            .prediction-item {{
                display: flex;
                justify-content: space-between;
                padding: 10px;
                margin-bottom: 5px;
                background-color: #f1f1f1;
                border-radius: 4px;
                transition: background-color 0.3s;
            }}
            .prediction-reason {{
                font-weight: bold;
                color: #0d6efd;
            }}
            .prediction-confidence {{
                color: #198754;
            }}
            .prediction-time {{
                color: #6c757d;
            }}
            .form-section {{
                margin-top: 20px;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 4px;
            }}
            .form-group {{
                margin-bottom: 15px;
            }}
            .form-label {{
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }}
            .form-control {{
                width: 100%;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
            .btn {{
                padding: 10px 15px;
                background-color: #0d6efd;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }}
            .btn:hover {{
                background-color: #0b5ed7;
            }}
            .result-section {{
                margin-top: 15px;
                padding: 15px;
                border-radius: 4px;
            }}
            .success {{
                color: #198754;
            }}
            .error {{
                color: #dc3545;
            }}
            .live-monitor {{
                margin-top: 20px;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 4px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Baby Cry Detection System</h1>
                <div class="status-indicator" id="micStatus">Connecting microphone...</div>
            </div>
            
            <div class="grid">
                <div>
                    <h2>Live Monitor</h2>
                    <div class="camera-container">
                        <iframe id="cameraStream" class="camera-stream" src="http://{camera_ip}/" frameborder="0">
                            Camera stream not available
                        </iframe>
                        <div class="baby-status">Connecting to baby monitor...</div>
                    </div>
                    
                    <div class="config-section">
                        <h3>Camera Settings</h3>
                        <div class="form-group">
                            <label class="form-label">Camera IP Address</label>
                            <div style="display: flex;">
                                <input type="text" id="cameraIp" class="form-control" value="{camera_ip}" style="flex: 1; margin-right: 10px;">
                                <button class="btn" onclick="updateCameraIp()">Update</button>
                            </div>
                        </div>
                    </div>
                    
                    <div class="live-monitor">
                        <h3>Live Detection</h3>
                        <!-- Audio visualization will be added here by JS -->
                    </div>
                </div>
                
                <div>
                    <h2>Prediction History</h2>
                    <div id="livePredictions" class="prediction-list">
                        <!-- Predictions will be added here by JS -->
                        <div class="prediction-item">
                            <span class="prediction-reason">Waiting for data...</span>
                            <span class="prediction-confidence">-</span>
                            <span class="prediction-time">-</span>
                        </div>
                    </div>
                    
                    <div class="form-section">
                        <h3>Upload Audio</h3>
                        <form id="upload-form">
                            <div class="form-group">
                                <label class="form-label">Audio File (WAV, MP3, M4A)</label>
                                <input type="file" name="file" class="form-control" accept=".wav,.mp3,.m4a" required>
                            </div>
                            <button type="submit" class="btn">Analyze</button>
                        </form>
                        <div id="result" class="result-section"></div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // WebSocket connection
            let ws;
            let reconnectAttempts = 0;
            const maxReconnectAttempts = 5;

            // Audio level visualization values
            let audioLevels = [];
            const maxAudioPoints = 50;

            function connectWebSocket() {{
                const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = wsProtocol + '//' + window.location.host + '/ws';
                
                ws = new WebSocket(wsUrl);
                
                ws.onopen = function(event) {{
                    console.log('WebSocket connected');
                    document.getElementById('micStatus').innerHTML = 'Microphone connected and listening';
                    document.getElementById('micStatus').style.color = '#27ae60';
                    reconnectAttempts = 0;
                }};
                
                ws.onmessage = function(event) {{
                    const data = JSON.parse(event.data);
                    if (data.status === 'success') {{
                        // Add prediction to the list
                        addPrediction(data);
                        
                        // Update baby status - FIXED with double braces
                        document.querySelector('.baby-status').innerHTML = 
                            `Your baby is <strong>${{data.reason}}</strong>!`;
                            
                        // Update volume display
                        updateVolumeDisplay(data);
                        
                        // Update counters display
                        if (data.counters) {{
                            document.getElementById('uncomfortableCounter').textContent = data.counters.uncomfortable || 0;
                            document.getElementById('sleepingCounter').textContent = data.counters.sleeping || 0;
                            document.getElementById('cryingCounter').textContent = data.counters.crying || 0;
                            document.getElementById('laughingCounter').textContent = data.counters.laughing || 0;
                            document.getElementById('tiredCounter').textContent = data.counters.tired || 0;
                        }}
                    }}
                }};
                
                ws.onclose = function(event) {{
                    document.getElementById('micStatus').innerHTML = 'Microphone disconnected';
                    document.getElementById('micStatus').style.color = '#e74c3c';
                    
                    // Try to reconnect with exponential backoff
                    if (reconnectAttempts < maxReconnectAttempts) {{
                        const timeout = Math.min(1000 * Math.pow(2, reconnectAttempts), 10000);
                        console.log(`Reconnecting in ${{timeout/1000}} seconds...`);
                        
                        reconnectAttempts++;
                        setTimeout(connectWebSocket, timeout);
                    }}
                }};
                
                ws.onerror = function(error) {{
                    console.error('WebSocket error:', error);
                }};
            }}

            function updateVolumeDisplay(data) {{
                // Update volume bar and value
                if (data.volume !== undefined) {{
                    // Update numeric value
                    document.getElementById('volumeValue').textContent = data.volume;
                    
                    // Add volume to time series
                    audioLevels.push(data.volume);
                    if (audioLevels.length > maxAudioPoints) {{
                        audioLevels.shift();
                    }}
                    
                    // Update volume bar
                    const volumeLevel = Math.min(100, Math.max(0, data.volume / 300));
                    const volumeBar = document.getElementById('volumeBar');
                    volumeBar.style.width = volumeLevel + '%';
                    
                    // Color coding based on volume
                    if (volumeLevel > 70) {{
                        volumeBar.style.backgroundColor = '#e74c3c'; // Red for loud
                    }} else if (volumeLevel > 40) {{
                        volumeBar.style.backgroundColor = '#f39c12'; // Orange for medium
                    }} else {{
                        volumeBar.style.backgroundColor = '#27ae60'; // Green for quiet
                    }}
                    
                    // Update volume graph
                    updateVolumeGraph();
                }}
            }}

            function updateVolumeGraph() {{
                const canvas = document.getElementById('volumeGraph');
                if (!canvas) return;
                
                const ctx = canvas.getContext('2d');
                const width = canvas.width;
                const height = canvas.height;
                
                // Clear canvas
                ctx.clearRect(0, 0, width, height);
                
                // Draw background grid
                ctx.strokeStyle = '#f0f0f0';
                ctx.lineWidth = 1;
                
                // Draw horizontal grid lines
                for (let i = 0; i < height; i += height/5) {{
                    ctx.beginPath();
                    ctx.moveTo(0, i);
                    ctx.lineTo(width, i);
                    ctx.stroke();
                }}
                
                // Draw audio level graph
                if (audioLevels.length > 1) {{
                    const maxVolume = 30000;  // Maximum expected volume
                    
                    // Draw path
                    ctx.beginPath();
                    ctx.moveTo(0, height - (height * audioLevels[0] / maxVolume));
                    
                    for (let i = 1; i < audioLevels.length; i++) {{
                        const x = width * i / (maxAudioPoints - 1);
                        const y = height - (height * audioLevels[i] / maxVolume);
                        ctx.lineTo(x, Math.max(0, Math.min(height, y)));
                    }}
                    
                    // Draw gradient fill
                    const gradient = ctx.createLinearGradient(0, 0, 0, height);
                    gradient.addColorStop(0, 'rgba(231, 76, 60, 0.3)');  // Red at top
                    gradient.addColorStop(0.5, 'rgba(241, 196, 15, 0.3)'); // Yellow in middle
                    gradient.addColorStop(1, 'rgba(39, 174, 96, 0.3)');  // Green at bottom
                    
                    ctx.strokeStyle = '#3498db';
                    ctx.lineWidth = 2;
                    ctx.stroke();
                    
                    // Fill below the line
                    ctx.lineTo(width, height);
                    ctx.lineTo(0, height);
                    ctx.closePath();
                    ctx.fillStyle = gradient;
                    ctx.fill();
                }}
            }}

            function addPrediction(data) {{
                const predictionsContainer = document.getElementById('livePredictions');
                
                // Create prediction item
                const item = document.createElement('div');
                item.className = 'prediction-item';
                
                // Format the confidence percentage
                const confidencePct = (data.confidence * 100).toFixed(1) + '%';
                
                // Create the HTML content - FIXED with double braces
                item.innerHTML = `
                    <span class="prediction-reason">${{data.reason}}</span>
                    <span class="prediction-confidence">${{confidencePct}}</span>
                    <span class="prediction-time">${{data.timestamp}}</span>
                `;
                
                // Add to the beginning of the list
                predictionsContainer.insertBefore(item, predictionsContainer.firstChild);
                
                // Limit to 10 predictions
                if (predictionsContainer.children.length > 10) {{
                    predictionsContainer.removeChild(predictionsContainer.lastChild);
                }}
                
                // Highlight new prediction briefly
                item.style.backgroundColor = '#d4edda';
                setTimeout(() => {{
                    item.style.backgroundColor = '#f1f1f1';
                }}, 2000);
            }}

            // Function to update camera IP
            async function updateCameraIp() {{
                const cameraIp = document.getElementById('cameraIp').value;
                
                try {{
                    // Update the config
                    const response = await fetch('/config', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ camera_ip: cameraIp }})
                    }});
                    
                    if (response.ok) {{
                        // Update the iframe src - FIXED with double braces
                        document.getElementById('cameraStream').src = `http://${{cameraIp}}/`;
                        alert('Camera IP updated successfully!');
                    }} else {{
                        alert('Failed to update camera IP');
                    }}
                }} catch (error) {{
                    console.error('Error:', error);
                    alert('Error updating camera IP');
                }}
            }}

            // Handle form submission
            document.getElementById('upload-form').addEventListener('submit', async function(e) {{
                e.preventDefault();
                const resultElement = document.getElementById('result');
                resultElement.textContent = "Uploading and analyzing...";
                resultElement.className = "result-section";
                
                const formData = new FormData(this);
                
                try {{
                    const response = await fetch('/upload-audio', {{
                        method: 'POST',
                        body: formData
                    }});
                    
                    const data = await response.json();
                    
                    if (data.status === 'success') {{
                        resultElement.innerHTML = `
                            <h3 class="success">Analysis Result</h3>
                            <p>Your baby is: <strong>${{data.reason}}</strong></p>
                            <p>Confidence: ${{(data.confidence * 100).toFixed(2)}}%</p>
                        `;
                        
                        // Also update the baby status under the video - FIXED with double braces
                        document.querySelector('.baby-status').innerHTML = `Your baby is <strong>${{data.reason}}</strong>!`;
                    }} else {{
                        resultElement.innerHTML = `<p class="error">Error: ${{data.message}}</p>`;
                    }}
                }} catch (error) {{
                    resultElement.innerHTML = `<p class="error">Error: ${{error.message}}</p>`;
                }}
            }});

            // Function to periodically update baby status
            async function updateBabyStatus() {{
                try {{
                    const response = await fetch('/history?limit=1');
                    const data = await response.json();
                    
                    if (data.history && data.history.length > 0) {{
                        const latestEvent = data.history[0];
                        document.querySelector('.baby-status').innerHTML = 
                            `Your baby is <strong>${{latestEvent.reason}}</strong>!`;
                    }}
                }} catch (error) {{
                    console.error("Error updating status:", error);
                }}
            }}

            // Function to add HTML for audio visualization components
            function addAudioVisualization() {{
                const container = document.querySelector('.live-monitor');
                if (!container) return;
                
                const visualizationHTML = `
                    <div style="margin-top: 15px;">
                        <h4>Audio Levels</h4>
                        <p>Current Volume: <span id="volumeValue">0</span></p>
                        <div style="width: 100%; background-color: #f1f1f1; border-radius: 4px;">
                            <div id="volumeBar" style="width: 0%; height: 10px; background-color: #27ae60; border-radius: 4px; transition: width 0.2s, background-color 0.2s;"></div>
                        </div>
                        <canvas id="volumeGraph" width="300" height="120" style="margin-top: 10px; width: 100%; height: 120px; border: 1px solid #ddd; border-radius: 4px;"></canvas>
                        
                        <h4 style="margin-top: 15px;">Detection Counters</h4>
                        <div style="display: flex; flex-wrap: wrap; margin-top: 5px; gap: 5px;">
                            <div style="flex: 1; padding: 5px; background: #ffcccb; border-radius: 4px; text-align: center; min-width: 100px;">
                                Unwell: <span id="uncomfortableCounter">0</span>
                            </div>
                            <div style="flex: 1; padding: 5px; background: #c8e6c9; border-radius: 4px; text-align: center; min-width: 100px;">
                                Sleeping: <span id="sleepingCounter">0</span>
                            </div>
                            <div style="flex: 1; padding: 5px; background: #ffecb3; border-radius: 4px; text-align: center; min-width: 100px;">
                                Crying: <span id="cryingCounter">0</span>
                            </div>
                            <div style="flex: 1; padding: 5px; background: #bbdefb; border-radius: 4px; text-align: center; min-width: 100px;">
                                Laughing: <span id="laughingCounter">0</span>
                            </div>
                            <div style="flex: 1; padding: 5px; background: #e8f4f8; border-radius: 4px; text-align: center; min-width: 100px;">
                                Tired: <span id="tiredCounter">0</span>
                            </div>
                        </div>
                    </div>
                `;
                
                // Add the HTML to the container
                container.insertAdjacentHTML('beforeend', visualizationHTML);
            }}

            // Update baby status every 5 seconds
            setInterval(updateBabyStatus, 5000);

            // On page load
            window.addEventListener('load', function() {{
                // Add audio visualization components
                addAudioVisualization();
                
                // Connect WebSocket
                connectWebSocket();
                
                // Initialize volume graph
                updateVolumeGraph();
            }});

            // Ping the WebSocket every 30 seconds to keep the connection alive
            setInterval(() => {{
                if (ws && ws.readyState === WebSocket.OPEN) {{
                    ws.send('ping');
                }}
            }}, 30000);
        </script>
    </body>
    </html>
    """
@app.get("/test-prediction")
async def test_prediction():
    """Test model with a static file"""
    try:
        # Replace with a path to your test file
        test_file = r"D:\ISDN2002\archive\Laugh\laugh_1.m4a_0.wav"
        
        # Check if file exists
        if not os.path.exists(test_file):
            return {"error": f"Test file not found: {test_file}"}
            
        # Load and process audio
        y, sr = librosa.load(test_file, sr=Config.SAMPLE_RATE)
        features = process_raw_audio(y)
        
        # Make prediction
        prediction = model.predict(features, verbose=0)[0]
        
        # Format response
        return {
            "Your baby is": Config.CLASS_LABELS[np.argmax(prediction)]
        }
    except Exception as e:
        return {"error": str(e)}
    
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
            
        logger.info(f"Received audio data of length: {len(audio_data)}, min: {audio_data.min()}, max: {audio_data.max()}")
        
        # Save audio to file for history
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"audio_{timestamp}.npy"
        audio_path = os.path.join(Config.UPLOAD_DIR, audio_filename)
        
        # Save raw audio data
        np.save(audio_path, audio_data)
        logger.info(f"Saved audio to {audio_path}")
        
        # Process the audio data
        features = process_raw_audio(audio_data)
        
        # Make prediction
        prediction = model.predict(features, verbose=0)[0]
        predicted_class, confidence, class_probs = map_prediction_to_new_labels(prediction)

        logger.info(f"Predicted class: {predicted_class} with confidence: {confidence:.4f}")
        
        # Store event in database
        store_cry_event(predicted_class, confidence, audio_path)
        
        # Try to notify devices (don't wait for completion)
        asyncio.create_task(notify_devices(predicted_class, confidence))
        
        # Return result
        return {
            "reason": predicted_class,
            "confidence": confidence,
            "class_probabilities": class_probs,
            "status": "success"
        }
            
    except Exception as e:
        logger.error(f"Error analyzing cry: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/history")
async def get_history(days: int = 7, limit: int = 100):
    """Get cry detection history"""
    history = get_cry_history(days, limit)
    return {"history": history}

@app.get("/config", response_model=ConfigModel)
async def get_config():
    """Get current system configuration"""
    global config
    return config.config

@app.post("/config")
async def update_config(config_data: ConfigModel):
    """Update system configuration"""
    global config
    new_config = config.update(config_data.dict())
    return {"status": "success", "config": new_config}

@app.post("/notify/watchy")
async def notify_watchy(reason: str, message: Optional[str] = None):
    """Manually send notification to Watchy"""
    try:
        watch_ip = config.get("watchy_ip")
        if not watch_ip:
            return {"status": "error", "message": "Watchy IP not configured"}
            
        if not message:
            message = f"BABY:{reason}"
            
        # Send to Watchy
        async with aiohttp.ClientSession() as session:
            watch_url = f"http://{watch_ip}/vibrate"
            async with session.post(
                watch_url, 
                json={"reason": reason, "message": message}, 
                timeout=5
            ) as response:
                if response.status == 200:
                    return {"status": "success", "message": "Notification sent to Watchy"}
                else:
                    return {"status": "error", "message": f"Failed with status {response.status}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/notify/parent")
async def notify_parent(reason: str, confidence: float):
    """Manually send notification to parent app"""
    # This would integrate with your parent app notification system
    logger.info(f"Parent notification: {reason} ({confidence:.2f})")
    return {"status": "success", "message": "Parent notification sent"}

@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    """Upload audio file for analysis"""
    try:
        # Save file code remains the same...
        
        # Make prediction
        full_prediction = model.predict(features, verbose=0)[0]
        
        # TRUNCATE to 6 classes
        prediction = full_prediction[:6]
        prediction = prediction / np.sum(prediction)  # Re-normalize
        
        # Get predicted class and confidence
        predicted_idx = np.argmax(prediction)
        confidence = float(prediction[predicted_idx])
        predicted_class = Config.CLASS_LABELS[predicted_idx]
        
        # Create class probabilities dictionary
        class_probs = {cls: float(prob) for cls, prob in zip(Config.CLASS_LABELS, prediction)}
        
        # Store in database
        store_cry_event(predicted_class, confidence, filepath)
        
        return {
            "reason": predicted_class,
            "confidence": confidence,
            "class_probabilities": class_probs,
            "status": "success",
            "filepath": filepath
        }
        
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/test")
async def test_endpoint():
    """Simple test endpoint to verify API functionality"""
    return {"status": "success", "message": "Test endpoint is working"}

# Expose static files for UI
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except:
    logger.warning("Static files directory not found. UI resources may be limited.")

# Print server access information
def print_server_info():
    local_ip = get_local_ip()
    logger.info("=" * 50)
    logger.info("Baby Cry Detection System")
    logger.info("=" * 50)
    logger.info(f"Server running at:")
    logger.info(f"- Local:   http://localhost:5000")
    logger.info(f"- Network: http://{local_ip}:5000")
    logger.info("=" * 50)
    
if __name__ == "__main__":
    # Print server access information
    print_server_info()
    
    # Run with uvicorn directly - no NiceGUI integration
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)