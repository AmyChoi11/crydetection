import numpy as np
import librosa
import tensorflow as tf
import logging
import os
import aiohttp
import asyncio
import sqlite3
import json
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
from nicegui import ui, app as nicegui_app
import scipy.stats  # Needed for entropy calculation

# Configure environment variables
os.environ["WATCHY_IP"] = "192.168.4.1"  # Replace with actual Watchy IP

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

nicegui_app.app = app  # Expose FastAPI through NiceGUI

# Create directories if they don't exist
os.makedirs("audio_clips", exist_ok=True)
os.makedirs("db", exist_ok=True)

# Global variables
model = None
mean_value = None
std_value = None
config = None

# Configuration
class Config:
    CLASS_LABELS = ['Unwell', 'Sleeping', 'Cry', 'Laugh', 'Tired', 'Silence']
    NUM_CLASSES = len(CLASS_LABELS)
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

# Create UI
@ui.page('/ui')
def ui_page():
    with ui.card().classes('w-full'):
        ui.label('Baby Cry Detection System').classes('text-2xl font-bold')
        ui.separator()
        
        with ui.tabs().classes('w-full') as tabs:
            dashboard = ui.tab('Dashboard')
            history = ui.tab('History')
            settings = ui.tab('Settings')
        
        with ui.tab_panels(tabs, value=dashboard).classes('w-full'):
            with ui.tab_panel(dashboard):
                ui.label('System Status').classes('text-xl')
                status_card = ui.card().classes('bg-green-100')
                with status_card:
                    status_label = ui.label('System Online')
                    
                ui.label('Last Detection:').classes('text-lg mt-4')
                last_detection = ui.card().classes('bg-blue-50')
                with last_detection:
                    reason_label = ui.label('No recent detection')
                
                async def update_dashboard():
                    # This would update in real implementation
                    pass
                
                ui.timer(10, update_dashboard)
                
            with ui.tab_panel(history):
                ui.label('Cry Event History').classes('text-xl')
                
                async def load_history():
                    history_data = get_cry_history()
                    with history_table:
                        for item in history_data:
                            ui.label(f"{item['timestamp']} - {item['reason']} ({item['confidence']:.1%})")
                
                history_table = ui.card().classes('w-full')
                ui.button('Refresh History', on_click=load_history)
                
            with ui.tab_panel(settings):
                ui.label('System Settings').classes('text-xl')
                
                threshold = ui.slider(min=100, max=1000, value=config.get('threshold'))
                ui.label('Detection Threshold').bind_value_from(threshold, 'value')
                
                cooldown = ui.slider(min=10, max=300, value=config.get('notification_cooldown'))
                ui.label('Notification Cooldown (seconds)').bind_value_from(cooldown, 'value')
                
                confidence = ui.slider(min=0.5, max=0.95, step=0.05, value=config.get('confidence_threshold'))
                ui.label('Minimum Confidence').bind_value_from(confidence, 'value')
                
                watchy_ip = ui.input('Watchy IP Address', value=config.get('watchy_ip'))
                
                notify_parent = ui.checkbox('Notify Parent App')
                notify_watchy = ui.checkbox('Notify Watchy Watch')
                
                async def save_settings():
                    new_config = {
                        'threshold': threshold.value,
                        'notification_cooldown': cooldown.value,
                        'confidence_threshold': confidence.value,
                        'watchy_ip': watchy_ip.value,
                        'notify_parent': notify_parent.value,
                        'notify_watchy': notify_watchy.value
                    }
                    
                    global config
                    config = AppConfiguration()
                    config.update(new_config)
                    ui.notify('Settings saved successfully')
                
                ui.button('Save Settings', on_click=save_settings).classes('mt-4 bg-blue-500 text-white')

# Load model at startup
@app.on_event("startup")
async def load_model():
    """Load the TensorFlow model and initialize system when the application starts"""
    global model, mean_value, std_value, config
    
    # Initialize database
    initialize_database()
    
    # Initialize configuration
    config = AppConfiguration()
    
    try:
        # Build model architecture
        from tensorflow.keras.layers import (
            InputLayer, BatchNormalization, Dense, Dropout, Add, 
            Input, InputLayer
        )
        from tensorflow.keras.regularizers import l2
        
        # Get feature size dynamically 
        feature_size = Config.N_MFCC * 13  # 13 feature types per coefficient
        logger.info(f"Building model with input size: {feature_size}")
        
        # Create model using Functional API
        inputs = Input(shape=(feature_size,))
        x = BatchNormalization()(inputs)
        
        # First dense layer with dropout
        x1 = Dense(320, activation='relu', kernel_regularizer=l2(1e-5))(x)
        x1 = BatchNormalization()(x1)
        x1 = Dropout(0.35)(x1)
        
        # First residual block
        x2 = Dense(160, activation='relu', kernel_regularizer=l2(1e-5))(x1)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(0.3)(x2)
        
        # Residual connection
        x_res1 = Dense(160, activation='linear')(x1)
        x2_combined = Add()([x2, x_res1])
        
        # Second residual block
        x3 = Dense(80, activation='relu', kernel_regularizer=l2(1e-5))(x2_combined)
        x3 = BatchNormalization()(x3)
        x3 = Dropout(0.25)(x3)
        
        # Another residual connection
        x_res2 = Dense(80, activation='linear')(x2_combined)
        x3_combined = Add()([x3, x_res2])
        
        # Final compression layer
        x4 = Dense(40, activation='relu', kernel_regularizer=l2(1e-5))(x3_combined)
        x4 = BatchNormalization()(x4)
        x4 = Dropout(0.2)(x4)
        
        # Output layer
        outputs = Dense(Config.NUM_CLASSES, activation='softmax', dtype='float32')(x4)
        
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        
        # Try to load saved model weights
        try:
            logger.info(f"Class labels: {Config.CLASS_LABELS}")
            model.load_weights('newest_model.keras')
            logger.info(f"Model weights loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load model weights: {str(e)}. Using uninitialized model.")
        
        # Try to load normalization values
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

# Process raw audio data
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
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio_float, 
            sr=Config.SAMPLE_RATE,
            n_mfcc=Config.N_MFCC,
            n_fft=Config.N_FFT,
            hop_length=Config.HOP_LENGTH
        )
        
        # Statistical features
        mean_features = np.mean(mfcc, axis=1)  # Central tendency
        std_features = np.std(mfcc, axis=1)    # Variability
        max_features = np.max(mfcc, axis=1)    # Peak values
        
        # If we have enough frames for temporal analysis
        if mfcc.shape[1] > 1:
            # Temporal variability (how quickly features change)
            delta_features = np.std(np.diff(mfcc, axis=1), axis=1)
            
            # Entropy (randomness measure)
            # Add small constant to avoid log(0)
            entropy_features = scipy.stats.entropy(np.abs(mfcc) + 1e-10, axis=1)
            
            # Zero-crossing rate - helps distinguish speech vs. noise
            zcr_features = np.mean(np.abs(np.diff(np.sign(mfcc), axis=1)), axis=1) / 2
            
            # Quartile features - help distinguish sound distributions
            q1_features = np.percentile(mfcc, 25, axis=1)
            q3_features = np.percentile(mfcc, 75, axis=1)
            
            # Envelope features - help with detecting cries vs other sounds
            env = np.max(np.abs(mfcc), axis=0)
            env_std = np.std(env) * np.ones(Config.N_MFCC)
            env_rate = np.mean(np.abs(np.diff(env))) * np.ones(Config.N_MFCC)
            
            # Spectral contrast for better noise/speech separation
            spectral_centroid = np.mean(np.abs(mfcc) * np.arange(mfcc.shape[1])[None, :] / mfcc.shape[1], axis=1)
            
            # Periodicity features (helps distinguish cry from laugh)
            autocorr_features = []
            for i in range(mfcc.shape[0]):
                # Get the series for this coefficient
                series = mfcc[i, :]
                # Normalize the series
                series = (series - np.mean(series)) / (np.std(series) + 1e-8)
                # Calculate autocorrelation at lag 1
                if len(series) > 1:
                    ac1 = np.corrcoef(series[:-1], series[1:])[0,1] if len(series) > 1 else 0
                    autocorr_features.append(ac1)
                else:
                    autocorr_features.append(0)
            autocorr_features = np.array(autocorr_features)
            
            # Spectral flatness (distinguish noise types)
            spec_flat = np.std(mfcc, axis=1) / (np.mean(np.abs(mfcc), axis=1) + 1e-8)
        else:
            # Fallback if very short audio - fill with zeros
            extra_features = np.zeros_like(mean_features)
            delta_features = extra_features
            entropy_features = extra_features
            zcr_features = extra_features
            q1_features = extra_features
            q3_features = extra_features
            env_std = extra_features
            env_rate = extra_features
            spectral_centroid = extra_features
            autocorr_features = extra_features
            spec_flat = extra_features
        
        # Combine all features
        combined_features = np.concatenate([
            mean_features, std_features, max_features, 
            delta_features, entropy_features, 
            zcr_features, q1_features, q3_features,
            env_std, env_rate,
            spectral_centroid, autocorr_features, spec_flat
        ])
        
        # Add batch dimension
        features = np.expand_dims(combined_features, axis=0)
        
        # Apply normalization
        normalized_features = (features - mean_value) / (std_value + 1e-8)
        
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
        logger.error(f"Failed to notify devices: {str(e)}"))
        
        # Notify mobile app if enabled
        if config.get("notify_parent"):
            logger.info("Would notify parent mobile app here")
            # In a real implementation, you would:
            # 1. Send push notification via Firebase Cloud Messaging
            # 2. Update mobile app via websocket connection
            # 3. Trigger other integrations as needed
                    
    except Exception as e:
        logger.error(f"Failed to notify devices: {str(e)}")

# You need to implement a notification system for mobile app
async def send_push_notification(reason, confidence):
    """Send push notification to parent's mobile app"""
    try:
        # Using Firebase Cloud Messaging (FCM) for push notifications
        # You'll need to set up a Firebase project and add credentials
        from firebase_admin import messaging
        
        message = messaging.Message(
            notification=messaging.Notification(
                title="Baby Alert",
                body=f"Baby is {reason} (Confidence: {confidence:.1%})"
            ),
            token="DEVICE_TOKEN_HERE"  # Replace with actual device token
        )
        
        response = messaging.send(message)
        logger.info(f"FCM notification sent: {response}")
        return True
    except Exception as e:
        logger.error(f"Failed to send push notification: {str(e)}")
        return False
    

# API Routes
@app.get("/", response_model=StatusResponse)
async def root():
    """Root endpoint to check if the API is running"""
    return {
        "status": "online", 
        "message": "Baby Cry Detection API is running"
    }

@app.get("/test-prediction")
async def test_prediction():
    """Test model with a static file"""
    try:
        import librosa
        # Replace with a path to your test file
        test_file = r"D:\ISDN2002\archive\belly_pain\69BDA5D6-0276-4462-9BF7-951799563728-1436936185-1.1-m-26-bp.wav" 
        
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
            "predictions": {cls: float(prob) for cls, prob in zip(Config.CLASS_LABELS, prediction)},
            "highest": Config.CLASS_LABELS[np.argmax(prediction)]
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
        
        # Get predicted class and confidence
        predicted_idx = np.argmax(prediction)
        confidence = float(prediction[predicted_idx])
        predicted_class = Config.CLASS_LABELS[predicted_idx]

        logger.info(f"Predicted class: {predicted_class} with confidence: {confidence:.4f}")
        
        # Create class probabilities dictionary
        class_probs = {cls: float(prob) for cls, prob in zip(Config.CLASS_LABELS, prediction)}
        
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
        # Create file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extension = file.filename.split(".")[-1].lower()
        filename = f"upload_{timestamp}.{extension}"
        filepath = os.path.join(Config.UPLOAD_DIR, filename)
        
        # Save the file
        with open(filepath, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Load the audio
        y, sr = librosa.load(filepath, sr=Config.SAMPLE_RATE)
        
        # Process audio
        features = process_raw_audio(y)
        
        # Make prediction
        prediction = model.predict(features, verbose=0)[0]
        
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

# For running directly
if __name__ == "__main__":
    import uvicorn
    import scipy.stats  # Import needed for audio processing
    uvicorn.run(app, host="0.0.0.0", port=5000)