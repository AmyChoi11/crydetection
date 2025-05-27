import requests
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import json

# API Configuration
BASE_URL = "http://10.89.195.233:5000"
API_ENDPOINT = f"{BASE_URL}/analyze-cry"

# Test FastAPI server
def test_server_connection():
    try:
        response = requests.get(BASE_URL)
        if response.status_code == 200:
            print("✅ Server connection successful")
            return True
        else:
            print(f"❌ Server error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Server connection failed: {e}")
        return False

# Upload a test WAV file and get prediction
def test_upload(file_path):
    try:
        with open(file_path, 'rb') as audio_file:
            # Send raw audio data directly to the analyze-cry endpoint
            response = requests.post(API_ENDPOINT, data=audio_file.read())
        
        if response.status_code == 200:
            result = response.json()
            print("\n=== Prediction Result ===")
            print(f"File: {os.path.basename(file_path)}")
            print(f"Detected: {result.get('reason', 'Unknown')}")
            print(f"Confidence: {result.get('confidence', 0):.2f}")
            
            # Print all probabilities
            print("\nAll probabilities:")
            probs = result.get('class_probabilities', {})
            for label, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                print(f"  {label}: {prob:.2f}")
            
            # Plot probabilities
            plot_probabilities(probs)
            
            return True
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return False

# Plot probability distribution
def plot_probabilities(probs):
    labels = list(probs.keys())
    values = list(probs.values())
    
    # Sort by value
    sorted_indices = np.argsort(values)
    sorted_labels = [labels[i] for i in sorted_indices[::-1]]
    sorted_values = [values[i] for i in sorted_indices[::-1]]
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(sorted_labels, sorted_values, color='skyblue')
    plt.xlabel('Probability (%)')
    plt.title('Prediction Probabilities')
    plt.xlim(0, 100)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2,
                 f'{width:.1f}%', ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig('prediction_result.png')
    plt.show()

def main():
    print("=== Baby Cry Detection System Test ===")
    
    # Test server connection
    if not test_server_connection():
        print("Please make sure the FastAPI server is running.")
        return
    
    # Test with sample file
    test_file = input("Enter path to WAV file for testing (or press Enter for default): ")
    if not test_file:
        # Look for any WAV file in the current directory
        wav_files = [f for f in os.listdir('.') if f.endswith('.wav')]
        if wav_files:
            test_file = wav_files[0]
            print(f"Using {test_file} for testing.")
        else:
            print("No WAV files found in current directory.")
            return
    
    if os.path.exists(test_file):
        test_upload(test_file)
    else:
        print(f"File not found: {test_file}")

if __name__ == "__main__":
    main()