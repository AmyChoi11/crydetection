#include <WiFi.h>
#include <HTTPClient.h>
#include <Arduino.h>

// Network credentials
const char* ssid = "SmarTone_HBB_4022";
const char* password = "5W5AABFB25";

// Server details
const char* serverUrl = "http://10.89.195.233:5000/analyze-cry";  // Replace with your PC's IP


// Audio recording configuration
const int micPin = 34;           // Microphone pin
const int statusLED = 2;         // Built-in LED pin
const int sampleRate = 8000;     // Sample rate in Hz
const int threshold = 500;       // Audio amplitude threshold for cry detection
const int recordingDuration = 5; // Duration in seconds
const int samplesPerSegment = recordingDuration * sampleRate; // Total samples to record
const int bufferSize = 1024;     // Samples per buffer
const int cooldownPeriod = 10;   // Time between detections in seconds

// Variables
int16_t audioBuffer[bufferSize]; // Buffer for audio samples
int16_t recordingBuffer[samplesPerSegment]; // Full recording buffer
unsigned long lastRecordingTime = 0;
bool isRecording = false;
int recordingIndex = 0;
int noiseLevel = 0;
int consecutiveHighSamples = 0;
const int minHighSamples = 20; // Minimum number of samples above threshold to trigger

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  
  // Configure pins
  pinMode(statusLED, OUTPUT);
  digitalWrite(statusLED, LOW);
  
  // Connect to WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    digitalWrite(statusLED, !digitalRead(statusLED)); // Blink LED while connecting
  }
  
  // Connected to WiFi
  Serial.println("\nWiFi connected");
  Serial.println("IP address: " + WiFi.localIP().toString());
  digitalWrite(statusLED, HIGH);
  
  // ADC settings for better audio quality
  analogSetWidth(12);        // 12-bit resolution for ADC
  analogSetAttenuation(ADC_11db); // Set attenuation for higher voltage range
  
  // Calculate background noise level
  calibrateMicrophone();
}

void loop() {
  // Check WiFi connection
  ensureWifiConnection();
  
  // If not recording, check for cry detection
  if (!isRecording) {
    detectCry();
  } else {
    // Continue recording
    recordAudio();
  }
}

void detectCry() {
  // Check if cooldown period has passed
  if (millis() - lastRecordingTime < cooldownPeriod * 1000) {
    return;
  }
  
  // Read current sample
  int sample = abs(analogRead(micPin) - 2048); // Convert to signed value centered at 0
  
  // Check if above threshold
  if (sample > threshold) {
    consecutiveHighSamples++;
    
    // Visual feedback
    if (consecutiveHighSamples % 5 == 0) {
      digitalWrite(statusLED, !digitalRead(statusLED));
    }
    
    // If enough consecutive samples above threshold, start recording
    if (consecutiveHighSamples >= minHighSamples) {
      Serial.println("Cry detected! Starting recording...");
      startRecording();
      consecutiveHighSamples = 0;
    }
  } else {
    // Reset counter if sample is below threshold
    consecutiveHighSamples = 0;
    digitalWrite(statusLED, HIGH); // Keep LED on when idle
  }
}

void startRecording() {
  isRecording = true;
  recordingIndex = 0;
  
  // Clear recording buffer
  memset(recordingBuffer, 0, samplesPerSegment * sizeof(int16_t));
  
  // Visual indication that recording has started
  digitalWrite(statusLED, LOW);
}

void recordAudio() {
  // Calculate how many samples we can read in this loop
  int samplesToRead = min(bufferSize, samplesPerSegment - recordingIndex);
  
  if (samplesToRead <= 0) {
    // Recording finished
    isRecording = false;
    lastRecordingTime = millis();
    
    Serial.println("Recording complete. Sending to server...");
    digitalWrite(statusLED, HIGH); // LED on while processing
    
    // Send recording to server
    sendRecording();
    
    return;
  }
  
  // Read samples into buffer
  for (int i = 0; i < samplesToRead; i++) {
    // Read from ADC and convert to signed 16-bit
    int16_t sample = analogRead(micPin) - 2048; // Center at 0
    sample = sample << 4; // Scale to 16-bit range
    
    // Store in recording buffer
    recordingBuffer[recordingIndex++] = sample;
    
    // Brief delay to maintain sample rate
    delayMicroseconds(1000000 / sampleRate);
  }
  
  // Visual feedback
  if (recordingIndex % (sampleRate / 2) == 0) { // Blink twice per second
    digitalWrite(statusLED, !digitalRead(statusLED));
  }
}

void sendRecording() {
  // Ensure WiFi is connected
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi not connected. Cannot send recording.");
    return;
  }
  
  HTTPClient http;
  
  // Configure HTTP client
  http.begin(serverUrl);
  http.addHeader("Content-Type", "application/octet-stream");
  
  // Visual indication - rapid blinking during transmission
  for (int i = 0; i < 5; i++) {
    digitalWrite(statusLED, HIGH);
    delay(50);
    digitalWrite(statusLED, LOW);
    delay(50);
  }
  
  // Send the recording as raw bytes
  int httpResponseCode = http.POST((uint8_t*)recordingBuffer, samplesPerSegment * sizeof(int16_t));
  
  if (httpResponseCode > 0) {
    String response = http.getString();
    Serial.println("HTTP Response code: " + String(httpResponseCode));
    Serial.println("Response: " + response);
    
    // Success indication - solid LED for 1 second
    digitalWrite(statusLED, HIGH);
    delay(1000);
  } else {
    Serial.println("Error sending recording: " + String(httpResponseCode));
    
    // Error indication - three quick blinks
    for (int i = 0; i < 3; i++) {
      digitalWrite(statusLED, HIGH);
      delay(100);
      digitalWrite(statusLED, LOW);
      delay(100);
    }
  }
  
  http.end();
}

void calibrateMicrophone() {
  Serial.println("Calibrating microphone...");
  digitalWrite(statusLED, LOW);
  
  // Sample background noise level
  long sum = 0;
  const int numSamples = 1000;
  
  for (int i = 0; i < numSamples; i++) {
    int sample = abs(analogRead(micPin) - 2048);
    sum += sample;
    delayMicroseconds(1000);
  }
  
  noiseLevel = sum / numSamples;
  
  Serial.println("Microphone calibration complete");
  Serial.println("Background noise level: " + String(noiseLevel));
  
  // Adjust threshold based on noise level (optional)
  // threshold = max(threshold, noiseLevel * 3);
  
  digitalWrite(statusLED, HIGH);
}

void ensureWifiConnection() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi connection lost. Reconnecting...");
    
    // Store LED state
    int ledState = digitalRead(statusLED);
    
    // Blink rapidly during reconnection
    for (int i = 0; i < 10; i++) {
      digitalWrite(statusLED, HIGH);
      delay(50);
      digitalWrite(statusLED, LOW);
      delay(50);
    }
    
    // Try to reconnect
    WiFi.begin(ssid, password);
    
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
      delay(500);
      Serial.print(".");
      attempts++;
    }
    
    if (WiFi.status() == WL_CONNECTED) {
      Serial.println("WiFi reconnected");
      // Restore LED
      digitalWrite(statusLED, ledState);
    } else {
      Serial.println("Failed to reconnect WiFi");
      // Error indication
      for (int i = 0; i < 3; i++) {
        digitalWrite(statusLED, HIGH);
        delay(300);
        digitalWrite(statusLED, LOW);
        delay(300);
      }
    }
  }
}