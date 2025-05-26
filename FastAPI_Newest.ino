#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>
#include <ArduinoJson.h>
#include <I2S.h>

// Audio Configuration
#define SAMPLE_RATE 16000
#define SAMPLE_BITS 16
#define CLIP_DURATION 5 // seconds
#define BUFFER_SIZE (SAMPLE_RATE * CLIP_DURATION)
#define VOLUME_THRESHOLD 3.0 // Multiplier of average volume
#define MIN_TRIGGER_INTERVAL 10000 // 10 seconds cooldown

// Network Configuration
const char* ssid = "Livebox-D510";
const char* password = "MwHUYQoKrYPVV5t4kM";

// AI Server Configuration (Updated with your endpoint)
const char* aiServer = "10.89.195.233";
const int aiPort = 5000;
const String aiEndpoint = "/analyze_audio"; // From your FastAPI docs

// Device Configuration
const char* watchyIP = "192.168.1.100"; // Update with your Watchy's IP
const char* appServer = "APP_SERVER_IP"; // Update with your app server IP

ESP8266WebServer localServer(80);
int16_t audioBuffer[BUFFER_SIZE];

void setup() {
  Serial.begin(115200);
  
  // Initialize I2S for audio recording
  I2S.setSckPin(D5);
  I2S.setDataPin(D7);
  if(!I2S.begin(SAMPLE_RATE, SAMPLE_BITS)) {
    Serial.println("Failed to initialize I2S!");
    while(1); // Halt if audio fails
  }

  // Connect to WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while(WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nConnected! IP: " + WiFi.localIP().toString());
  
  // Setup local API endpoint
  localServer.on("/trigger-recording", HTTP_POST, handleRecording);
  localServer.begin();
}

void loop() {
  localServer.handleClient();
  monitorAudio();
}

void monitorAudio() {
  static unsigned long lastTrigger = 0;
  static bool isRecording = false;
  static size_t recordIndex = 0;
  static float avgVolume = 0;
  int16_t sample;
  
  if(I2S.read(&sample, sizeof(sample))) {
    float instantVolume = abs(sample);
    avgVolume = 0.95 * avgVolume + 0.05 * instantVolume;
    
    // Trigger recording when volume exceeds threshold and cooldown has passed
    if(!isRecording && instantVolume > avgVolume * VOLUME_THRESHOLD && 
       millis() - lastTrigger > MIN_TRIGGER_INTERVAL) {
      isRecording = true;
      recordIndex = 0;
      Serial.println("Starting recording...");
    }
    
    if(isRecording) {
      audioBuffer[recordIndex++] = sample;
      if(recordIndex >= BUFFER_SIZE) {
        Serial.println("Processing recording...");
        processRecording();
        isRecording = false;
        lastTrigger = millis();
      }
    }
  }
}

void processRecording() {
  WiFiClient client;
  HTTPClient http;
  
  // Construct AI server URL
  String aiUrl = "http://" + String(aiServer) + ":" + String(aiPort) + aiEndpoint;
  
  http.begin(client, aiUrl);
  http.addHeader("Content-Type", "application/octet-stream");
  
  // Send audio data (16-bit samples, so BUFFER_SIZE * 2 bytes)
  int httpCode = http.POST((uint8_t*)audioBuffer, BUFFER_SIZE * 2);
  
  if(httpCode == HTTP_CODE_OK) {
    String payload = http.getString();
    Serial.println("AI Response: " + payload);
    
    DynamicJsonDocument doc(256);
    deserializeJson(doc, payload);
    
    if(doc.containsKey("reason")) {
      String reason = doc["reason"];
      notifyDevices(reason);
    } else {
      Serial.println("No 'reason' field in response");
    }
  } else {
    Serial.printf("AI request failed, error: %s\n", http.errorToString(httpCode).c_str());
  }
  http.end();
}

void notifyDevices(String reason) {
  // 1. Notify Watchy to vibrate
  HTTPClient watchyHttp;
  watchyHttp.begin("http://" + String(watchyIP) + "/vibrate");
  int watchyCode = watchyHttp.POST("");
  Serial.printf("Watchy notification: %d\n", watchyCode);
  watchyHttp.end();
  
  // 2. Send detailed reason to app server
  HTTPClient appHttp;
  appHttp.begin("http://" + String(appServer) + "/baby-alert");
  
  DynamicJsonDocument doc(128);
  doc["reason"] = reason;
  doc["timestamp"] = millis();
  
  String json;
  serializeJson(doc, json);
  
  appHttp.addHeader("Content-Type", "application/json");
  int appCode = appHttp.POST(json);
  Serial.printf("App notification: %d\n", appCode);
  appHttp.end();
}

void handleRecording() {
  // Manual trigger endpoint if needed
  monitorAudio(); // Force check/recording
  localServer.send(200, "text/plain", "Recording triggered");
}