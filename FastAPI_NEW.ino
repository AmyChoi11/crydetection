#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>
#include <ArduinoJson.h>
#include <I2S.h> // For audio recording

// Audio Config
#define SAMPLE_RATE 16000
#define SAMPLE_BITS 16
#define CLIP_DURATION 5 // seconds
#define BUFFER_SIZE (SAMPLE_RATE * CLIP_DURATION)

// WiFi Config
const char* ssid = "Livebox-D510";
const char* password = "MwHUYQoKrYPVV5t4kM";

// Server Config
const char* aiServer = "AI_SERVER_IP:5000";
const char* watchyIP = "WATCHY_IP";
const char* appServer = "APP_SERVER_IP";

ESP8266WebServer server(80);
int16_t audioBuffer[BUFFER_SIZE];

void setup() {
  Serial.begin(115200);
  
  // Init I2S for audio
  I2S.setSckPin(D5);
  I2S.setDataPin(D7);
  if(!I2S.begin(SAMPLE_RATE, SAMPLE_BITS)) {
    Serial.println("Failed to init I2S!");
  }

  // Connect WiFi
  WiFi.begin(ssid, password);
  while(WiFi.status() != WL_CONNECTED) delay(500);
  
  // API Endpoints
  server.on("/trigger-recording", HTTP_POST, handleRecording);
  server.begin();
}

void loop() {
  server.handleClient();
  monitorAudio();
}

void monitorAudio() {
  static unsigned long lastTrigger = 0;
  static bool isRecording = false;
  static size_t recordIndex = 0;
  
  // Simple VAD (Voice Activity Detection)
  int16_t sample;
  static float avgVolume = 0;
  
  if(I2S.read(&sample, sizeof(sample))) {
    float instantVolume = abs(sample);
    avgVolume = 0.95 * avgVolume + 0.05 * instantVolume;
    
    // Start recording if volume threshold crossed
    if(!isRecording && instantVolume > avgVolume * 3.0 && 
       millis() - lastTrigger > 10000) {
      isRecording = true;
      recordIndex = 0;
    }
    
    if(isRecording) {
      audioBuffer[recordIndex++] = sample;
      if(recordIndex >= BUFFER_SIZE) {
        processRecording();
        isRecording = false;
        lastTrigger = millis();
      }
    }
  }
}

void processRecording() {
  // Send to AI Model
  WiFiClient client;
  HTTPClient http;
  
  http.begin(client, String("http://") + aiServer + "/analyze-cry");
  http.addHeader("Content-Type", "application/octet-stream");
  
  // Send raw audio
  int httpCode = http.POST((uint8_t*)audioBuffer, BUFFER_SIZE * 2);
  
  if(httpCode == HTTP_CODE_OK) {
    String payload = http.getString();
    DynamicJsonDocument doc(256);
    deserializeJson(doc, payload);
    
    String reason = doc["reason"];
    storeAndNotify(reason);
  }
  http.end();
}

void storeAndNotify(String reason) {
  // Store in local memory
  // (For persistent storage, use EEPROM or SPIFFS)
  
  // Notify Watchy
  WiFiClient watchyClient;
  HTTPClient watchyHttp;
  watchyHttp.begin(watchyClient, String("http://") + watchyIP + "/vibrate");
  watchyHttp.POST("");
  watchyHttp.end();
  
  // Notify App
  WiFiClient appClient;
  HTTPClient appHttp;
  appHttp.begin(appClient, String("http://") + appServer + "/baby-alert");
  
  DynamicJsonDocument doc(128);
  doc["reason"] = reason;
  String json;
  serializeJson(doc, json);
  
  appHttp.addHeader("Content-Type", "application/json");
  appHttp.POST(json);
  appHttp.end();
}