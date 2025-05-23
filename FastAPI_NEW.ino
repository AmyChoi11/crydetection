#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>
#include <ArduinoJson.h>
// Use the correct I2S library
#include <ESP8266Audio.h>
#include <AudioInputI2S.h>

// Audio Config
#define SAMPLE_RATE 16000
#define SAMPLE_BITS 16
#define CLIP_DURATION 5 // seconds
#define BUFFER_SIZE (SAMPLE_RATE * CLIP_DURATION)

// WiFi Config
const char* ssid = "Livebox-D510";
const char* password = "MwHUYQoKrYPVV5t4kM";

// Server Config
const char* aiServer = "10.89.195.233:5000";  // Add the port number
const char* predictEndpoint = "/analyze-cry";
// Update these with your actual device IPs when ready
const char* watchyIP = "192.168.1.X";  // Replace X with your Watchy's IP last octet
const char* appServer = "192.168.1.Y"; // Replace Y with your app server's IP last octet

ESP8266WebServer server(80);
int16_t audioBuffer[BUFFER_SIZE];
bool debugMode = true; // Enable debug messages

// Audio input object
AudioInputI2S i2sIn;

// Send audio data to AI server
String sendToAIServer(int16_t* audioData, int dataSize) {
  if (debugMode) {
    Serial.println("Sending audio data to AI server...");
    Serial.print("URL: http://");
    Serial.print(aiServer);
    Serial.println(predictEndpoint);
    Serial.print("Data size: ");
    Serial.println(dataSize * 2);
  }
  
  WiFiClient client;
  HTTPClient http;
  
  String url = String("http://") + aiServer + predictEndpoint;
  
  http.begin(client, url);
  http.addHeader("Content-Type", "application/octet-stream");
  
  // Send raw audio data
  int httpCode = http.POST((uint8_t*)audioData, dataSize * 2); // *2 because int16_t is 2 bytes
  
  if (debugMode) {
    Serial.print("HTTP response code: ");
    Serial.println(httpCode);
  }
  
  String response = "";
  if (httpCode > 0) {
    response = http.getString();
    if (debugMode) {
      Serial.print("Response: ");
      Serial.println(response);
    }
  } else {
    if (debugMode) {
      Serial.print("Error: ");
      Serial.println(http.errorToString(httpCode));
    }
  }
  
  http.end();
  return response;
}

// Handle manual recording trigger
void handleRecording() {
  // Start recording
  server.send(200, "text/plain", "Recording started");
  
  if (debugMode) Serial.println("Manual recording triggered");
  
  // Fill buffer with audio
  int samples = 0;
  while (samples < BUFFER_SIZE) {
    // Read audio from I2S
    size_t bytesRead = i2sIn.read(audioBuffer + samples, sizeof(int16_t));
    if (bytesRead == sizeof(int16_t)) {
      samples++;
      if (samples % 1000 == 0 && debugMode) {
        Serial.print("Recording... ");
        Serial.print(samples);
        Serial.print("/");
        Serial.println(BUFFER_SIZE);
      }
    }
  }
  
  if (debugMode) Serial.println("Recording complete, processing...");
  processRecording();
}

// Test the AI server connection with a small amount of data
void testConnection() {
  if (debugMode) Serial.println("Testing connection to AI server...");
  
  // Create simple test data
  int16_t testData[100];
  for (int i = 0; i < 100; i++) {
    testData[i] = i * 100;
  }
  
  // Send test data
  String response = sendToAIServer(testData, 100);
  
  if (debugMode) {
    Serial.println("Test complete");
  }
}

void storeAndNotify(String reason) {
  if (debugMode) {
    Serial.print("Baby cry detected: ");
    Serial.println(reason);
  }
  
  // Notify Watchy
  if (watchyIP && strlen(watchyIP) > 0) {
    WiFiClient watchyClient;
    HTTPClient watchyHttp;
    String watchyUrl = String("http://") + watchyIP + "/vibrate";
    
    if (debugMode) {
      Serial.print("Notifying Watchy at: ");
      Serial.println(watchyUrl);
    }
    
    watchyHttp.begin(watchyClient, watchyUrl);
    int httpCode = watchyHttp.POST("");
    
    if (debugMode) {
      Serial.print("Watchy notification result: ");
      Serial.println(httpCode);
    }
    
    watchyHttp.end();
  }
  
  // Notify App
  if (appServer && strlen(appServer) > 0) {
    WiFiClient appClient;
    HTTPClient appHttp;
    String appUrl = String("http://") + appServer + "/baby-alert";
    
    if (debugMode) {
      Serial.print("Notifying App at: ");
      Serial.println(appUrl);
    }
    
    DynamicJsonDocument doc(128);
    doc["reason"] = reason;
    String json;
    serializeJson(doc, json);
    
    appHttp.begin(appClient, appUrl);
    appHttp.addHeader("Content-Type", "application/json");
    int httpCode = appHttp.POST(json);
    
    if (debugMode) {
      Serial.print("App notification result: ");
      Serial.println(httpCode);
    }
    
    appHttp.end();
  }
}

void setup() {
  Serial.begin(115200);
  Serial.println("\n\nBaby Cry Detection System Starting");
  
  // Init I2S for audio
  Serial.println("Initializing I2S...");
  i2sIn.begin(SAMPLE_RATE, I2S_PHILIPS_MODE, I2S_BITS_PER_SAMPLE_16BIT, I2S_MONO);
  
  // Set I2S pins
  i2sIn.setPin(I2S_PIN_BCLK, D5);
  i2sIn.setPin(I2S_PIN_DATA, D7);
  
  Serial.println("I2S initialized successfully");

  // Connect WiFi
  Serial.print("Connecting to WiFi ");
  Serial.print(ssid);
  Serial.println("...");
  
  WiFi.begin(ssid, password);
  int attempts = 0;
  while(WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nFailed to connect to WiFi");
  }
  
  // API Endpoints
  server.on("/trigger-recording", HTTP_GET, handleRecording);
  server.on("/test-connection", HTTP_GET, []() {
    server.send(200, "text/plain", "Testing connection...");
    testConnection();
  });
  server.on("/", HTTP_GET, []() {
    server.send(200, "text/plain", "Baby Cry Detection System\nEndpoints:\n/trigger-recording - Start a recording\n/test-connection - Test AI server connection");
  });
  
  server.begin();
  Serial.println("HTTP server started");
  
  // Test connection to AI server
  Serial.println("Testing connection to AI server...");
  testConnection();
}

void loop() {
  server.handleClient();
  monitorAudio();
}

void processRecording() {
  if (debugMode) Serial.println("Processing recording...");
  
  // Send to AI Model
  String response = sendToAIServer(audioBuffer, BUFFER_SIZE);
  
  if (response.length() > 0) {
    DynamicJsonDocument doc(256);
    DeserializationError error = deserializeJson(doc, response);
    
    if (error) {
      if (debugMode) {
        Serial.print("JSON parsing failed: ");
        Serial.println(error.c_str());
      }
    } else {
      String reason = doc["reason"].as<String>();
      storeAndNotify(reason);
    }
  }
}

void monitorAudio() {
  static unsigned long lastTrigger = 0;
  static bool isRecording = false;
  static size_t recordIndex = 0;
  static int silenceCounter = 0;
  
  // Simple VAD (Voice Activity Detection)
  int16_t sample = 0;
  static float avgVolume = 0;
  
  // Read audio from I2S
  size_t bytesRead = i2sIn.read(&sample, sizeof(int16_t));
  
  if(bytesRead == sizeof(int16_t)) {
    float instantVolume = abs(sample);
    avgVolume = 0.95 * avgVolume + 0.05 * instantVolume;
    
    // Start recording if volume threshold crossed
    if(!isRecording && instantVolume > avgVolume * 3.0 && 
       millis() - lastTrigger > 10000) {
      isRecording = true;
      recordIndex = 0;
      silenceCounter = 0;
      
      if (debugMode) {
        Serial.println("Sound detected! Recording started");
        Serial.print("Volume level: ");
        Serial.println(instantVolume);
      }
    }
    
    if(isRecording) {
      audioBuffer[recordIndex++] = sample;
      
      // Debugging info
      if (debugMode && recordIndex % 8000 == 0) {
        Serial.print("Recording: ");
        Serial.print(recordIndex / (float)SAMPLE_RATE);
        Serial.println(" seconds");
      }
      
      if(recordIndex >= BUFFER_SIZE) {
        if (debugMode) Serial.println("Buffer full, processing recording");
        processRecording();
        isRecording = false;
        lastTrigger = millis();
      }
    }
  }
}