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
const char* aiServer = "10.89.195.233:5000";  // Add the port number
const char* predictEndpoint = "/analyze-cry";
// Comment these out for now until you have them set up
// const char* watchyIP = "192.168.x.x";
// const char* appServer = "192.168.x.x";

ESP8266WebServer server(80);
int16_t audioBuffer[BUFFER_SIZE];
bool debugMode = true; // Enable debug messages

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
    if (I2S.read(&audioBuffer[samples], sizeof(int16_t))) {
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
  
  // For now, just log it
  // Uncomment these when you have watchy and app servers set up
  /*
  // Notify Watchy
  if (watchyIP && strlen(watchyIP) > 0) {
    WiFiClient watchyClient;
    HTTPClient watchyHttp;
    watchyHttp.begin(watchyClient, String("http://") + watchyIP + "/vibrate");
    watchyHttp.POST("");
    watchyHttp.end();
  }
  
  // Notify App
  if (appServer && strlen(appServer) > 0) {
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
  */
}

void setup() {
  Serial.begin(115200);
  Serial.println("\n\nBaby Cry Detection System Starting");
  
  // Init I2S for audio
  Serial.println("Initializing I2S...");
  I2S.setSckPin(D5);
  I2S.setDataPin(D7);
  if(!I2S.begin(SAMPLE_RATE, SAMPLE_BITS)) {
    Serial.println("Failed to init I2S!");
  } else {
    Serial.println("I2S initialized successfully");
  }

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