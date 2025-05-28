#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>
#include <ArduinoJson.h>
#include <ESP8266Audio.h>
#include <AudioInputI2S.h>
#include <EEPROM.h>

// Audio Config
#define SAMPLE_RATE 16000
#define SAMPLE_BITS 16
#define CLIP_DURATION 5 // seconds
#define BUFFER_SIZE (SAMPLE_RATE * CLIP_DURATION)

// Config structure for storing settings - saved in EEPROM
struct DeviceConfig {
  char wifi_ssid[32];
  char wifi_password[64];
  char server_ip[40];     // Increased size to handle IP:port format
  char watchy_ip[40];
  char app_server[40];
  int server_port;
  char device_name[32];
  bool initialized;       // Flag to check if EEPROM has been initialized
};

// Global variables
ESP8266WebServer server(80);
int16_t audioBuffer[BUFFER_SIZE];
bool debugMode = true;
AudioInputI2S i2sIn;
DeviceConfig config;
const int EEPROM_CONFIG_ADDR = 0;
const int CONFIG_VERSION = 1; // Increment when changing config structure

// Function prototypes
void loadConfig();
void saveConfig();
bool connectToWifi();
void setupConfigWebServer();
void enterSerialConfigMode();

// Load configuration from EEPROM
void loadConfig() {
  EEPROM.begin(512);
  EEPROM.get(EEPROM_CONFIG_ADDR, config);
  
  // If config looks empty or not initialized, use defaults
  if (!config.initialized) {
    if (debugMode) Serial.println("Loading default configuration");
    
    strcpy(config.wifi_ssid, "Livebox-D510");
    strcpy(config.wifi_password, "MwHUYQoKrYPVV5t4kM");
    strcpy(config.server_ip, "10.89.195.233:5000");
    strcpy(config.watchy_ip, "192.168.1.X");
    strcpy(config.app_server, "192.168.1.Y");
    config.server_port = 5000;
    strcpy(config.device_name, "BabyCryDetector");
    config.initialized = true;
    saveConfig();
  }
  
  if (debugMode) {
    Serial.println("Configuration loaded:");
    Serial.print("WiFi SSID: ");
    Serial.println(config.wifi_ssid);
    Serial.print("Server IP: ");
    Serial.println(config.server_ip);
    Serial.print("Watchy IP: ");
    Serial.println(config.watchy_ip);
    Serial.print("App Server: ");
    Serial.println(config.app_server);
    Serial.print("Device Name: ");
    Serial.println(config.device_name);
  }
  
  EEPROM.end();
}

// Save configuration to EEPROM
void saveConfig() {
  EEPROM.begin(512);
  EEPROM.put(EEPROM_CONFIG_ADDR, config);
  EEPROM.commit();
  EEPROM.end();
  if (debugMode) Serial.println("Configuration saved to EEPROM");
}

// Connect to WiFi using stored credentials
bool connectToWifi() {
  Serial.print("Connecting to WiFi ");
  Serial.print(config.wifi_ssid);
  Serial.println("...");
  
  WiFi.begin(config.wifi_ssid, config.wifi_password);
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
    return true;
  } else {
    Serial.println("\nFailed to connect to WiFi");
    return false;
  }
}

// Set up the web server for configuration
void setupConfigWebServer() {
  // Root - serve info page
  server.on("/", HTTP_GET, []() {
    String html = "<html><head>";
    html += "<title>Baby Cry Detector</title>";
    html += "<meta name='viewport' content='width=device-width, initial-scale=1'>";
    html += "<style>body{font-family:Arial;margin:20px;} .btn{background:#1E88E5;color:white;padding:10px;border:none;border-radius:4px;cursor:pointer;}";
    html += ".card{background:#f5f5f5;padding:15px;margin:10px 0;border-radius:5px;box-shadow:0 2px 4px rgba(0,0,0,0.1);}</style>";
    html += "</head><body>";
    html += "<h1>Baby Cry Detection System</h1>";
    
    html += "<div class='card'>";
    html += "<h2>Device Status</h2>";
    html += "<p>IP Address: " + WiFi.localIP().toString() + "</p>";
    html += "<p>WiFi Network: " + String(config.wifi_ssid) + "</p>";
    html += "<p>Server: " + String(config.server_ip) + "</p>";
    html += "</div>";
    
    html += "<div class='card'>";
    html += "<h2>Actions</h2>";
    html += "<p><button class='btn' onclick='location.href=\"/config\"'>Device Settings</button></p>";
    html += "<p><button class='btn' onclick='location.href=\"/trigger-recording\"'>Trigger Recording</button></p>";
    html += "<p><button class='btn' onclick='location.href=\"/test-connection\"'>Test AI Connection</button></p>";
    html += "</div>";
    html += "</body></html>";
    
    server.send(200, "text/html", html);
  });
  
  // Configuration page
  server.on("/config", HTTP_GET, []() {
    String html = "<html><head>";
    html += "<title>Device Configuration</title>";
    html += "<meta name='viewport' content='width=device-width, initial-scale=1'>";
    html += "<style>body{font-family:Arial;margin:20px;} input{width:100%;padding:8px;margin:8px 0;box-sizing:border-box;}";
    html += ".btn{background:#1E88E5;color:white;padding:10px;border:none;border-radius:4px;cursor:pointer;}</style>";
    html += "</head><body>";
    html += "<h1>Device Configuration</h1>";
    html += "<form method='post' action='/update-config'>";
    
    html += "<h2>WiFi Settings</h2>";
    html += "SSID:<br><input type='text' name='ssid' value='" + String(config.wifi_ssid) + "'><br>";
    html += "Password:<br><input type='password' name='password' value='" + String(config.wifi_password) + "'><br>";
    
    html += "<h2>Server Settings</h2>";
    html += "AI Server (IP:port):<br><input type='text' name='server_ip' value='" + String(config.server_ip) + "'><br>";
    html += "Watchy IP:<br><input type='text' name='watchy_ip' value='" + String(config.watchy_ip) + "'><br>";
    html += "App Server:<br><input type='text' name='app_server' value='" + String(config.app_server) + "'><br>";
    html += "Device Name:<br><input type='text' name='device_name' value='" + String(config.device_name) + "'><br><br>";
    
    html += "<input class='btn' type='submit' value='Save Configuration'>";
    html += "</form>";
    html += "<p><a href='/'>Back to Home</a></p>";
    html += "</body></html>";
    
    server.send(200, "text/html", html);
  });
  
  // Handle configuration update
  server.on("/update-config", HTTP_POST, []() {
    bool needsReconnect = false;
    
    if (server.hasArg("ssid") && strcmp(server.arg("ssid").c_str(), config.wifi_ssid) != 0) {
      server.arg("ssid").toCharArray(config.wifi_ssid, sizeof(config.wifi_ssid));
      needsReconnect = true;
    }
    
    if (server.hasArg("password") && server.arg("password").length() > 0 && 
        strcmp(server.arg("password").c_str(), config.wifi_password) != 0) {
      server.arg("password").toCharArray(config.wifi_password, sizeof(config.wifi_password));
      needsReconnect = true;
    }
    
    if (server.hasArg("server_ip")) {
      server.arg("server_ip").toCharArray(config.server_ip, sizeof(config.server_ip));
    }
    
    if (server.hasArg("watchy_ip")) {
      server.arg("watchy_ip").toCharArray(config.watchy_ip, sizeof(config.watchy_ip));
    }
    
    if (server.hasArg("app_server")) {
      server.arg("app_server").toCharArray(config.app_server, sizeof(config.app_server));
    }
    
    if (server.hasArg("device_name")) {
      server.arg("device_name").toCharArray(config.device_name, sizeof(config.device_name));
    }
    
    // Save the configuration
    saveConfig();
    
    String html = "<html><head>";
    html += "<title>Configuration Updated</title>";
    html += "<meta name='viewport' content='width=device-width, initial-scale=1'>";
    html += "<style>body{font-family:Arial;margin:20px;} .success{color:green;background:#e8f5e9;padding:15px;border-radius:4px;}</style>";
    html += "<meta http-equiv='refresh' content='3;url=/'>";
    html += "</head><body>";
    html += "<h1>Configuration Updated</h1>";
    html += "<div class='success'>Settings have been saved successfully!</div>";
    html += "<p>You will be redirected to the home page in 3 seconds...</p>";
    html += "<p><a href='/'>Return to home now</a></p>";
    html += "</body></html>";
    
    server.send(200, "text/html", html);
    
    if (needsReconnect) {
      // Schedule a reconnection after the response has been sent
      delay(500);
      WiFi.disconnect();
      connectToWifi();
    }
  });
}

// Serial configuration mode - activated by holding GPIO0 at boot
void enterSerialConfigMode() {
  Serial.println("\n==== Configuration Mode ====");
  Serial.println("Enter values for each setting or press Enter to keep current value");
  
  Serial.print("WiFi SSID [");
  Serial.print(config.wifi_ssid);
  Serial.print("]: ");
  
  // Wait for serial input
  while (!Serial.available()) delay(100);
  
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    if (input.length() > 0) {
      input.toCharArray(config.wifi_ssid, sizeof(config.wifi_ssid));
    }
  }
  
  delay(500);
  Serial.flush();
  
  Serial.print("WiFi Password: ");
  while (!Serial.available()) delay(100);
  
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    if (input.length() > 0) {
      input.toCharArray(config.wifi_password, sizeof(config.wifi_password));
    }
  }
  
  delay(500);
  Serial.flush();
  
  Serial.print("Server IP:Port [");
  Serial.print(config.server_ip);
  Serial.print("]: ");
  
  while (!Serial.available()) delay(100);
  
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    if (input.length() > 0) {
      input.toCharArray(config.server_ip, sizeof(config.server_ip));
    }
  }
  
  delay(500);
  Serial.flush();
  
  Serial.print("Watchy IP [");
  Serial.print(config.watchy_ip);
  Serial.print("]: ");
  
  while (!Serial.available()) delay(100);
  
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    if (input.length() > 0) {
      input.toCharArray(config.watchy_ip, sizeof(config.watchy_ip));
    }
  }
  
  delay(500);
  Serial.flush();
  
  Serial.print("App Server [");
  Serial.print(config.app_server);
  Serial.print("]: ");
  
  while (!Serial.available()) delay(100);
  
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    if (input.length() > 0) {
      input.toCharArray(config.app_server, sizeof(config.app_server));
    }
  }
  
  saveConfig();
  Serial.println("\nConfiguration saved! Continuing with startup...");
}

// Send audio data to AI server
String sendToAIServer(int16_t* audioData, int dataSize) {
  if (debugMode) {
    Serial.println("Sending audio data to AI server...");
    Serial.print("URL: http://");
    Serial.print(config.server_ip);
    Serial.println("/analyze-cry");
    Serial.print("Data size: ");
    Serial.println(dataSize * 2);
  }
  
  WiFiClient client;
  HTTPClient http;
  
  String url = String("http://") + config.server_ip;
  // Check if port is already included in the server IP
  if (strstr(config.server_ip, ":") == NULL) {
    url += ":" + String(config.server_port);
  }
  url += "/analyze-cry";
  
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
  
  String resultMessage;
  if (response.length() > 0) {
    resultMessage = "Test successful! Server responded.";
  } else {
    resultMessage = "Test failed. Server not responding.";
  }
  
  server.send(200, "text/plain", resultMessage);
  
  if (debugMode) {
    Serial.println("Test complete: " + resultMessage);
  }
}

void storeAndNotify(String reason) {
  if (debugMode) {
    Serial.print("Baby cry detected: ");
    Serial.println(reason);
  }
  
  // Notify Watchy
  if (strlen(config.watchy_ip) > 0 && strcmp(config.watchy_ip, "192.168.1.X") != 0) {
    WiFiClient watchyClient;
    HTTPClient watchyHttp;
    String watchyUrl = String("http://") + config.watchy_ip + "/vibrate";
    
    if (debugMode) {
      Serial.print("Notifying Watchy at: ");
      Serial.println(watchyUrl);
    }
    
    watchyHttp.begin(watchyClient, watchyUrl);
    watchyHttp.addHeader("Content-Type", "application/json");
    
    // Create JSON payload
    DynamicJsonDocument doc(128);
    doc["reason"] = reason;
    String json;
    serializeJson(doc, json);
    
    int httpCode = watchyHttp.POST(json);
    
    if (debugMode) {
      Serial.print("Watchy notification result: ");
      Serial.println(httpCode);
    }
    
    watchyHttp.end();
  }
  
  // Notify App
  if (strlen(config.app_server) > 0 && strcmp(config.app_server, "192.168.1.Y") != 0) {
    WiFiClient appClient;
    HTTPClient appHttp;
    String appUrl = String("http://") + config.app_server + "/baby-alert";
    
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

void setup() {
  Serial.begin(115200);
  delay(1000); // Allow time for serial to initialize
  Serial.println("\n\nBaby Cry Detection System Starting");
  
  // Load configuration from EEPROM
  loadConfig();
  
  // Check if button is pressed to enter config mode
  pinMode(0, INPUT_PULLUP); // GPIO0 is usually the flash/boot button
  if (digitalRead(0) == LOW) {
    Serial.println("Button pressed - entering configuration mode");
    enterSerialConfigMode();
  }
  
  // Init I2S for audio
  Serial.println("Initializing I2S...");
  if (!i2sIn.begin(SAMPLE_RATE, I2S_PHILIPS_MODE, I2S_BITS_PER_SAMPLE_16BIT, I2S_MONO)) {
    Serial.println("Failed to initialize I2S!");
  } else {
    Serial.println("I2S initialized successfully");
  }
  
  // Set I2S pins
  i2sIn.setPin(I2S_PIN_BCLK, D5);
  i2sIn.setPin(I2S_PIN_WS, D6);
  i2sIn.setPin(I2S_PIN_DATA, D7);

  // Connect WiFi
  if (!connectToWifi()) {
    // If WiFi connection fails, start AP mode for configuration
    String apName = "BabyCryDetector_" + String(ESP.getChipId(), HEX);
    Serial.print("Starting AP mode: ");
    Serial.println(apName);
    WiFi.softAP(apName.c_str());
    Serial.print("AP IP address: ");
    Serial.println(WiFi.softAPIP());
  }
  
  // Setup the configuration web server
  setupConfigWebServer();
  
  // API Endpoints for normal operation
  server.on("/trigger-recording", HTTP_GET, handleRecording);
  server.on("/test-connection", HTTP_GET, testConnection);
  
  server.begin();
  Serial.println("HTTP server started");
  
  // Test connection to AI server
  Serial.println("Testing connection to AI server...");
  testConnection();
}

void loop() {
  server.handleClient();
  
  // Only monitor audio if we're connected to WiFi
  if (WiFi.status() == WL_CONNECTED) {
    monitorAudio();
  } else {
    // If WiFi disconnects, try to reconnect periodically
    static unsigned long lastReconnectAttempt = 0;
    if (millis() - lastReconnectAttempt > 30000) {
      lastReconnectAttempt = millis();
      Serial.println("WiFi disconnected, attempting to reconnect...");
      connectToWifi();
    }
  }
}