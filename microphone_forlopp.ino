#include <WiFi.h>
#include <HTTPClient.h>
#include <Arduino.h>

const int micPin = 34;           // Microphone analog pin
const int sampleRate = 8000;      // 8kHz sample rate
const int threshold = 500;        // Amplitude threshold
const int recordingDuration = 5;  // 5-second clips
const int totalSamples = sampleRate * recordingDuration;

// WiFi credentials
const char* ssid = "SmarTone_HBB_4022";
const char* password = "5W5AABFB25";

// FastAPI server
const char* serverUrl = "http://127.0.0.1:8000/upload";

int16_t audioBuffer[totalSamples];
bool isRecording = false;
unsigned long recordingStartTime = 0;
int sampleCount = 0;

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);
  
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
}

void loop() {
  int sample = analogRead(micPin) - 2048;  // Center around 0
  
  // Trigger recording if threshold crossed and not already recording
  if (abs(sample) > threshold && !isRecording) {
    isRecording = true;
    sampleCount = 0;
    recordingStartTime = millis();
    Serial.println("Threshold exceeded - starting recording");
  }

  // If recording, collect samples
  if (isRecording && sampleCount < totalSamples) {
    audioBuffer[sampleCount] = sample;
    sampleCount++;
    
    // If buffer full, send to server
    if (sampleCount >= totalSamples) {
      isRecording = false;
      Serial.println("Recording complete - sending to server");
      sendAudioToServer();
    }
  }

  // Fixed delay to maintain sample rate
  delayMicroseconds(1000000 / sampleRate);
}

void sendAudioToServer() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected");
    return;
  }

  HTTPClient http;
  http.begin(serverUrl);
  
  // Create WAV header
  byte wavHeader[44];
  createWavHeader(wavHeader, totalSamples);

  // Combine header and audio data
  uint8_t* postData = (uint8_t*)malloc(44 + totalSamples * 2);
  memcpy(postData, wavHeader, 44);
  memcpy(postData + 44, audioBuffer, totalSamples * 2);

  // Send POST request
  int httpCode = http.POST(postData, 44 + totalSamples * 2);
  
  if (httpCode > 0) {
    Serial.printf("Upload successful, response: %d\n", httpCode);
  } else {
    Serial.printf("Upload failed, error: %s\n", http.errorToString(httpCode).c_str());
  }

  http.end();
  free(postData);
}

void createWavHeader(byte* header, int dataSize) {
  // RIFF header
  header[0] = 'R'; header[1] = 'I'; header[2] = 'F'; header[3] = 'F';
  unsigned int fileSize = dataSize * 2 + 36;
  header[4] = fileSize & 0xFF;
  header[5] = (fileSize >> 8) & 0xFF;
  header[6] = (fileSize >> 16) & 0xFF;
  header[7] = (fileSize >> 24) & 0xFF;
  header[8] = 'W'; header[9] = 'A'; header[10] = 'V'; header[11] = 'E';
  
  // fmt chunk
  header[12] = 'f'; header[13] = 'm'; header[14] = 't'; header[15] = ' ';
  header[16] = 16; header[17] = 0; header[18] = 0; header[19] = 0;
  header[20] = 1; header[21] = 0;  // PCM format
  header[22] = 1; header[23] = 0;  // Mono
  header[24] = 0x40; header[25] = 0x1F;  // 8000 Hz
  header[26] = 0x00; header[27] = 0x00;
  header[28] = 0x80; header[29] = 0x3E;  // Byte rate
  header[30] = 0x00; header[31] = 0x00;
  header[32] = 2; header[33] = 0;       // Block align
  header[34] = 16; header[35] = 0;      // Bits per sample
  
  // data chunk
  header[36] = 'd'; header[37] = 'a'; header[38] = 't'; header[39] = 'a';
  unsigned int chunkSize = dataSize * 2;
  header[40] = chunkSize & 0xFF;
  header[41] = (chunkSize >> 8) & 0xFF;
  header[42] = (chunkSize >> 16) & 0xFF;
  header[43] = (chunkSize >> 24) & 0xFF;
}