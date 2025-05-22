#ifndef WATCHY_H
#define WATCHY_H

// Vibration Configuration
#define VIB_MOTOR_PIN D8
#define VIBRATION_DURATION 100   // 100ms vibration pulses
#define VIBRATION_COUNT 10       // 10 vibrations total
#define VIBRATION_INTERVAL 300   // 300ms between vibrations

#include <ESP8266WebServer.h>
#include <Arduino.h>
#include <WiFiManager.h>
#include <HTTPClient.h>
#include <NTPClient.h>
#include <WiFiUdp.h>
#include <Arduino_JSON.h>
#include <GxEPD2_BW.h>
#include <Wire.h>
#include <Fonts/FreeMonoBold9pt7b.h>
#include "DSEG7_Classic_Bold_53.h"
#include "WatchyRTC.h"
#include "BLE.h"
#include "bma.h"
#include "config.h"

typedef struct weatherData {
  int8_t temperature;
  int16_t weatherConditionCode;
  bool isMetric;
  String weatherDescription;
} weatherData;

typedef struct watchySettings {
  // Weather Settings
  String cityID;
  String weatherAPIKey;
  String weatherURL;
  String weatherUnit;
  String weatherLang;
  int8_t weatherUpdateInterval;
  // NTP Settings
  String ntpServer;
  int gmtOffset;
  int dstOffset;
  // Alert System Settings
  String aiServerIP;     // IP of the ESP8266 audio processor
  uint16_t aiServerPort; // Typically 80
} watchySettings;

class Watchy {
public:
  static WatchyRTC RTC;
  static GxEPD2_BW<GxEPD2_154_D67, GxEPD2_154_D67::HEIGHT> display;
  static ESP8266WebServer server;
  
  tmElements_t currentTime;
  watchySettings settings;

public:
  explicit Watchy(const watchySettings &s) : settings(s) {}
  void init(String datetime = "");
  void deepSleep();
  static void displayBusyCallback(const void *);
  float getBatteryVoltage();
  
  // Vibration Methods
  void vibMotor(uint8_t intervalMs = VIBRATION_DURATION, 
               uint8_t length = VIBRATION_COUNT);
  void handleVibration();
  void triggerBabyAlert();

  // Existing UI methods
  void handleButtonPress();
  void showMenu(byte menuIndex, bool partialRefresh);
  void showFastMenu(byte menuIndex);
  void showAbout();
  void showBuzz();
  void showAccelerometer();
  void showUpdateFW();
  void showSyncNTP();
  bool syncNTP();
  bool syncNTP(long gmt, int dst, String ntpServer);
  void setTime();
  void setupWifi();
  bool connectWiFi();
  weatherData getWeatherData();
  weatherData getWeatherData(String cityID, String units, String lang,
                           String url, String apiKey, uint8_t updateInterval);
  void updateFWBegin();
  void showWatchFace(bool partialRefresh);
  virtual void drawWatchFace();

private:
  void _bmaConfig();
  static void _configModeCallback(WiFiManager *myWiFiManager);
  static uint16_t _readRegister(uint8_t address, uint8_t reg, uint8_t *data,
                              uint16_t len);
  static uint16_t _writeRegister(uint8_t address, uint8_t reg, uint8_t *data,
                               uint16_t len);
};

#endif