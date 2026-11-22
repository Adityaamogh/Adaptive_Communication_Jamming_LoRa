// LoRa Receiver – AI-Driven Frequency Hopping via ESP-NOW OOB Control
// Frequency: 433MHz baseline | Hops to 433.75MHz on jam detection
// Pins: SCK=18, MISO=19, MOSI=23, NSS=5, RST=14, DIO0=26
// ESP-NOW: Broadcasts freq-hop commands to Transmitter over 2.4GHz WiFi

#include <SPI.h>
#include <LoRa.h>
#include <WiFi.h>
#include <esp_now.h>

#define LORA_SCK   18
#define LORA_MISO  19
#define LORA_MOSI  23
#define LORA_SS    5
#define LORA_RST   14
#define LORA_DIO0  26
#define LED_PIN    2
#define BOOT_BTN   0    // Physical BOOT button on this ESP32 (GPIO 0)

// ── LoRa Parameters ──
long  frequency       = 433000000;   // 433.00 MHz baseline
int   spreadingFactor = 8;
int   lastPacketCount = 0;
int   totalExpected   = 0;
int   totalLost       = 0;

// ── BOOT button jamming toggle ──
bool  jammingMode     = false;
bool  lastBtnState    = HIGH;
unsigned long lastDebounce = 0;
#define DEBOUNCE_MS   250

// ── ESP-NOW broadcast address (FF:FF:FF:FF:FF:FF = all peers) ──
uint8_t broadcastAddr[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

// ESP-NOW send callback (optional debug)
void onDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  // Can add Serial debug here if needed
}

// Broadcast a command string to all ESP-NOW peers (i.e. the Transmitter)
void espnowBroadcast(String msg) {
  uint8_t buf[250];
  int len = msg.length();
  if (len > 249) len = 249;
  msg.getBytes(buf, len + 1);
  esp_now_send(broadcastAddr, buf, len);
}

void checkBootButton() {
  bool currentState = digitalRead(BOOT_BTN);
  if (lastBtnState == HIGH && currentState == LOW) {
    unsigned long now = millis();
    if (now - lastDebounce > DEBOUNCE_MS) {
      lastDebounce = now;
      jammingMode = !jammingMode;
      if (jammingMode) {
        digitalWrite(LED_PIN, HIGH);
        Serial.println("JAM:1");
      } else {
        digitalWrite(LED_PIN, LOW);
        Serial.println("JAM:0");
      }
    }
  }
  lastBtnState = currentState;
}

// Handle commands from Python serial (COM4)
void applyCommand(String cmd) {
  cmd.trim();

  if (cmd.startsWith("FREQ:")) {
    // ── Frequency Hop Command ──
    long newFreq = cmd.substring(5).toInt();  // e.g. "FREQ:433750000" → 433750000
    if (newFreq >= 430000000 && newFreq <= 440000000) {
      frequency = newFreq;
      LoRa.setFrequency(frequency);
      // Broadcast the same command to Transmitter via ESP-NOW 2.4GHz
      espnowBroadcast(cmd);
      // Confirm to Python
      Serial.println("FREQACK:" + String(frequency));
      // Triple-blink LED to confirm hop
      for (int i = 0; i < 3; i++) {
        digitalWrite(LED_PIN, HIGH); delay(80);
        digitalWrite(LED_PIN, LOW);  delay(80);
      }
    }
  } else if (cmd == "RESET") {
    // ── Reset to baseline ──
    frequency       = 433000000;
    spreadingFactor = 8;
    LoRa.setFrequency(frequency);
    LoRa.setSpreadingFactor(spreadingFactor);
    // Broadcast RESET to Transmitter via ESP-NOW
    espnowBroadcast("RESET");
    Serial.println("FREQACK:" + String(frequency));
  } else if (cmd.startsWith("SF:")) {
    int sf = cmd.substring(3).toInt();
    if (sf >= 6 && sf <= 12) {
      spreadingFactor = sf;
      LoRa.setSpreadingFactor(sf);
    }
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  pinMode(BOOT_BTN, INPUT_PULLUP);

  // ── Initialize WiFi in STA mode for ESP-NOW ──
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();   // No router needed

  // ── Initialize ESP-NOW ──
  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init FAILED");
  }
  esp_now_register_send_cb(onDataSent);

  // Add broadcast peer
  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, broadcastAddr, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;
  esp_now_add_peer(&peerInfo);

  // ── Initialize LoRa ──
  SPI.begin(LORA_SCK, LORA_MISO, LORA_MOSI, LORA_SS);
  LoRa.setPins(LORA_SS, LORA_RST, LORA_DIO0);
  if (!LoRa.begin(frequency)) {
    Serial.println("LoRa init FAILED");
    while (true);
  }
  LoRa.setSpreadingFactor(spreadingFactor);

  Serial.println("RX READY | FREQ:" + String(frequency) + " | ESP-NOW: ON");
}

void loop() {
  checkBootButton();

  // ── Read commands from Python (COM4 serial) ──
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    applyCommand(cmd);
  }

  // ── Receive LoRa packets ──
  int packetSize = LoRa.parsePacket();
  if (packetSize) {
    String payload = "";
    while (LoRa.available()) {
      payload += (char)LoRa.read();
    }

    int pktNum = 0;
    if (payload.startsWith("PKT:")) {
      String numPart = payload.substring(4);
      int sep = numPart.indexOf('|');
      if (sep != -1) numPart = numPart.substring(0, sep);
      pktNum = numPart.toInt();
    }

    // Check if transmitter sent JAM flag over LoRa RF
    if (payload.indexOf("JAM:1") != -1) {
      Serial.println("JAM:1");
    } else if (payload.indexOf("JAM:0") != -1) {
      Serial.println("JAM:0");
    }

    int   rssi = LoRa.packetRssi();
    float snr  = LoRa.packetSnr();

    if (lastPacketCount == 0) {
      lastPacketCount = pktNum;
      totalExpected   = 1;
    } else {
      int gap = pktNum - lastPacketCount;
      if (gap < 0) {
        lastPacketCount = pktNum;
        totalExpected   = 1;
        totalLost       = 0;
      } else if (gap > 0) {
        totalExpected  += gap;
        totalLost      += (gap - 1);
        lastPacketCount = pktNum;
      }
    }

    float lossRate = (totalExpected > 0)
                     ? (float(totalLost) / float(totalExpected)) * 100.0
                     : 0.0;

    // Output block — Python parses these fields
    Serial.println("---");
    Serial.println("PKT:"  + String(pktNum));
    Serial.println("RSSI:" + String(rssi) + " dBm");
    Serial.println("SNR:"  + String(snr, 1) + " dB");
    Serial.println("LOSS:" + String(lossRate, 1) + " %");
    Serial.println("FREQ:" + String(frequency));
    Serial.println("---");

    // Quick blink on receive
    digitalWrite(LED_PIN, HIGH);
    delay(40);
    digitalWrite(LED_PIN, LOW);
  }
}
