// LoRa Transmitter – Standalone with ESP-NOW Frequency Hop Receiver
// Frequency: 433MHz baseline | Hops wirelessly when Receiver commands via 2.4GHz
// Power: 20dBm | SF8 baseline
// NO USB connection to laptop needed — powered by any 5V supply
// Press BOOT button (GPIO 0) to toggle simulated jamming (packet drops)

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
#define LED_PIN    2    // Built-in LED
#define BOOT_BTN   0    // BOOT button = GPIO 0

// ── LoRa Parameters ──
long  frequency       = 433000000;   // 433.00 MHz baseline
int   spreadingFactor = 8;
int   txPower         = 20;
int   packetCount     = 0;
bool  jammingON       = false;

// ── Button debounce ──
bool  lastBtnState    = HIGH;
unsigned long lastDebounce = 0;
#define DEBOUNCE_MS   300

// ── Flag: set by ESP-NOW callback, handled in loop() ──
volatile bool  freqUpdatePending = false;
volatile long  pendingFreq       = 0;
volatile bool  resetPending      = false;

// ── ESP-NOW receive callback (runs in WiFi task context) ──
void onDataRecv(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
  // Convert to String
  char buf[251];
  int  copyLen = (len < 250) ? len : 250;
  memcpy(buf, data, copyLen);
  buf[copyLen] = '\0';
  String cmd = String(buf);
  cmd.trim();

  if (cmd.startsWith("FREQ:")) {
    long newFreq = atol(cmd.c_str() + 5);
    if (newFreq >= 430000000 && newFreq <= 440000000) {
      pendingFreq       = newFreq;
      freqUpdatePending = true;
    }
  } else if (cmd == "RESET") {
    resetPending = true;
  }
}

void checkBootButton() {
  bool currentState = digitalRead(BOOT_BTN);
  if (lastBtnState == HIGH && currentState == LOW) {
    unsigned long now = millis();
    if (now - lastDebounce > DEBOUNCE_MS) {
      lastDebounce = now;
      jammingON = !jammingON;
      digitalWrite(LED_PIN, jammingON ? HIGH : LOW);
    }
  }
  lastBtnState = currentState;
}

// Triple-blink LED to confirm a frequency hop
void confirmBlink() {
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_PIN, HIGH); delay(80);
    digitalWrite(LED_PIN, LOW);  delay(80);
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN,  OUTPUT);
  pinMode(BOOT_BTN, INPUT_PULLUP);
  digitalWrite(LED_PIN, LOW);

  // ── Initialize WiFi in STA mode for ESP-NOW ──
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();

  // ── Initialize ESP-NOW ──
  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW init FAILED");
  }
  esp_now_register_recv_cb(onDataRecv);

  // ── Initialize LoRa ──
  SPI.begin(LORA_SCK, LORA_MISO, LORA_MOSI, LORA_SS);
  LoRa.setPins(LORA_SS, LORA_RST, LORA_DIO0);
  if (!LoRa.begin(frequency)) {
    Serial.println("LoRa init FAILED");
    while (true);
  }
  LoRa.setSpreadingFactor(spreadingFactor);
  LoRa.setTxPower(txPower);

  Serial.println("TX READY | FREQ:" + String(frequency) + " | ESP-NOW: LISTENING");
}

void loop() {
  checkBootButton();

  // ── Handle pending ESP-NOW frequency hop (safe in loop context) ──
  if (freqUpdatePending) {
    freqUpdatePending = false;
    frequency = pendingFreq;
    LoRa.setFrequency(frequency);
    Serial.println("HOP -> FREQ:" + String(frequency));
    confirmBlink();
  }

  if (resetPending) {
    resetPending    = false;
    frequency       = 433000000;
    spreadingFactor = 8;
    txPower         = 20;
    jammingON       = false;
    LoRa.setFrequency(frequency);
    LoRa.setSpreadingFactor(spreadingFactor);
    LoRa.setTxPower(txPower);
    digitalWrite(LED_PIN, LOW);
    Serial.println("RESET -> FREQ:" + String(frequency));
    confirmBlink();
  }

  // ── Transmit LoRa Packet ──
  packetCount++;

  if (jammingON) {
    // Drop 80% of packets when jamming is simulated
    if (random(100) < 80) {
      delay(1000);
      return;
    } else {
      LoRa.beginPacket();
      LoRa.print("PKT:" + String(packetCount) + "|JAM:1|FREQ:" + String(frequency));
      LoRa.endPacket();
    }
  } else {
    // Normal transmission
    LoRa.beginPacket();
    LoRa.print("PKT:" + String(packetCount) + "|JAM:0|FREQ:" + String(frequency));
    LoRa.endPacket();

    digitalWrite(LED_PIN, HIGH);
    delay(40);
    digitalWrite(LED_PIN, LOW);
  }

  delay(1000);
}
