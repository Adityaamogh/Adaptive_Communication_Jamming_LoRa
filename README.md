# LoRa AI Anti-Jamming & Adaptive Communication System


## What This Project Does

Two ESP32 development boards communicate over a 433 MHz LoRa (SX1276) radio link. A Python dashboard on the PC trains a **Random Forest classifier** on live RF telemetry and, when jamming is detected, autonomously commands both radios to hop to **433.75 MHz** — using a separate **2.4 GHz ESP-NOW channel** so the hop command itself cannot be jammed along with the LoRa link.

```
[Transmitter ESP32]  ──── 433 MHz LoRa ────▶  [Receiver ESP32] ──── USB/COM4 ────▶  [PC / Streamlit]
        ▲                                              │                                     │
        └────────────── 2.4 GHz ESP-NOW ◀─────────────┘◀──── hop / jam commands ────────────┘
```

### The two nodes

| Node | File | Connected to PC? | Role |
|------|------|-----------------|------|
| **Transmitter** | `firmware/transmitter/transmitter.ino` | ❌ Battery-powered, standalone | Sends LoRa packets every ~1 s; presses BOOT to toggle simulated jamming (80 % packet drop + reduced TX power) |
| **Receiver** | `firmware/receiver/receiver.ino` | ✅ USB → COM4 | Receives LoRa packets, computes RSSI / SNR / packet-loss, streams telemetry to Python; receives hop/jam commands from Python and relays them to the Transmitter via ESP-NOW |

---

## How Interference Is Emulated

There is no external RF jammer hardware. Jamming is **emulated in the Transmitter firmware**:

1. Press the **BOOT button (GPIO 0)** on the Transmitter → `jammingON` latches `true`.
2. While `jammingON`:
   - **80 % of packets are silently dropped** (`random(100) < 80` → early `return`).
   - The remaining 20 % are sent with the `JAM:1` flag inside the LoRa payload.
3. The Receiver sees large sequence gaps → packet-loss jumps from ~0 % to ~80 %.
4. RSSI and SNR vary because the Transmitter also reduces `txPower` while jammed.

The Receiver's BOOT button toggles a **label flag** (`jammingMode`) and echoes `JAM:1` / `JAM:0` over serial so Python knows the ground-truth label for each training packet.

---

## Where the Random Forest Runs

The model runs entirely on the **PC** (Python / scikit-learn), not on the microcontrollers.

### Input features (6 total — all computed from raw radio telemetry)

| Feature | Source | Unit |
|---------|--------|------|
| `rssi` | SX1276 silicon (`LoRa.packetRssi()`) | dBm |
| `snr` | SX1276 silicon (`LoRa.packetSnr()`) | dB |
| `loss` | Instantaneous packet-loss from sequence numbers | % |
| `rssi_var` | Variance of last 5 RSSI readings | dBm² |
| `rssi_delta` | Current RSSI − RSSI 5 packets ago | dBm |
| `snr_delta` | Current SNR − SNR 5 packets ago | dB |

### Labels

| Label | Meaning |
|-------|---------|
| `0` | Normal link (jammer OFF) |
| `1` | Jammed (jammer ON — BOOT pressed on Receiver or Transmitter) |

### Decision threshold

scikit-learn default: `predict()` returns `1` when `predict_proba()[1] ≥ 0.5`.  
The dashboard confirms jamming only after **2 consecutive** `pred == 1` packets (`HOP_CONFIRM = 2`) before issuing the hop command, to avoid single-packet false positives.

---

## How the Channel Switch Is Triggered

```
ML pred=1 (packet N)
ML pred=1 (packet N+1)   ← HOP_CONFIRM = 2 consecutive hits
         │
         ▼
Python sends "FREQ:433750000\n" to COM4
         │
         ▼
Receiver calls LoRa.setFrequency(433750000)   ← RX hops immediately
Receiver calls espnowBroadcast("FREQ:433750000")  ← 2.4 GHz OOB
         │
         ▼
Transmitter onDataRecv() sets pendingFreq = 433750000
Transmitter loop() calls LoRa.setFrequency(433750000)   ← TX hops
         │
         ▼
Link restored on 433.75 MHz — emulated jammer still on 433.00 MHz
```

The **out-of-band (OOB) ESP-NOW channel** is the key novelty: hop commands travel over 2.4 GHz WiFi and are therefore immune to 433 MHz jamming. In-band recovery (sending the hop command on the same jammed channel) is the coordination deadlock identified by Huang et al. (IEEE VTC 2019).

---

## Wiring Diagram

Both boards use the **same SPI pinout** for the RA-02 module:

| RA-02 pin | ESP32 GPIO | Note |
|-----------|-----------|------|
| SCK | 18 | SPI clock |
| MISO | 19 | SPI MISO |
| MOSI | 23 | SPI MOSI |
| NSS / CS | 5 | Chip select |
| RST | 14 | Reset |
| DIO0 | 26 | Interrupt (packet done) |
| 3.3 V | 3V3 | — |
| GND | GND | — |



## References

- Huang et al., "Jamming-Resilient LoRa," IEEE VTC 2019 — https://ieeexplore.ieee.org/document/8746374
- Di Pinto et al., IEEE WCNC 2025 — https://ieeexplore.ieee.org/document/10978351
- Grunewald et al., arXiv:2412.01609 — https://arxiv.org/abs/2412.01609
