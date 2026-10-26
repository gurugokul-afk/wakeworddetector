# BANKAI v4: Ultra-Low-Power TinyML Wake-Word Detector for ESP32-S3

[![Target Platform](https://img.shields.io/badge/Platform-ESP32--S3%20Xtensa%20LX7-red.svg)](https://www.espressif.com/en/products/socs/esp32-s3)
[![Framework](https://img.shields.io/badge/Framework-ESP--IDF%20v6.1%20%7C%20FreeRTOS-blue.svg)](https://github.com/espressif/esp-idf)
[![Inference Engine](https://img.shields.io/badge/Inference-TFLite%20Micro%20%2B%20ESP--NN-green.svg)](https://github.com/espressif/esp-tflite-micro)
[![RAM Footprint](https://img.shields.io/badge/RAM-244.5%20KB%20(%3C%20256%20KB)-brightgreen.svg)]()
[![CPU Utilization](https://img.shields.io/badge/CPU-1.94%25%20Active-brightgreen.svg)]()
[![Model Accuracy](https://img.shields.io/badge/Accuracy-100%25%20Detection-brightgreen.svg)]()

An end-to-end, ultra-low-power, low-latency **TinyML Wake-Word Detection System** deployed on the **ESP32-S3** microcontroller. The system listens for the target wake-word **"BANKAI"** while strictly rejecting non-target speech, background speech, and ambient noise.

---

## 1. Key Performance Highlights

| Metric | Target Specification | Measured On-Device Result | Margin / Status |
| :--- | :--- | :--- | :--- |
| **RAM Footprint** | **< 256 KB** (262,144 bytes) | **244.5 KB** (`libmain.a` total 251.5 KB) | **PASS** (10.6 KB Headroom) |
| **CPU Utilization** | **< 10.0%** Average Duty Cycle | **0.06%** (Silence) / **1.94%** (Conversation) | **PASS** (80.6% Margin) |
| **Model Accuracy** | High-Confidence Detection | **100.0%** Detection ("BANKAI")<br>**100.0%** Rejection ("UNKNOWN") | **PASS** |
| **Peak Latency** | Real-time response | **188.9 ms** total pipeline (56.2 ms NN, 132.7 ms DSP) | **PASS** (Real-time) |
| **Flash / ROM Usage**| 1 MB Partition Budget | **438 KB** total image (30.7 KB Model) | **PASS** (57% Free) |

---

## 2. System Architecture

The pipeline uses a **Two-Tier Hierarchical Energy Gating Architecture** to achieve sub-2% average CPU utilization:

```
[ INMP441 I2S MEMS Mic ]
          │ 16 kHz, 16-bit Mono PCM
          ▼
┌────────────────────────────────────────┐
│  Tier 1: RMS Energy Gate               │
│  Computes frame energy in 250ms chunks │
└──────────────────┬─────────────────────┘
                   │
         Energy > Threshold?
         ├── NO  ──► Sleep / Idle (0.06% CPU duty cycle)
         └── YES ──► Wake Tier 2 DSP
                   │
                   ▼
┌────────────────────────────────────────┐
│  Tier 2: DSP Spectrogram Processor     │
│  - 512-point Real FFT (rfft_512)       │
│  - 40-band Mel-Filterbank              │
│  - Log compression                     │
│  Output: 40 x 101 Spectrogram Tensor   │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│  Tier 3: TFLM DS-CNN v4 INT8 Engine    │
│  - Accelerated by ESP-NN SIMD          │
│  - Zero Flash-to-RAM leakage (.rodata) │
│  - Static Tensor Arena (132 KB)        │
└──────────────────┬─────────────────────┘
                   │
                   ▼
       [ "BANKAI" Detected? ]
         ├── YES ──► Trigger Action / LED Pulse
         └── NO  ──► Return to Listening
```

### Memory Footprint Breakdown

All allocations run strictly within the internal 240 MHz Static Data RAM (DIRAM) with **Zero PSRAM dependency**:

* `s_tensor_arena`: 132.0 KiB (TFLM working scratchpad)
* `s_audio_window`: 62.5 KiB (1.0s Ring Buffer @ 16 kHz)
* `s_i2s_raw_buffer`: 31.25 KiB (250ms I2S DMA Ping-Pong buffer)
* `AudioProcessor Spectrogram`: 15.78 KiB (40 x 101 Mel-Spectrogram float matrix)
* `AudioProcessor FFT / Window`: 3.00 KiB (512-pt Real FFT scratchpad + power bins)
* FreeRTOS & System overhead: 7.01 KiB
* **Total Active RAM**: **251.54 KiB (< 256 KiB budget)**

---

## 3. Hardware Wiring

Connect the **INMP441 I2S MEMS Microphone** to the **ESP32-S3**:

| INMP441 Pin | ESP32-S3 GPIO | Description |
| :--- | :--- | :--- |
| **VDD** | 3.3V | Power Supply (3.3V DC) |
| **GND** | GND | Ground |
| **SD** | GPIO 4 | I2S Serial Data input |
| **WS** | GPIO 5 | I2S Word Select (L/R Clock) |
| **SCK** | GPIO 6 | I2S Bit Clock (BCLK) |
| **L/R** | GND | Left Channel Selection |

---

## 4. Repository Structure

```
wakeworddetector/
├── bankai_v4_esp32/             # Production ESP-IDF firmware
│   ├── main/
│   │   ├── main.cc             # FreeRTOS task, energy gate, TFLM runner
│   │   ├── audio_processor.h   # Mel-spectrogram & FFT DSP pipeline
│   │   ├── rfft_512.h          # Fast 512-point Real FFT implementation
│   │   ├── mel_tables.h        # Precomputed 40-band triangular filter weights
│   │   ├── bankai_v4_model.cc  # INT8 quantized model data array
│   │   └── test_audio_samples.h# Pre-baked test samples for on-device testing
│   ├── CMakeLists.txt
│   ├── sdkconfig.defaults      # ESP32-S3 optimized configurations
│   ├── generate_test_audio.py  # Converts WAV to C header test arrays
│   └── test_triggers.py        # Serial automated test script
├── esp32_test/                 # Basic ESP32 functionality sanity check
├── inmp441_test/               # Hardware test for INMP441 I2S audio stream
├── models/                     # Trained FP32 & INT8 models
│   ├── bankai_dscnn_v4_int8.tflite   # Production quantized INT8 model (31 KB)
│   ├── bankai_dscnn_v4_best.keras    # Production Keras training checkpoint
│   ├── bankai_v4_model_data.cc / .h  # C++ header & source exports
│   └── bankai_dscnn_v3_*             # Previous iteration (v3) baseline
├── results/                    # Validation benchmarks, reports & confusion matrices
│   ├── classification_report_v4.txt
│   ├── confusion_matrix_v4.csv
│   ├── int8_v4_test_results.txt
│   ├── training_history_v4.csv
│   └── edge_benchmark_analysis.md
├── scripts/                    # End-to-end Python ML Pipeline
│   ├── 01_fetch_online_data.py           # Fetch Google Speech Commands dataset
│   ├── 02_generate_synthetic_noise.py    # Generate synthetic noise audio
│   ├── 03_prepare_and_augment.py         # Data augmentation (pitch, speed, reverb)
│   ├── 03c_integrate_krishna_dataset.py  # Dataset merger & balancer
│   ├── 04_extract_features.py            # 40x101 Log-Mel extraction
│   ├── 05_create_balanced_split.py       # Train / Val / Test 80-10-10 split
│   ├── 06_train_dscnn_v4.py              # Depthwise Separable CNN training
│   ├── 07_quantize_tinyml_v4.py          # INT8 full integer quantization
│   ├── 08_evaluate_comparison_v4.py      # Benchmark comparison & confusion matrix
│   ├── 09_live_test.py                   # PC microphone real-time inference
│   └── 10_benchmark_edge.py              # Edge compute & latency profiler
├── isro_bankai_v4_test_report.md         # Comprehensive verification report
├── eim_config.toml
└── .gitignore
```

---

## 5. Getting Started

### Building & Flashing Firmware (ESP-IDF v6.1)

1. Set up the ESP-IDF environment:
   ```bash
   get-idf
   ```
2. Navigate to the firmware directory:
   ```bash
   cd bankai_v4_esp32
   ```
3. Set the target to ESP32-S3:
   ```bash
   idf.py set-target esp32s3
   ```
4. Build, flash, and monitor:
   ```bash
   idf.py build flash monitor
   ```

### Running the Python ML Pipeline

1. Install dependencies:
   ```bash
   pip install tensorflow librosa soundfile numpy scipy pyserial
   ```
2. Run data preparation & feature extraction:
   ```bash
   python scripts/01_fetch_online_data.py
   python scripts/04_extract_features.py
   python scripts/05_create_balanced_split.py
   ```
3. Train and quantize the model:
   ```bash
   python scripts/06_train_dscnn_v4.py
   python scripts/07_quantize_tinyml_v4.py
   ```
4. Test live with a PC microphone:
   ```bash
   python scripts/09_live_test.py
   ```

---

## 6. Detailed Benchmark Report

For comprehensive memory maps, GCC ELF layout, duty-cycle calculations, and edge performance analysis, please see [isro_bankai_v4_test_report.md](isro_bankai_v4_test_report.md).

---

## License

This project is open-source under the [MIT License](LICENSE) (or Apache-2.0).
