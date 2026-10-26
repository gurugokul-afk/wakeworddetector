# Edge Hardware Benchmark & RAM Budget Analysis (ISRO Problem 26172)

## Target Platform
- **Microcontroller**: ESP32-S3 (Xtensa Dual-Core 32-bit LX7 @ 240 MHz)
- **Audio Input**: I2S MEMS Microphone (INMP441)
- **Audio Spec**: 16,000 Hz, 16-bit PCM, Mono
- **Inference Engine**: TensorFlow Lite Micro (TFLM)
- **Quantization**: Full INT8 (integer weights, activations, inputs, outputs)

---

## 1. Memory Budget Verification (<256 KB ISRO Requirement)

| Memory Component | Allocation | Share of RAM | Status |
| :--- | :---: | :---: | :---: |
| **Tensor Arena (TFLM Activations)** | 40.00 KB | 15.6% | PASS |
| **Raw Audio Ring Buffer (1.0 sec @ 16 kHz)** | 31.25 KB | 12.2% | PASS |
| **Log-Mel Spectrogram Buffer (40x101)** | 15.78 KB | 6.2% | PASS |
| **I2S DMA & FreeRTOS Stack/Heap** | 48.00 KB | 18.8% | PASS |
| **Application & Networking Runtime** | 24.00 KB | 9.4% | PASS |
| **TOTAL WORKING RAM CONSUMPTION** | **159.03 KB** | **62.1%** | **VERIFIED (<256 KB)** |
| **SAFETY HEADROOM / FREE SRAM** | **96.97 KB** | **37.9%** | **SAFE MARGIN** |

> [!NOTE]
> The total working RAM of **159.0 KB** is strictly below the **256 KB** ceiling, satisfying the primary ISRO hardware competition constraint with over **97.0 KB** of safety margin.

---

## 2. Flash Memory & Model Footprint

| Metric | DS-CNN v3 INT8 | ESP32-S3 Flash Budget | Margin |
| :--- | :---: | :---: | :---: |
| **TFLite Model Size** | **30.69 KB** | ~4,096 - 8,192 KB Flash | **>99% Free** |
| **C Byte Array Header Size** | **30.69 KB** | Embedded in `.rodata` | Fits easily in SPI Flash |
