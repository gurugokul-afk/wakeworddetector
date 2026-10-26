# Head-to-Head Model Evaluation: isro2 v2 vs isrogemini v3 vs v4

## Test Dataset Support
- Total Test Samples: **1156**
- **BANKAI**: 324
- **UNKNOWN**: 632
- **NOISE**: 200

---

## 1. Key Performance Comparison

| Metric | isro2 Baseline (v2) | isrogemini v3 (INT8) | isrogemini v4 (FP32) | isrogemini v4 (INT8 TinyML) |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | **48.10%** | **98.53%** | **96.80%** | **96.80%** |
| **BANKAI Recall (TPR)** | 99.38% | 98.46% | 98.15% | 98.15% |
| **BANKAI Precision** | 35.62% | 97.85% | 98.15% | 98.76% |
| **BANKAI F1-Score** | 52.44% | 98.15% | 98.15% | 98.45% |
| **UNKNOWN Recall** | 10.13% | 98.10% | 98.58% | 99.05% |
| **UNKNOWN Precision** | 100.00% | 99.20% | 95.70% | 95.28% |
| **NOISE Recall** | 85.00% | 100.00% | 89.00% | 87.50% |
| **NOISE Precision** | 90.43% | 97.56% | 98.34% | 98.87% |
| **Model Size** | 165.34 KB | 30.69 KB | 270.31 KB | **30.69 KB** |
| **Inference Latency** | 0.66 ms | 0.29 ms | 0.79 ms | **0.26 ms** |

---

## 2. Confusion Matrices

### isro2 Baseline (v2)
```
              BANKAI  UNKNOWN  NOISE
BANKAI           322        0        2
UNKNOWN          552       64       16
NOISE             30        0      170
```

### isrogemini DS-CNN v4 (INT8 TinyML)
```
              BANKAI  UNKNOWN  NOISE
BANKAI           318        6        0
UNKNOWN            4      626        2
NOISE              0       25      175
```
