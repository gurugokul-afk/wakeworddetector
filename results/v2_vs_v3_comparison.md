# Head-to-Head Model Evaluation: isro2 v2 vs isrogemini v3

## Test Dataset Support
- Total Test Samples: **1044**
- **BANKAI**: 261
- **UNKNOWN**: 501
- **NOISE**: 282

---

## 1. Key Performance Comparison

| Metric | isro2 Baseline (v2) | isrogemini v3 (FP32) | isrogemini v3 (INT8 TinyML) |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | **80.27%** | **97.99%** | **97.99%** |
| **BANKAI Recall (TPR)** | 99.23% | 97.70% | 98.47% |
| **BANKAI Precision** | 58.86% | 98.46% | 97.72% |
| **BANKAI F1-Score** | 73.89% | 98.08% | 98.09% |
| **UNKNOWN Recall** | 60.48% | 97.21% | 97.01% |
| **UNKNOWN Precision** | 100.00% | 98.58% | 98.98% |
| **NOISE Recall** | 97.87% | 99.65% | 99.29% |
| **NOISE Precision** | 91.69% | 96.56% | 96.55% |
| **Model Size** | 165.34 KB | 270.31 KB | **30.69 KB** |
| **Inference Latency** | 0.65 ms | 0.79 ms | **0.31 ms** |

---

## 2. Confusion Matrices

### isro2 Baseline (v2)
```
              BANKAI  UNKNOWN  NOISE
BANKAI           259        0        2
UNKNOWN          175      303       23
NOISE              6        0      276
```

### isrogemini DS-CNN v3 (FP32)
```
              BANKAI  UNKNOWN  NOISE
BANKAI           255        6        0
UNKNOWN            4      487       10
NOISE              0        1      281
```

### isrogemini DS-CNN v3 (INT8 TinyML)
```
              BANKAI  UNKNOWN  NOISE
BANKAI           257        4        0
UNKNOWN            5      486       10
NOISE              1        1      280
```
