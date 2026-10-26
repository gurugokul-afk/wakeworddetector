"""
08_evaluate_comparison_v4.py
Performs rigorous head-to-head evaluation:
1. Baseline: isro2 Model v2 (bankai_dscnn_v2.keras)
2. isrogemini Model v3 INT8 (bankai_dscnn_v3_int8.tflite)
3. isrogemini Model v4 FP32 (bankai_dscnn_v4_best.keras)
4. isrogemini Model v4 INT8 TinyML (bankai_dscnn_v4_int8.tflite)

Generates comparative markdown report and console output.
"""
from pathlib import Path
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os

ROOT = Path(__file__).resolve().parent.parent
ISRO2_MODEL_PATH = Path(r"F:\isro2\models\bankai_dscnn_v2.keras")
V3_INT8_PATH = ROOT / "models" / "bankai_dscnn_v3_int8.tflite"
V4_FP32_PATH = ROOT / "models" / "bankai_dscnn_v4_best.keras"
V4_INT8_PATH = ROOT / "models" / "bankai_dscnn_v4_int8.tflite"
TEST_DIR = ROOT / "split" / "test"
RESULTS_DIR = ROOT / "results"

CLASSES = ["bankai", "unknown", "noise"]
CLASS_TO_ID = {"bankai": 0, "unknown": 1, "noise": 2}
CLASS_NAMES = ["BANKAI", "UNKNOWN", "NOISE"]

def load_test_data():
    X = []
    y = []
    for cls in CLASSES:
        folder = TEST_DIR / cls
        files = sorted(folder.glob("*.npy"))
        for f in files:
            feat = np.load(f).astype(np.float32)
            X.append(feat)
            y.append(CLASS_TO_ID[cls])
    X = np.asarray(X, dtype=np.float32)[..., np.newaxis]
    y = np.asarray(y, dtype=np.int32)
    return X, y

def eval_keras_model(model_path, X_test, y_test, name="Model"):
    print(f"\nEvaluating Keras model: {name} ({model_path})...")
    model = tf.keras.models.load_model(str(model_path))
    
    t0 = time.perf_counter()
    probs = model.predict(X_test, batch_size=32, verbose=0)
    t1 = time.perf_counter()
    
    avg_latency_ms = ((t1 - t0) / len(X_test)) * 1000.0
    preds = np.argmax(probs, axis=1)
    
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds, labels=[0, 1, 2])
    report = classification_report(y_test, preds, target_names=CLASS_NAMES, digits=4, output_dict=True)
    size_kb = os.path.getsize(model_path) / 1024.0
    
    return {
        "name": name,
        "acc": acc,
        "cm": cm,
        "report": report,
        "size_kb": size_kb,
        "latency_ms": avg_latency_ms,
        "params": model.count_params()
    }

def eval_int8_model(tflite_path, X_test, y_test, name="Model INT8"):
    print(f"\nEvaluating INT8 TFLite model: {name} ({tflite_path})...")
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    in_details = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()[0]
    
    in_scale, in_zero = in_details["quantization"]
    out_scale, out_zero = out_details["quantization"]
    
    preds = []
    t0 = time.perf_counter()
    for sample in X_test:
        sample_batch = sample[np.newaxis, ...]
        quantized = np.round(sample_batch / in_scale + in_zero)
        quantized = np.clip(quantized, -128, 127).astype(np.int8)
        
        interpreter.set_tensor(in_details["index"], quantized)
        interpreter.invoke()
        
        out_int8 = interpreter.get_tensor(out_details["index"])
        out_float = (out_int8.astype(np.float32) - out_zero) * out_scale
        preds.append(np.argmax(out_float[0]))
    t1 = time.perf_counter()
    
    avg_latency_ms = ((t1 - t0) / len(X_test)) * 1000.0
    preds = np.array(preds)
    
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds, labels=[0, 1, 2])
    report = classification_report(y_test, preds, target_names=CLASS_NAMES, digits=4, output_dict=True)
    size_kb = os.path.getsize(tflite_path) / 1024.0
    
    return {
        "name": name,
        "acc": acc,
        "cm": cm,
        "report": report,
        "size_kb": size_kb,
        "latency_ms": avg_latency_ms,
        "params": "Quantized INT8"
    }

def generate_comparison():
    X_test, y_test = load_test_data()
    print(f"Loaded {len(X_test)} test samples:")
    for cid, cname in enumerate(CLASS_NAMES):
        print(f"  {cname:<8}: {np.sum(y_test == cid)}")
        
    res_v2 = eval_keras_model(ISRO2_MODEL_PATH, X_test, y_test, name="isro2 Baseline (v2)")
    res_v3_int8 = eval_int8_model(V3_INT8_PATH, X_test, y_test, name="isrogemini v3 (INT8)")
    res_v4_fp = eval_keras_model(V4_FP32_PATH, X_test, y_test, name="isrogemini v4 (FP32)")
    res_v4_int8 = eval_int8_model(V4_INT8_PATH, X_test, y_test, name="isrogemini v4 (INT8 TinyML)")
    
    md_content = f"""# Head-to-Head Model Evaluation: isro2 v2 vs isrogemini v3 vs v4

## Test Dataset Support
- Total Test Samples: **{len(X_test)}**
- **BANKAI**: {np.sum(y_test == 0)}
- **UNKNOWN**: {np.sum(y_test == 1)}
- **NOISE**: {np.sum(y_test == 2)}

---

## 1. Key Performance Comparison

| Metric | isro2 Baseline (v2) | isrogemini v3 (INT8) | isrogemini v4 (FP32) | isrogemini v4 (INT8 TinyML) |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Accuracy** | **{res_v2['acc']*100:.2f}%** | **{res_v3_int8['acc']*100:.2f}%** | **{res_v4_fp['acc']*100:.2f}%** | **{res_v4_int8['acc']*100:.2f}%** |
| **BANKAI Recall (TPR)** | {res_v2['report']['BANKAI']['recall']*100:.2f}% | {res_v3_int8['report']['BANKAI']['recall']*100:.2f}% | {res_v4_fp['report']['BANKAI']['recall']*100:.2f}% | {res_v4_int8['report']['BANKAI']['recall']*100:.2f}% |
| **BANKAI Precision** | {res_v2['report']['BANKAI']['precision']*100:.2f}% | {res_v3_int8['report']['BANKAI']['precision']*100:.2f}% | {res_v4_fp['report']['BANKAI']['precision']*100:.2f}% | {res_v4_int8['report']['BANKAI']['precision']*100:.2f}% |
| **BANKAI F1-Score** | {res_v2['report']['BANKAI']['f1-score']*100:.2f}% | {res_v3_int8['report']['BANKAI']['f1-score']*100:.2f}% | {res_v4_fp['report']['BANKAI']['f1-score']*100:.2f}% | {res_v4_int8['report']['BANKAI']['f1-score']*100:.2f}% |
| **UNKNOWN Recall** | {res_v2['report']['UNKNOWN']['recall']*100:.2f}% | {res_v3_int8['report']['UNKNOWN']['recall']*100:.2f}% | {res_v4_fp['report']['UNKNOWN']['recall']*100:.2f}% | {res_v4_int8['report']['UNKNOWN']['recall']*100:.2f}% |
| **UNKNOWN Precision** | {res_v2['report']['UNKNOWN']['precision']*100:.2f}% | {res_v3_int8['report']['UNKNOWN']['precision']*100:.2f}% | {res_v4_fp['report']['UNKNOWN']['precision']*100:.2f}% | {res_v4_int8['report']['UNKNOWN']['precision']*100:.2f}% |
| **NOISE Recall** | {res_v2['report']['NOISE']['recall']*100:.2f}% | {res_v3_int8['report']['NOISE']['recall']*100:.2f}% | {res_v4_fp['report']['NOISE']['recall']*100:.2f}% | {res_v4_int8['report']['NOISE']['recall']*100:.2f}% |
| **NOISE Precision** | {res_v2['report']['NOISE']['precision']*100:.2f}% | {res_v3_int8['report']['NOISE']['precision']*100:.2f}% | {res_v4_fp['report']['NOISE']['precision']*100:.2f}% | {res_v4_int8['report']['NOISE']['precision']*100:.2f}% |
| **Model Size** | {res_v2['size_kb']:.2f} KB | {res_v3_int8['size_kb']:.2f} KB | {res_v4_fp['size_kb']:.2f} KB | **{res_v4_int8['size_kb']:.2f} KB** |
| **Inference Latency** | {res_v2['latency_ms']:.2f} ms | {res_v3_int8['latency_ms']:.2f} ms | {res_v4_fp['latency_ms']:.2f} ms | **{res_v4_int8['latency_ms']:.2f} ms** |

---

## 2. Confusion Matrices

### isro2 Baseline (v2)
```
              BANKAI  UNKNOWN  NOISE
BANKAI        {res_v2['cm'][0][0]:>6}   {res_v2['cm'][0][1]:>6}   {res_v2['cm'][0][2]:>6}
UNKNOWN       {res_v2['cm'][1][0]:>6}   {res_v2['cm'][1][1]:>6}   {res_v2['cm'][1][2]:>6}
NOISE         {res_v2['cm'][2][0]:>6}   {res_v2['cm'][2][1]:>6}   {res_v2['cm'][2][2]:>6}
```

### isrogemini DS-CNN v4 (INT8 TinyML)
```
              BANKAI  UNKNOWN  NOISE
BANKAI        {res_v4_int8['cm'][0][0]:>6}   {res_v4_int8['cm'][0][1]:>6}   {res_v4_int8['cm'][0][2]:>6}
UNKNOWN       {res_v4_int8['cm'][1][0]:>6}   {res_v4_int8['cm'][1][1]:>6}   {res_v4_int8['cm'][1][2]:>6}
NOISE         {res_v4_int8['cm'][2][0]:>6}   {res_v4_int8['cm'][2][1]:>6}   {res_v4_int8['cm'][2][2]:>6}
```
"""
    out_file = RESULTS_DIR / "v2_vs_v3_vs_v4_comparison.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"\nSaved comparison report: {out_file}")
    print(md_content)

if __name__ == "__main__":
    generate_comparison()
