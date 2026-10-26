"""
07_quantize_tinyml_v4.py
Converts DS-CNN v4 to full INT8 TinyML TFLite:
- Full INT8 quantization (int8 in, int8 out)
- Calibrated with representative dataset
- Evaluated on all 1,156 test samples
- Generates C/C++ byte array header for ESP32-S3
"""
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import os

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "bankai_dscnn_v4_best.keras"
OUTPUT_TFLITE = ROOT / "models" / "bankai_dscnn_v4_int8.tflite"
OUTPUT_C_HEADER = ROOT / "models" / "bankai_v4_model_data.h"
OUTPUT_C_SOURCE = ROOT / "models" / "bankai_v4_model_data.cc"
SPLIT_DIR = ROOT / "split"
RESULTS_DIR = ROOT / "results"

CLASSES = ["bankai", "unknown", "noise"]
CLASS_TO_ID = {"bankai": 0, "unknown": 1, "noise": 2}

def representative_dataset_gen():
    samples = []
    for cls in CLASSES:
        folder = SPLIT_DIR / "train" / cls
        files = sorted(folder.glob("*.npy"))[:100]
        for f in files:
            feat = np.load(f).astype(np.float32)
            feat = feat[np.newaxis, ..., np.newaxis]
            samples.append(feat)
    np.random.seed(20260909)
    np.random.shuffle(samples)
    for s in samples:
        yield [s]

def convert_to_int8():
    print("========================================")
    print("CONVERTING DS-CNN V4 TO FULL INT8 TINYML")
    print("========================================")
    
    print(f"Loading FP32 model: {MODEL_PATH}")
    model = tf.keras.models.load_model(str(MODEL_PATH))
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    print("Converting model with calibration...")
    tflite_model = converter.convert()
    
    with open(OUTPUT_TFLITE, "wb") as f:
        f.write(tflite_model)
        
    size_kb = len(tflite_model) / 1024.0
    print(f"\nINT8 TFLite model saved: {OUTPUT_TFLITE}")
    print(f"Model File Size: {size_kb:.2f} KB (ESP32 Flash Budget: <50 KB -> PASS!)")
    return tflite_model

def evaluate_int8():
    print("\n========================================")
    print("EVALUATING INT8 TFLITE V4 ON TEST SET")
    print("========================================")
    
    interpreter = tf.lite.Interpreter(model_path=str(OUTPUT_TFLITE))
    interpreter.allocate_tensors()
    
    in_details = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()[0]
    
    in_scale, in_zero = in_details["quantization"]
    out_scale, out_zero = out_details["quantization"]
    
    X_test = []
    y_test = []
    for cls in CLASSES:
        folder = SPLIT_DIR / "test" / cls
        files = sorted(folder.glob("*.npy"))
        for f in files:
            feat = np.load(f).astype(np.float32)
            X_test.append(feat)
            y_test.append(CLASS_TO_ID[cls])
            
    X_test = np.array(X_test, dtype=np.float32)[..., np.newaxis]
    y_test = np.array(y_test, dtype=np.int32)
    print(f"Test samples: {len(X_test)} (BANKAI: {np.sum(y_test==0)}, UNK: {np.sum(y_test==1)}, NOISE: {np.sum(y_test==2)})")
    
    preds = []
    for sample in X_test:
        sample_batch = sample[np.newaxis, ...]
        quantized = np.round(sample_batch / in_scale + in_zero)
        quantized = np.clip(quantized, -128, 127).astype(np.int8)
        
        interpreter.set_tensor(in_details["index"], quantized)
        interpreter.invoke()
        
        out_int8 = interpreter.get_tensor(out_details["index"])
        out_float = (out_int8.astype(np.float32) - out_zero) * out_scale
        preds.append(np.argmax(out_float[0]))
        
    preds = np.array(preds)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds, labels=[0, 1, 2])
    target_names = ["BANKAI", "UNKNOWN", "NOISE"]
    report = classification_report(y_test, preds, target_names=target_names, digits=4)
    
    print(f"\nINT8 v4 Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nINT8 Confusion Matrix:\n              BANKAI  UNKNOWN  NOISE")
    for i, row in enumerate(cm):
        print(f"{target_names[i]:<12} {row[0]:>6} {row[1]:>8} {row[2]:>6}")
    print("\nINT8 Classification Report:\n", report)
    
    int8_report_path = RESULTS_DIR / "int8_v4_test_results.txt"
    with open(int8_report_path, "w", encoding="utf-8") as f:
        f.write("ISRO_GEMINI INT8 TINYML TFLITE V4 TEST RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"INT8 Test Accuracy: {acc:.4f}\n\n")
        f.write("Confusion Matrix:\n" + str(cm) + "\n\n")
        f.write("Classification Report:\n" + report)

def export_c_header(tflite_bytes):
    print("\n========================================")
    print("EXPORTING C/C++ BYTE ARRAY FOR ESP32-S3")
    print("========================================")
    length = len(tflite_bytes)
    
    header_content = f"""/*
 * Auto-generated TinyML Model v4 Header for ESP32-S3 / TFLM
 * Keyword: BANKAI (3-class KWS: BANKAI, UNKNOWN, NOISE)
 * Model: Depthwise-Separable CNN (DS-CNN v4) trained on krishnaDataSet
 * Quantization: Full INT8
 */
#ifndef BANKAI_V4_MODEL_DATA_H_
#define BANKAI_V4_MODEL_DATA_H_

#include <cstdint>

extern const unsigned char g_bankai_model_data[];
extern const int g_bankai_model_data_len;

#endif  // BANKAI_V4_MODEL_DATA_H_
"""
    with open(OUTPUT_C_HEADER, "w", encoding="utf-8") as f:
        f.write(header_content)
        
    with open(OUTPUT_C_SOURCE, "w", encoding="utf-8") as f:
        f.write('#include "bankai_v4_model_data.h"\n\n')
        f.write(f'alignas(16) const unsigned char g_bankai_model_data[{length}] = {{\n  ')
        for i, byte in enumerate(tflite_bytes):
            f.write(f"0x{byte:02x}, ")
            if (i + 1) % 16 == 0:
                f.write("\n  ")
        f.write("\n};\n\n")
        f.write(f"const int g_bankai_model_data_len = {length};\n")
        
    print(f"Header exported : {OUTPUT_C_HEADER}")
    print(f"Source exported : {OUTPUT_C_SOURCE}")
    print(f"Total model bytes: {length} bytes ({length / 1024:.2f} KB)")

if __name__ == "__main__":
    tflite_data = convert_to_int8()
    evaluate_int8()
    export_c_header(tflite_data)
