"""
09_live_test.py
Production-grade live microphone keyword spotter with:
1. Energy Gate (VAD): Rejects ambient room silence / thermal mic hiss without amplifying noise
2. Safe Normalization: Avoids blowing up quiet background hiss by 500x
3. INT8 TinyML TFLite Inference: Ultra-fast sub-millisecond execution
4. Multi-frame confirmation: Requires 2 consecutive positive detections to eliminate transient false alarms
"""
import time
import queue
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V4_PATH = ROOT / "models" / "bankai_dscnn_v4_int8.tflite"
V3_PATH = ROOT / "models" / "bankai_dscnn_v3_int8.tflite"
MODEL_PATH = V4_PATH if V4_PATH.exists() else V3_PATH

SR = 16000
WINDOW_SAMPLES = 16000
HOP_SAMPLES = 4000  # 250 ms hop -> 4 inferences per sec

# Voice Activity Detection (VAD) energy thresholds
SILENCE_RMS_THRESHOLD = 0.006
MIN_PEAK_THRESHOLD = 0.025

# Detection trigger settings
THRESHOLD = 0.82
REQUIRED_CONSECUTIVE_FRAMES = 2
COOLDOWN_SEC = 1.5

audio_queue = queue.Queue()
audio_buffer = np.zeros(WINDOW_SAMPLES, dtype=np.float32)

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_queue.put(indata[:, 0].copy())

def make_logmel_feature(audio):
    audio = audio.astype(np.float32)
    peak = np.max(np.abs(audio))
    
    # Safe normalization: only scale if actual audio/speech exists
    if peak > MIN_PEAK_THRESHOLD:
        audio = (audio / peak) * 0.92
        
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=512,
        hop_length=160,
        n_mels=40,
        power=2.0
    )
    feature = librosa.power_to_db(mel, ref=np.max)
    
    # Exactly 101 frames
    if feature.shape[1] < 101:
        feature = np.pad(feature, ((0, 0), (0, 101 - feature.shape[1])), mode="constant")
    elif feature.shape[1] > 101:
        start = (feature.shape[1] - 101) // 2
        feature = feature[:, start:start + 101]
        
    feature = (feature - feature.mean()) / (feature.std() + 1e-6)
    return feature[np.newaxis, :, :, np.newaxis].astype(np.float32)

def main():
    if not MODEL_PATH.exists():
        print(f"Error: Model file not found at {MODEL_PATH}")
        return
        
    print(f"Loading INT8 TFLite model: {MODEL_PATH}")
    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    
    in_details = interpreter.get_input_details()[0]
    out_details = interpreter.get_output_details()[0]
    
    in_scale, in_zero = in_details["quantization"]
    out_scale, out_zero = out_details["quantization"]
    
    print("\n==============================================")
    print("      ISRO_GEMINI BANKAI LIVE TEST (INT8)")
    print("==============================================")
    print(f"Model: {MODEL_PATH.name} ({MODEL_PATH.stat().st_size / 1024:.2f} KB)")
    print(f"Features: Energy Gate (VAD) + 2-Frame Confirmation")
    print("Speak normally or say 'BANKAI' to activate.")
    print("Press Ctrl+C to stop.")
    print("----------------------------------------------\n")
    
    last_detection = 0.0
    consecutive_detections = 0
    
    try:
        with sd.InputStream(
            samplerate=SR,
            channels=1,
            dtype="float32",
            blocksize=HOP_SAMPLES,
            callback=audio_callback
        ):
            while True:
                new_audio = audio_queue.get()
                audio_buffer[:-len(new_audio)] = audio_buffer[len(new_audio):]
                audio_buffer[-len(new_audio):] = new_audio
                
                # Check RMS energy for ambient room silence
                current_rms = float(np.sqrt(np.mean(audio_buffer**2)))
                current_peak = float(np.max(np.abs(audio_buffer)))
                
                # Energy Gate: If room is silent, directly assign NOISE without false-triggering
                if current_rms < SILENCE_RMS_THRESHOLD or current_peak < MIN_PEAK_THRESHOLD:
                    bankai_score = 0.001
                    unknown_score = 0.001
                    noise_score = 0.998
                    t_inference_ms = 0.1
                    consecutive_detections = 0
                else:
                    feat = make_logmel_feature(audio_buffer)
                    
                    # Quantize input: float -> int8
                    quant_in = np.round(feat / in_scale + in_zero)
                    quant_in = np.clip(quant_in, -128, 127).astype(np.int8)
                    
                    t_start = time.perf_counter()
                    interpreter.set_tensor(in_details["index"], quant_in)
                    interpreter.invoke()
                    out_int8 = interpreter.get_tensor(out_details["index"])
                    t_inference_ms = (time.perf_counter() - t_start) * 1000.0
                    
                    # Dequantize: int8 -> float
                    out_float = (out_int8.astype(np.float32) - out_zero) * out_scale
                    probs = out_float[0]
                    total = np.sum(probs)
                    if total > 0:
                        probs = probs / total
                        
                    bankai_score = float(probs[0])
                    unknown_score = float(probs[1])
                    noise_score = float(probs[2])
                    
                    if bankai_score >= THRESHOLD:
                        consecutive_detections += 1
                    else:
                        consecutive_detections = 0
                        
                now = time.time()
                if consecutive_detections >= REQUIRED_CONSECUTIVE_FRAMES and (now - last_detection) >= COOLDOWN_SEC:
                    print(f"\n🔥 >>> BANKAI DETECTED! <<< [score: {bankai_score:.3f}, latency: {t_inference_ms:.1f}ms] 🔥\n")
                    last_detection = now
                    consecutive_detections = 0
                else:
                    bar = "█" * int(bankai_score * 20)
                    status_tag = "[SILENCE]" if current_rms < SILENCE_RMS_THRESHOLD else "[ACTIVE] "
                    print(f"{status_tag} BANKAI: {bankai_score:.3f} | UNK: {unknown_score:.3f} | NOISE: {noise_score:.3f} | [{bar:<20}] {t_inference_ms:4.1f}ms (rms={current_rms:.4f})", end="\r")
                    
    except KeyboardInterrupt:
        print("\n\nLive test stopped successfully.")

if __name__ == "__main__":
    main()
