"""
04_extract_features.py
Extracts 40 x 101 Log-Mel spectrograms for all WAV files in:
  dataset/bankai
  dataset/unknown
  dataset/noise

Outputs .npy feature matrices into:
  features/bankai
  features/unknown
  features/noise
"""
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import csv
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
FEATURES = ROOT / "features"

SR = 16000
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 40
TARGET_SAMPLES = SR
TARGET_FRAMES = 101

CLASSES = ["bankai", "unknown", "noise"]

def compute_logmel(audio):
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak
        
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )
    
    log_mel = librosa.power_to_db(mel, ref=np.max)
    
    if log_mel.shape[1] < TARGET_FRAMES:
        pad_width = TARGET_FRAMES - log_mel.shape[1]
        log_mel = np.pad(
            log_mel,
            ((0, 0), (0, pad_width)),
            mode="constant",
            constant_values=log_mel.min()
        )
    elif log_mel.shape[1] > TARGET_FRAMES:
        start = (log_mel.shape[1] - TARGET_FRAMES) // 2
        log_mel = log_mel[:, start:start + TARGET_FRAMES]
        
    mean = np.mean(log_mel)
    std = np.std(log_mel)
    log_mel = (log_mel - mean) / (std + 1e-6)
    
    return log_mel.astype(np.float32)

def process_file(item):
    wav_path, out_path = item
    try:
        if out_path.exists():
            return True, wav_path.name
        audio, sr_in = sf.read(wav_path, dtype="float32")
        if len(audio) != TARGET_SAMPLES:
            if len(audio) < TARGET_SAMPLES:
                out = np.zeros(TARGET_SAMPLES, dtype=np.float32)
                out[:len(audio)] = audio
                audio = out
            else:
                start = (len(audio) - TARGET_SAMPLES) // 2
                audio = audio[start:start + TARGET_SAMPLES]
        feature = compute_logmel(audio)
        np.save(out_path, feature)
        return True, wav_path.name
    except Exception as e:
        return False, f"{wav_path.name}: {e}"

def extract_all():
    print("========================================")
    print("LOG-MEL FEATURE EXTRACTION (40 x 101)")
    print("========================================")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    for cls in CLASSES:
        in_dir = DATASET / cls
        out_dir = FEATURES / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        
        wav_files = sorted(in_dir.glob("*.wav"))
        print(f"\nProcessing class '{cls}': {len(wav_files)} WAV files...")
        
        tasks = []
        for p in wav_files:
            out_p = out_dir / f"{p.stem}.npy"
            tasks.append((p, out_p))
            
        success = 0
        failed = 0
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_file, t) for t in tasks]
            for i, fut in enumerate(as_completed(futures), start=1):
                ok, msg = fut.result()
                if ok:
                    success += 1
                else:
                    failed += 1
                    print("  Failed:", msg)
                if i % 300 == 0 or i == len(tasks):
                    print(f"  [{i}/{len(tasks)}] extracted", end="\r")
                    
        print(f"\n  Finished '{cls}': {success} success, {failed} failed.")

if __name__ == "__main__":
    extract_all()

