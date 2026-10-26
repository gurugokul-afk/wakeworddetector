"""
03b_augment_bankai.py
Generates high-accuracy wake word augmentations for BANKAI:
- Reads all 193 unique original BANKAI recordings
- Centers the active 1-second speech region
- Applies:
    1. Clean original
    2. Time shifts (-80ms, +80ms, -40ms, +40ms)
    3. Speed/pitch variation (0.96x - 1.04x)
    4. Room reverberation
    5. Background noise injection at SNR 20dB, 15dB, 10dB, 5dB
- Maintains parent stem for zero-leakage group splitting.
"""
from pathlib import Path
import random
import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
BANKAI_DIR = ROOT / "dataset" / "bankai"
NOISE_DIR = ROOT / "dataset" / "noise"
ISRO1_DIR = Path(r"F:\isro")
ISRO2_DIR = Path(r"F:\isro2")

SR = 16000
TARGET_SAMPLES = SR
SEED = 20260909

random.seed(SEED)
np.random.seed(SEED)

def rms(x):
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + 1e-12))

def center_or_pad_1sec(audio):
    if len(audio) == TARGET_SAMPLES:
        return audio.astype(np.float32)
    if len(audio) < TARGET_SAMPLES:
        out = np.zeros(TARGET_SAMPLES, dtype=np.float32)
        start = (TARGET_SAMPLES - len(audio)) // 2
        out[start:start + len(audio)] = audio
        return out
    
    # Audio is longer than 1 second: find highest energy 1-second window
    best_start = 0
    best_energy = -1.0
    step = 160
    for s in range(0, len(audio) - TARGET_SAMPLES + 1, step):
        segment = audio[s:s + TARGET_SAMPLES]
        e = np.mean(segment ** 2)
        if e > best_energy:
            best_energy = e
            best_start = s
    return audio[best_start:best_start + TARGET_SAMPLES].astype(np.float32)

def normalize_peak(x, target=0.92):
    peak = float(np.max(np.abs(x)))
    if peak > 1e-6:
        x = x * (target / peak)
    return np.clip(x, -0.999, 0.999).astype(np.float32)

def add_noise_snr(clean, noise, snr_db):
    clean_rms = rms(clean)
    noise_rms = rms(noise)
    if clean_rms < 1e-6 or noise_rms < 1e-6:
        return clean
    target_noise_rms = clean_rms / (10 ** (snr_db / 20.0))
    scaled_noise = noise * (target_noise_rms / noise_rms)
    return normalize_peak(clean + scaled_noise)

def apply_reverb(x):
    y = x.copy()
    for delay_ms, decay in [(30, random.uniform(0.08, 0.14)), (65, random.uniform(0.04, 0.08))]:
        d = int(SR * delay_ms / 1000)
        if d < len(y):
            y[d:] += decay * x[:-d]
    return y

def time_shift(x, shift_ms):
    shift = int(SR * shift_ms / 1000)
    y = np.zeros_like(x)
    if shift > 0:
        y[shift:] = x[:-shift]
    elif shift < 0:
        y[:shift] = x[-shift:]
    else:
        y[:] = x
    return y

def speed_pitch(x):
    factor = random.uniform(0.96, 1.04)
    n = max(1, int(round(len(x) / factor)))
    y = signal.resample(x, n)
    return center_or_pad_1sec(y)

def main():
    print("========================================")
    print("BANKAI DATASET CURATION & AUGMENTATION")
    print("========================================")
    
    orig_map = {}
    if (ISRO1_DIR / "dataset" / "bankai").exists():
        for p in (ISRO1_DIR / "dataset" / "bankai").glob("*.wav"):
            orig_map[p.stem] = p
            
    if (ISRO2_DIR / "dataset" / "bankai").exists():
        for p in (ISRO2_DIR / "dataset" / "bankai").glob("*.wav"):
            if "_aug" not in p.stem and p.stem not in orig_map:
                orig_map[p.stem] = p
                
    print(f"Total unique original clean BANKAI files: {len(orig_map)}")
    
    noise_files = list(NOISE_DIR.glob("*.wav"))
    print(f"Available noise files for mixing: {len(noise_files)}")
    
    total_created = 0
    for idx, (stem, path) in enumerate(sorted(orig_map.items()), start=1):
        data, sr_in = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]
        if sr_in != SR:
            n = max(1, int(round(len(data) * SR / sr_in)))
            data = signal.resample(data, n)
            
        clean_1s = normalize_peak(center_or_pad_1sec(data))
        
        # 1. Clean original
        sf.write(BANKAI_DIR / f"{stem}_orig.wav", clean_1s, SR, subtype="PCM_16")
        total_created += 1
        
        # 2. Time shifts
        for s_idx, ms in enumerate([-80, 80, -40, 40], start=1):
            shifted = normalize_peak(time_shift(clean_1s, ms))
            sf.write(BANKAI_DIR / f"{stem}_shift_{s_idx:02d}.wav", shifted, SR, subtype="PCM_16")
            total_created += 1
            
        # 3. Speed/Pitch
        sp = normalize_peak(speed_pitch(clean_1s))
        sf.write(BANKAI_DIR / f"{stem}_speed.wav", sp, SR, subtype="PCM_16")
        total_created += 1
        
        # 4. Reverb
        rev = normalize_peak(apply_reverb(clean_1s))
        sf.write(BANKAI_DIR / f"{stem}_reverb.wav", rev, SR, subtype="PCM_16")
        total_created += 1
        
        # 5. Background Noise Injection
        for snr_db in [20, 15, 10, 5]:
            n_file = random.choice(noise_files)
            n_data, _ = sf.read(n_file, dtype="float32")
            if n_data.ndim > 1:
                n_data = n_data[:, 0]
            n_1s = center_or_pad_1sec(n_data)
            noisy = add_noise_snr(clean_1s, n_1s, snr_db)
            sf.write(BANKAI_DIR / f"{stem}_noise_{snr_db}dB.wav", noisy, SR, subtype="PCM_16")
            total_created += 1
            
        if idx % 25 == 0 or idx == len(orig_map):
            print(f"  Processed {idx}/{len(orig_map)} originals -> {total_created} BANKAI clips created", end="\r")
            
    print(f"\nFinished: Created {total_created} total BANKAI clips in {BANKAI_DIR}")

if __name__ == "__main__":
    main()
