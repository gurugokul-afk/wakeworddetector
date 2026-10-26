"""
rebuild_high_precision_dataset.py
Rebuilds dataset with focus on ZERO false activations:
1. BANKAI: Crisp clean wake word (193 originals + time shift + speed/pitch + reverb + subtle -30dB hiss).
   NO loud noise injection that causes noise to be misidentified as BANKAI.
2. UNKNOWN: Massive negative speech library (isro2 unknown + isro2 unknown_new + Speech Commands words).
3. NOISE: Massive environmental noise (WhatsApp + Speech Commands silence + synthetic profiles).
"""
from pathlib import Path
import shutil
import tarfile
import random
import numpy as np
import soundfile as sf
import librosa
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
BANKAI_DIR = DATASET / "bankai"
UNKNOWN_DIR = DATASET / "unknown"
NOISE_DIR = DATASET / "noise"

ISRO1_DIR = Path(r"F:\isro")
ISRO2_DIR = Path(r"F:\isro2")
SC_TAR = ROOT / "raw_downloads" / "speech_commands_test_set_v0.02.tar.gz"

SR = 16000
TARGET_SAMPLES = SR
SEED = 20260909

random.seed(SEED)
np.random.seed(SEED)

def center_or_pad_1sec(audio):
    if len(audio) == TARGET_SAMPLES:
        return audio.astype(np.float32)
    if len(audio) < TARGET_SAMPLES:
        out = np.zeros(TARGET_SAMPLES, dtype=np.float32)
        start = (TARGET_SAMPLES - len(audio)) // 2
        out[start:start + len(audio)] = audio
        return out
    # Longer than 1 sec: center highest energy window
    best_start = 0
    best_energy = -1.0
    for s in range(0, len(audio) - TARGET_SAMPLES + 1, 160):
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

def speed_pitch(x, factor):
    n = max(1, int(round(len(x) / factor)))
    y = signal.resample(x, n)
    return center_or_pad_1sec(y)

def apply_reverb(x):
    y = x.copy()
    for delay_ms, decay in [(28, 0.10), (55, 0.05)]:
        d = int(SR * delay_ms / 1000)
        if d < len(y):
            y[d:] += decay * x[:-d]
    return y

def add_subtle_hiss(x, noise_db=-30):
    # Very subtle floor hiss only (never overwhelming speech)
    noise = np.random.randn(len(x)).astype(np.float32)
    clean_rms = np.sqrt(np.mean(x**2) + 1e-12)
    noise_rms = np.sqrt(np.mean(noise**2) + 1e-12)
    scale = (clean_rms * (10 ** (noise_db / 20.0))) / max(noise_rms, 1e-9)
    return x + noise * scale

def clean_and_rebuild_bankai():
    print("\n--- 1. Rebuilding Clean BANKAI Dataset ---")
    if BANKAI_DIR.exists():
        shutil.rmtree(BANKAI_DIR)
    BANKAI_DIR.mkdir(parents=True, exist_ok=True)
    
    orig_map = {}
    if (ISRO1_DIR / "dataset" / "bankai").exists():
        for p in (ISRO1_DIR / "dataset" / "bankai").glob("*.wav"):
            orig_map[p.stem] = p
    if (ISRO2_DIR / "dataset" / "bankai").exists():
        for p in (ISRO2_DIR / "dataset" / "bankai").glob("*.wav"):
            if "_aug" not in p.stem and p.stem not in orig_map:
                orig_map[p.stem] = p
                
    print(f"Found {len(orig_map)} clean original BANKAI recordings.")
    
    created = 0
    for stem, path in sorted(orig_map.items()):
        data, sr_in = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]
        if sr_in != SR:
            n = max(1, int(round(len(data) * SR / sr_in)))
            data = signal.resample(data, n)
        clean_1s = normalize_peak(center_or_pad_1sec(data))
        
        # 1. Clean original
        sf.write(BANKAI_DIR / f"{stem}_orig.wav", clean_1s, SR, subtype="PCM_16")
        created += 1
        
        # 2. Time shifts (mild -80, -40, +40, +80 ms)
        for idx, ms in enumerate([-80, -40, 40, 80], start=1):
            s = normalize_peak(time_shift(clean_1s, ms))
            sf.write(BANKAI_DIR / f"{stem}_shift_{idx:02d}.wav", s, SR, subtype="PCM_16")
            created += 1
            
        # 3. Speed variations
        for idx, factor in enumerate([0.96, 1.04], start=1):
            sp = normalize_peak(speed_pitch(clean_1s, factor))
            sf.write(BANKAI_DIR / f"{stem}_speed_{idx:02d}.wav", sp, SR, subtype="PCM_16")
            created += 1
            
        # 4. Mild room reverb
        rev = normalize_peak(apply_reverb(clean_1s))
        sf.write(BANKAI_DIR / f"{stem}_reverb.wav", rev, SR, subtype="PCM_16")
        created += 1
        
        # 5. Subtle quiet hiss
        hiss = normalize_peak(add_subtle_hiss(clean_1s, noise_db=-30))
        sf.write(BANKAI_DIR / f"{stem}_hiss.wav", hiss, SR, subtype="PCM_16")
        created += 1
        
    print(f"BANKAI dataset rebuilt: {created} clean, crisp clips (no loud noise corruption).")

def expand_unknown():
    print("\n--- 2. Expanding UNKNOWN Speech Dataset ---")
    # UNKNOWN_DIR already contains isro2 clips; let's extract extra words from Speech Commands tar sequentially!
    if not SC_TAR.exists():
        print("Warning: SC_TAR not found")
        return
        
    existing_count = len(list(UNKNOWN_DIR.glob("*.wav")))
    print(f"Current UNKNOWN clips: {existing_count}")
    
    cmd_targets = {"down": 100, "go": 100, "left": 100, "no": 100, "off": 100,
                   "on": 100, "right": 100, "stop": 100, "up": 100, "yes": 100}
    cmd_collected = {k: 0 for k in cmd_targets}
    
    extracted = 0
    with tarfile.open(SC_TAR, "r:gz") as tar:
        for member in tar:
            if not member.name.endswith(".wav"):
                continue
            parts = member.name.split("/")
            if len(parts) >= 3:
                word = parts[1]
                if word in cmd_targets and cmd_collected[word] < cmd_targets[word]:
                    dst = UNKNOWN_DIR / f"sc_{word}_{Path(member.name).name}"
                    if not dst.exists():
                        f = tar.extractfile(member)
                        if f:
                            data, sr_in = sf.read(f, dtype="float32")
                            clean = normalize_peak(center_or_pad_1sec(data))
                            sf.write(dst, clean, SR, subtype="PCM_16")
                            extracted += 1
                            cmd_collected[word] += 1
                            
    total_unk = len(list(UNKNOWN_DIR.glob("*.wav")))
    print(f"Extracted {extracted} extra speech command words. Total UNKNOWN now: {total_unk}")

def verify_noise():
    total_noise = len(list(NOISE_DIR.glob("*.wav")))
    print(f"\n--- 3. NOISE Dataset Status ---")
    print(f"Total NOISE clips available: {total_noise}")

if __name__ == "__main__":
    clean_and_rebuild_bankai()
    expand_unknown()
    verify_noise()
    print("\nDataset ready for feature extraction and training!")
