"""
03c_integrate_krishna_dataset.py
Integrates krishnaDataSet into dataset:
1. BANKAI: 52 new real-person BANKAI recordings from krishnaDataSet/dataset/bankai:
   - Centers the 1-second active region
   - Generates conservative augmentations (time shifts, speed/pitch, reverb, subtle -30dB hiss)
2. UNKNOWN: Extracts 2,400 clean speech command WAVs from mini_speech_commands.zip (300 each from 8 words)
3. NOISE: Adds 300 synthetic ambient clips (typing, breath, room floor)
"""
from pathlib import Path
import zipfile
import random
import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
BANKAI_DIR = DATASET / "bankai"
UNKNOWN_DIR = DATASET / "unknown"
NOISE_DIR = DATASET / "noise"

KRISHNA_DIR = ROOT / "krishnaDataSet" / "dataset"
KRISHNA_BANKAI = KRISHNA_DIR / "bankai"
KRISHNA_SC_ZIP = KRISHNA_DIR / "mini_speech_commands.zip"

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
    noise = np.random.randn(len(x)).astype(np.float32)
    clean_rms = np.sqrt(np.mean(x**2) + 1e-12)
    noise_rms = np.sqrt(np.mean(noise**2) + 1e-12)
    scale = (clean_rms * (10 ** (noise_db / 20.0))) / max(noise_rms, 1e-9)
    return x + noise * scale

def process_krishna_bankai():
    print("\n--- Integrating krishnaDataSet BANKAI ---")
    wav_files = sorted(KRISHNA_BANKAI.glob("*.wav"))
    print(f"Found {len(wav_files)} new BANKAI recordings in {KRISHNA_BANKAI}")
    
    created = 0
    for p in wav_files:
        stem = p.stem
        data, sr_in = sf.read(p, dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]
        if sr_in != SR:
            n = max(1, int(round(len(data) * SR / sr_in)))
            data = signal.resample(data, n)
        clean_1s = normalize_peak(center_or_pad_1sec(data))
        
        # 1. Original
        sf.write(BANKAI_DIR / f"krishna_{stem}_orig.wav", clean_1s, SR, subtype="PCM_16")
        created += 1
        
        # 2. Time shifts
        for idx, ms in enumerate([-60, 60, -30, 30], start=1):
            s = normalize_peak(time_shift(clean_1s, ms))
            sf.write(BANKAI_DIR / f"krishna_{stem}_shift_{idx:02d}.wav", s, SR, subtype="PCM_16")
            created += 1
            
        # 3. Speed/Pitch
        for idx, factor in enumerate([0.96, 1.04], start=1):
            sp = normalize_peak(speed_pitch(clean_1s, factor))
            sf.write(BANKAI_DIR / f"krishna_{stem}_speed_{idx:02d}.wav", sp, SR, subtype="PCM_16")
            created += 1
            
        # 4. Reverb
        rev = normalize_peak(apply_reverb(clean_1s))
        sf.write(BANKAI_DIR / f"krishna_{stem}_reverb.wav", rev, SR, subtype="PCM_16")
        created += 1
        
        # 5. Subtle quiet floor hiss
        hiss = normalize_peak(add_subtle_hiss(clean_1s, noise_db=-30))
        sf.write(BANKAI_DIR / f"krishna_{stem}_hiss.wav", hiss, SR, subtype="PCM_16")
        created += 1
        
    total_bankai = len(list(BANKAI_DIR.glob("*.wav")))
    print(f"Added {created} new BANKAI clips. Total BANKAI now: {total_bankai}")

def process_mini_speech_commands():
    print("\n--- Extracting Speech Commands from mini_speech_commands.zip ---")
    if not KRISHNA_SC_ZIP.exists():
        print(f"Zip not found: {KRISHNA_SC_ZIP}")
        return
        
    extracted = 0
    words = ["down", "go", "left", "no", "right", "stop", "up", "yes"]
    word_targets = {w: 300 for w in words}
    word_counts = {w: 0 for w in words}
    
    with zipfile.ZipFile(KRISHNA_SC_ZIP) as z:
        for info in z.infolist():
            if not info.filename.endswith(".wav") or "__MACOSX" in info.filename:
                continue
            parts = info.filename.split("/")
            if len(parts) >= 3:
                w = parts[1]
                if w in word_targets and word_counts[w] < word_targets[w]:
                    fname = Path(info.filename).name
                    dst = UNKNOWN_DIR / f"krishna_sc_{w}_{fname}"
                    if not dst.exists():
                        with z.open(info) as f:
                            data, sr_in = sf.read(f, dtype="float32")
                            clean = normalize_peak(center_or_pad_1sec(data))
                            sf.write(dst, clean, SR, subtype="PCM_16")
                    extracted += 1
                    word_counts[w] += 1
                    
    total_unk = len(list(UNKNOWN_DIR.glob("*.wav")))
    print(f"Extracted {extracted} Speech Commands into UNKNOWN. Total UNKNOWN now: {total_unk}")

def add_extra_ambient_noise():
    print("\n--- Adding Extra Ambient Noise Profiles ---")
    created = 0
    for idx in range(150):
        # Keyboard typing simulation (clicks on pink background)
        samples = TARGET_SAMPLES
        audio = np.zeros(samples, dtype=np.float32)
        num_clicks = np.random.randint(2, 6)
        for _ in range(num_clicks):
            pos = np.random.randint(1000, samples - 1000)
            clen = np.random.randint(60, 250)
            audio[pos:pos+clen] += np.random.randn(clen) * np.exp(-np.linspace(0, 6, clen))
        # Add subtle room air
        white = np.random.randn(samples).astype(np.float32)
        b, a = signal.butter(2, 500.0 / (SR / 2), btype='low')
        room_air = signal.lfilter(b, a, white) * 0.05
        audio += room_air
        audio = normalize_peak(audio, target=np.random.uniform(0.1, 0.4))
        sf.write(NOISE_DIR / f"synth_ambient_typing_{idx+1:04d}.wav", audio, SR, subtype="PCM_16")
        created += 1
        
    for idx in range(150):
        # Low frequency room rumble
        white = np.random.randn(TARGET_SAMPLES).astype(np.float32)
        b, a = signal.butter(2, 120.0 / (SR / 2), btype='low')
        rumble = signal.lfilter(b, a, white)
        rumble = normalize_peak(rumble, target=np.random.uniform(0.1, 0.35))
        sf.write(NOISE_DIR / f"synth_ambient_rumble_{idx+1:04d}.wav", rumble, SR, subtype="PCM_16")
        created += 1
        
    total_noise = len(list(NOISE_DIR.glob("*.wav")))
    print(f"Added {created} extra ambient noise clips. Total NOISE now: {total_noise}")

if __name__ == "__main__":
    process_krishna_bankai()
    process_mini_speech_commands()
    add_extra_ambient_noise()
    print("\nIntegration of krishnaDataSet complete!")
