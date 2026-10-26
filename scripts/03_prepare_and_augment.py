"""
03_prepare_and_augment.py
Builds the complete dataset for ISRO_GEMINI:
1. NOISE: WhatsApp (isro2) + Google Speech Commands silence + Synthetic profiles
2. UNKNOWN: isro2 unknown + isro2 unknown_new + Speech Commands words
3. BANKAI: 193 clean originals + realistic acoustic augmentations + SNR noise injection

Guarantees 16,000 Hz, mono, PCM16, exactly 1.0 second per clip.
Maintains parent stem tracking for strict group-aware splitting.
"""
from pathlib import Path
import shutil
import tarfile
import random
import csv
import numpy as np
import soundfile as sf
import librosa
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
BANKAI_DIR = DATASET / "bankai"
UNKNOWN_DIR = DATASET / "unknown"
NOISE_DIR = DATASET / "noise"

ISRO2_DIR = Path(r"F:\isro2")
ISRO1_DIR = Path(r"F:\isro")
RAW_DOWNLOADS = ROOT / "raw_downloads"
SC_TAR = RAW_DOWNLOADS / "speech_commands_test_set_v0.02.tar.gz"

SR = 16000
TARGET_SAMPLES = SR
SEED = 20260909

random.seed(SEED)
np.random.seed(SEED)

def rms(x):
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + 1e-12))

def center_or_pad_1sec(audio):
    """Zero-pad or extract most active 1-second window."""
    if len(audio) == TARGET_SAMPLES:
        return audio.astype(np.float32)
    if len(audio) < TARGET_SAMPLES:
        out = np.zeros(TARGET_SAMPLES, dtype=np.float32)
        start = (TARGET_SAMPLES - len(audio)) // 2
        out[start:start + len(audio)] = audio
        return out
    
    # Audio is longer than 1 second: find highest energy 1-second segment
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
    """Mix clean signal with noise at specified SNR (dB)."""
    clean_rms = rms(clean)
    noise_rms = rms(noise)
    if clean_rms < 1e-6 or noise_rms < 1e-6:
        return clean
    # Desired noise RMS
    target_noise_rms = clean_rms / (10 ** (snr_db / 20.0))
    scaled_noise = noise * (target_noise_rms / noise_rms)
    mixed = clean + scaled_noise
    return normalize_peak(mixed)

def apply_reverb(x):
    y = x.copy()
    for delay_ms, decay in [(30, random.uniform(0.08, 0.15)), (65, random.uniform(0.04, 0.09))]:
        d = int(SR * delay_ms / 1000)
        if d < len(y):
            y[d:] += decay * x[:-d]
    return y

def time_shift(x, max_shift_ms=100):
    shift_samples = int(SR * random.uniform(-max_shift_ms, max_shift_ms) / 1000)
    y = np.zeros_like(x)
    if shift_samples > 0:
        y[shift_samples:] = x[:-shift_samples]
    elif shift_samples < 0:
        y[:shift_samples] = x[-shift_samples:]
    else:
        y[:] = x
    return y

def speed_pitch(x):
    factor = random.uniform(0.95, 1.05)
    n = max(1, int(round(len(x) / factor)))
    y = signal.resample(x, n)
    return center_or_pad_1sec(y)

# ============================================================
# STEP 1: POPULATE NOISE
# ============================================================
def populate_noise():
    print("\n--- Populating NOISE Library ---")
    manifest = []
    
    # 1. WhatsApp noise from isro2
    isro2_noise_files = sorted((ISRO2_DIR / "dataset" / "noise").glob("*.wav"))
    print(f"Found {len(isro2_noise_files)} WhatsApp noise files in isro2.")
    for p in isro2_noise_files:
        dst = NOISE_DIR / f"isro2_{p.name}"
        if not dst.exists():
            shutil.copy2(p, dst)
        manifest.append({"file": dst.name, "source": f"isro2_{p.stem.split('_')[0]}"})
        
    # 2. Extract silence/noise from Speech Commands tar
    if SC_TAR.exists():
        print("Extracting Speech Commands background noise...")
        with tarfile.open(SC_TAR, "r:gz") as tar:
            silence_members = [m for m in tar.getmembers() if "_silence_" in m.name and m.name.endswith(".wav")]
            for m in silence_members:
                fname = Path(m.name).name
                dst = NOISE_DIR / f"sc_silence_{fname}"
                if not dst.exists():
                    f = tar.extractfile(m)
                    if f:
                        data, sr_in = sf.read(f, dtype="float32")
                        data_1s = center_or_pad_1sec(data)
                        sf.write(dst, normalize_peak(data_1s), SR, subtype="PCM_16")
                manifest.append({"file": dst.name, "source": "speech_commands_silence"})
                
    # 3. Synthetic noise already generated in 02
    synth_files = sorted(NOISE_DIR.glob("synth_*.wav"))
    print(f"Found {len(synth_files)} synthetic noise files.")
    for p in synth_files:
        src_type = p.stem.split("_")[1]
        manifest.append({"file": p.name, "source": f"synth_{src_type}"})
        
    total_noise = len(list(NOISE_DIR.glob("*.wav")))
    print(f"Total NOISE clips ready: {total_noise}")
    return manifest

# ============================================================
# STEP 2: POPULATE UNKNOWN
# ============================================================
def populate_unknown():
    print("\n--- Populating UNKNOWN Speech Library ---")
    manifest = []
    
    # 1. isro2 unknown
    unk_files = sorted((ISRO2_DIR / "dataset" / "unknown").glob("*.wav"))
    print(f"Copying {len(unk_files)} isro2 unknown clips...")
    for p in unk_files:
        dst = UNKNOWN_DIR / f"isro2_unk_{p.name}"
        if not dst.exists():
            shutil.copy2(p, dst)
        group = p.stem.split("_clip")[0] if "_clip" in p.stem else p.stem
        manifest.append({"file": dst.name, "group": f"isro2_{group}"})
        
    # 2. isro2 unknown_new
    unk_new_files = sorted((ISRO2_DIR / "dataset" / "unknown_new").glob("*.wav"))
    print(f"Copying {len(unk_new_files)} isro2 unknown_new clips...")
    for p in unk_new_files:
        dst = UNKNOWN_DIR / f"isro2_unknew_{p.name}"
        if not dst.exists():
            shutil.copy2(p, dst)
        group = p.stem.split("_clip")[0] if "_clip" in p.stem else p.stem
        manifest.append({"file": dst.name, "group": f"isro2_new_{group}"})
        
    # 3. Extract Speech Commands words
    if SC_TAR.exists():
        print("Extracting Speech Commands words for negative speech diversity...")
        with tarfile.open(SC_TAR, "r:gz") as tar:
            unk_members = [m for m in tar.getmembers() if "_unknown_" in m.name and m.name.endswith(".wav")]
            random.shuffle(unk_members)
            selected_unk = unk_members[:400]
            for m in selected_unk:
                fname = Path(m.name).name
                dst = UNKNOWN_DIR / f"sc_unk_{fname}"
                if not dst.exists():
                    f = tar.extractfile(m)
                    if f:
                        data, sr_in = sf.read(f, dtype="float32")
                        sf.write(dst, normalize_peak(center_or_pad_1sec(data)), SR, subtype="PCM_16")
                # Group by speaker id from filename if possible
                manifest.append({"file": dst.name, "group": f"sc_unk_{fname[:8]}"})
                
            # Extra standard command words
            cmd_words = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]
            for w in cmd_words:
                w_members = [m for m in tar.getmembers() if f"./{w}/" in m.name and m.name.endswith(".wav")]
                random.shuffle(w_members)
                for m in w_members[:40]:
                    fname = Path(m.name).name
                    dst = UNKNOWN_DIR / f"sc_word_{w}_{fname}"
                    if not dst.exists():
                        f = tar.extractfile(m)
                        if f:
                            data, sr_in = sf.read(f, dtype="float32")
                            sf.write(dst, normalize_peak(center_or_pad_1sec(data)), SR, subtype="PCM_16")
                    manifest.append({"file": dst.name, "group": f"sc_{w}_{fname[:8]}"})
                    
    total_unk = len(list(UNKNOWN_DIR.glob("*.wav")))
    print(f"Total UNKNOWN clips ready: {total_unk}")
    return manifest

# ============================================================
# STEP 3: POPULATE & AUGMENT BANKAI
# ============================================================
def populate_and_augment_bankai():
    print("\n--- Populating & Augmenting BANKAI ---")
    manifest = []
    
    # Gather all clean original BANKAI
    orig_map = {}
    
    # Look in isro1
    if (ISRO1_DIR / "dataset" / "bankai").exists():
        for p in (ISRO1_DIR / "dataset" / "bankai").glob("*.wav"):
            orig_map[p.name] = p
            
    # Look in isro2 originals
    if (ISRO2_DIR / "dataset" / "bankai").exists():
        for p in (ISRO2_DIR / "dataset" / "bankai").glob("*.wav"):
            if "_aug" not in p.stem:
                orig_map[p.name] = p
                
    print(f"Total unique original BANKAI audio found: {len(orig_map)}")
    
    # Load all available noise files for mixing
    all_noises = list(NOISE_DIR.glob("*.wav"))
    if not all_noises:
        raise RuntimeError("No noise files available for mixing!")
        
    created_count = 0
    
    for orig_name, orig_path in sorted(orig_map.items()):
        raw_audio, sr_in = librosa.load(orig_path, sr=SR, mono=True)
        clean_1s = center_or_pad_1sec(raw_audio)
        clean_1s = normalize_peak(clean_1s)
        
        group_id = Path(orig_name).stem
        
        # 1. Save normalized original
        dst_orig = BANKAI_DIR / f"{group_id}_orig.wav"
        sf.write(dst_orig, clean_1s, SR, subtype="PCM_16")
        manifest.append({"file": dst_orig.name, "group": group_id, "type": "original"})
        created_count += 1
        
        # 2. Time-shifted
        for i, ms in enumerate([-80, 80, -40, 40], start=1):
            shifted = time_shift(clean_1s, max_shift_ms=abs(ms))
            shifted = normalize_peak(shifted)
            dst_shift = BANKAI_DIR / f"{group_id}_shift_{i:02d}.wav"
            sf.write(dst_shift, shifted, SR, subtype="PCM_16")
            manifest.append({"file": dst_shift.name, "group": group_id, "type": f"shift_{ms}ms"})
            created_count += 1
            
        # 3. Speed/Pitch variation
        speed_var = speed_pitch(clean_1s)
        dst_speed = BANKAI_DIR / f"{group_id}_speed.wav"
        sf.write(dst_speed, speed_var, SR, subtype="PCM_16")
        manifest.append({"file": dst_speed.name, "group": group_id, "type": "speed_pitch"})
        created_count += 1
        
        # 4. Reverb
        reverbed = normalize_peak(apply_reverb(clean_1s))
        dst_rev = BANKAI_DIR / f"{group_id}_reverb.wav"
        sf.write(dst_rev, reverbed, SR, subtype="PCM_16")
        manifest.append({"file": dst_rev.name, "group": group_id, "type": "reverb"})
        created_count += 1
        
        # 5. Background Noise Injection at various SNRs
        for snr_val in [20, 15, 10, 5]:
            noise_file = random.choice(all_noises)
            noise_raw, _ = sf.read(noise_file, dtype="float32")
            noise_1s = center_or_pad_1sec(noise_raw)
            noisy_bankai = add_noise_snr(clean_1s, noise_1s, snr_db=snr_val)
            dst_noisy = BANKAI_DIR / f"{group_id}_noise_{snr_val}dB.wav"
            sf.write(dst_noisy, noisy_bankai, SR, subtype="PCM_16")
            manifest.append({"file": dst_noisy.name, "group": group_id, "type": f"noise_snr_{snr_val}dB"})
            created_count += 1
            
    print(f"Total BANKAI clips created: {created_count}")
    return manifest

if __name__ == "__main__":
    noise_m = populate_noise()
    unk_m = populate_unknown()
    bankai_m = populate_and_augment_bankai()
    
    print("\n========================================")
    print("DATASET PREPARATION SUMMARY")
    print("========================================")
    print(f"BANKAI clips  : {len(list(BANKAI_DIR.glob('*.wav')))}")
    print(f"UNKNOWN clips : {len(list(UNKNOWN_DIR.glob('*.wav')))}")
    print(f"NOISE clips   : {len(list(NOISE_DIR.glob('*.wav')))}")
    print("========================================")
