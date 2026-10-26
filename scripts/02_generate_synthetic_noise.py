"""
02_generate_synthetic_noise.py
Generates diverse synthetic noise profiles:
- Pink noise (1/f)
- Brownian noise (1/f^2)
- White noise
- HVAC / Fan hums (50/60 Hz harmonics + low-frequency air rumble)
- Room silence & ambient floor
- Transient clicks / mic thumps / office pops
- Distant acoustic babble simulation

All clips are strictly 1.0 second, 16,000 Hz, mono PCM16.
"""
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parent.parent
NOISE_DIR = ROOT / "dataset" / "noise"
NOISE_DIR.mkdir(parents=True, exist_ok=True)

SR = 16000
DURATION = 1.0
NUM_SAMPLES = int(SR * DURATION)
SEED = 20260909

np.random.seed(SEED)

def make_pink_noise(samples):
    # Pink noise via 1/f filter
    white = np.random.randn(samples).astype(np.float32)
    # Voss-McCartney or IIR approximation (Paul Kellet's filter)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1.0, -2.494956002, 2.017265875, -0.522189400]
    pink = signal.lfilter(b, a, white)
    return pink.astype(np.float32)

def make_brown_noise(samples):
    # Brown noise via integration of white noise
    white = np.random.randn(samples).astype(np.float32)
    brown = np.cumsum(white)
    # High-pass filter to prevent DC drift
    b, a = signal.butter(2, 30.0 / (SR / 2), btype='high')
    brown = signal.lfilter(b, a, brown)
    return brown.astype(np.float32)

def make_white_noise(samples):
    return np.random.randn(samples).astype(np.float32)

def make_hvac_fan_hum(samples, freq=50.0):
    t = np.linspace(0, DURATION, samples, endpoint=False)
    # Fundamental + harmonics
    hum = (0.6 * np.sin(2 * np.pi * freq * t) +
           0.3 * np.sin(2 * np.pi * (2 * freq) * t) +
           0.15 * np.sin(2 * np.pi * (3 * freq) * t) +
           0.08 * np.sin(2 * np.pi * (4 * freq) * t))
    # Add turbulent air flow rumble
    rumble = make_brown_noise(samples)
    rumble = rumble / (np.max(np.abs(rumble)) + 1e-6)
    audio = 0.5 * hum + 0.5 * rumble
    return audio.astype(np.float32)

def make_transient_clicks(samples, num_clicks=4):
    audio = np.zeros(samples, dtype=np.float32)
    click_positions = np.random.randint(500, samples - 1000, size=num_clicks)
    for pos in click_positions:
        click_len = np.random.randint(80, 400)
        decay = np.exp(-np.linspace(0, 8, click_len))
        click = np.random.randn(click_len) * decay
        audio[pos:pos+click_len] += click
    # Add mild ambient floor
    audio += 0.05 * make_pink_noise(samples)
    return audio.astype(np.float32)

def make_room_silence(samples):
    # Extremely low level room floor noise
    noise = make_pink_noise(samples)
    noise = noise / (np.max(np.abs(noise)) + 1e-6)
    rms_target = np.random.uniform(0.0005, 0.004)
    return (noise * rms_target).astype(np.float32)

def make_babble_noise(samples):
    # Overlay multiple random narrowband formant bands to simulate distant unintelligible murmurs
    audio = np.zeros(samples, dtype=np.float32)
    t = np.linspace(0, DURATION, samples, endpoint=False)
    for _ in range(8):
        cf = np.random.uniform(250, 3200)
        bw = np.random.uniform(80, 200)
        low = max(20, cf - bw / 2)
        high = min(SR / 2 - 50, cf + bw / 2)
        b, a = signal.butter(2, [low / (SR / 2), high / (SR / 2)], btype='band')
        src = np.random.randn(samples)
        filtered = signal.lfilter(b, a, src)
        # AM modulation 2-8 Hz (speech rate)
        mod = 0.5 * (1.0 + np.sin(2 * np.pi * np.random.uniform(2, 7) * t + np.random.uniform(0, 2*np.pi)))
        audio += (filtered * mod).astype(np.float32)
    return audio.astype(np.float32)

def generate_all(target_count_per_type=60):
    generators = [
        ("pink", make_pink_noise),
        ("brown", make_brown_noise),
        ("white", make_white_noise),
        ("fan50", lambda s: make_hvac_fan_hum(s, 50.0)),
        ("fan60", lambda s: make_hvac_fan_hum(s, 60.0)),
        ("clicks", make_transient_clicks),
        ("silence", make_room_silence),
        ("babble", make_babble_noise),
    ]
    
    total_created = 0
    print(f"Generating synthetic noise profiles into {NOISE_DIR}...")
    for name, fn in generators:
        for idx in range(target_count_per_type):
            audio = fn(NUM_SAMPLES)
            # Scale to varied RMS/peak levels
            peak = np.max(np.abs(audio))
            if peak > 1e-6:
                if name == "silence":
                    target_peak = np.random.uniform(0.01, 0.04)
                else:
                    target_peak = np.random.uniform(0.3, 0.85)
                audio = (audio / peak) * target_peak
            audio = np.clip(audio, -0.999, 0.999).astype(np.float32)
            
            filename = f"synth_{name}_{idx+1:04d}.wav"
            sf.write(NOISE_DIR / filename, audio, SR, subtype="PCM_16")
            total_created += 1
            
        print(f"  Created {target_count_per_type} clips for {name}")
        
    print(f"Total synthetic noise clips created: {total_created}")

if __name__ == "__main__":
    generate_all(target_count_per_type=60)
