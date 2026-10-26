import wave
import numpy as np

def load_int16(path):
    with wave.open(path, 'rb') as wf:
        data = wf.readframes(wf.getnframes())
        samples = np.frombuffer(data, dtype=np.int16)
        if len(samples) < 16000:
            samples = np.pad(samples, (0, 16000 - len(samples)))
        else:
            samples = samples[:16000]
        return samples

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
bankai_pcm = load_int16(ROOT / 'dataset' / 'bankai' / 'a10_orig.wav')
unk_pcm = load_int16(ROOT / 'dataset' / 'unknown' / 'isro2_unknew_unknown_new_src01_clip0004.wav')

with open(Path(__file__).resolve().parent / 'main' / 'test_audio_samples.h', 'w') as f:
    f.write('// Auto-generated test audio samples for on-device verification\n')
    f.write('#ifndef TEST_AUDIO_SAMPLES_H_\n')
    f.write('#define TEST_AUDIO_SAMPLES_H_\n\n')
    f.write('#include <stdint.h>\n\n')
    f.write('#define TEST_AUDIO_LEN 16000\n\n')
    
    f.write('// Spoken BANKAI sample\n')
    f.write('alignas(4) const int16_t kBankaiTestAudio[TEST_AUDIO_LEN] = {\n')
    for i in range(0, len(bankai_pcm), 16):
        chunk = bankai_pcm[i:i+16]
        f.write('    ' + ', '.join(f'{x}' for x in chunk) + ',\n')
    f.write('};\n\n')

    f.write('// Spoken UNKNOWN word sample\n')
    f.write('alignas(4) const int16_t kUnknownTestAudio[TEST_AUDIO_LEN] = {\n')
    for i in range(0, len(unk_pcm), 16):
        chunk = unk_pcm[i:i+16]
        f.write('    ' + ', '.join(f'{x}' for x in chunk) + ',\n')
    f.write('};\n\n')
    f.write('#endif // TEST_AUDIO_SAMPLES_H_\n')

print('Generated test_audio_samples.h successfully!')
