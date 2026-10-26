#ifndef AUDIO_PROCESSOR_H_
#define AUDIO_PROCESSOR_H_

#include <math.h>
#include <stdint.h>
#include <string.h>
#include "mel_tables.h"
#include "rfft_512.h"

#define AUDIO_WINDOW_SAMPLES 16000
#define NUM_MEL_BANDS 40
#define NUM_TIME_FRAMES 101
#define STFT_HOP_SAMPLES 160

class AudioProcessor {
public:
    AudioProcessor() {
        memset(mel_spectrogram_, 0, sizeof(mel_spectrogram_));
    }

    // Calculates RMS and Peak energy for Voice Activity Detection (VAD)
    void calculate_energy(const float* audio, int length, float& out_rms, float& out_peak) {
        float sum_sq = 0.0f;
        float peak = 0.0f;
        for (int i = 0; i < length; i++) {
            float val = audio[i];
            float abs_val = fabsf(val);
            if (abs_val > peak) {
                peak = abs_val;
            }
            sum_sq += val * val;
        }
        out_rms = sqrtf(sum_sq / (float)length);
        out_peak = peak;
    }

    // Extracts 40 x 101 INT8 Log-Mel Spectrogram matching 04_extract_features.py & 09_live_test.py
    void extract_features(
        const float* raw_audio,
        float in_scale,
        int32_t in_zero_point,
        int8_t* out_tensor_int8
    ) {
        // Peak normalization scale
        float peak = 0.0f;
        for (int i = 0; i < AUDIO_WINDOW_SAMPLES; i++) {
            float a = fabsf(raw_audio[i]);
            if (a > peak) peak = a;
        }
        float norm_factor = (peak > 0.025f) ? (0.92f / peak) : 1.0f;

        for (int f = 0; f < NUM_TIME_FRAMES; f++) {
            int start_sample = f * STFT_HOP_SAMPLES - 256;

            // Apply on-the-fly zero padding and Hann window (saves 66 KB RAM!)
            for (int i = 0; i < FFT_LEN; i++) {
                int idx = start_sample + i;
                float sample = (idx >= 0 && idx < AUDIO_WINDOW_SAMPLES) ? (raw_audio[idx] * norm_factor) : 0.0f;
                frame_[i] = sample * kHannWindow[i];
            }

            // Compute 512-point RFFT power
            fft_.compute_power(frame_, power_spectrum_);

            // Apply triangular Mel filters
            for (int m = 0; m < NUM_MEL_BANDS; m++) {
                const MelFilter& filter = kMelFilters[m];
                float mel_sum = 0.0f;
                for (int b = 0; b < filter.num_bins; b++) {
                    mel_sum += power_spectrum_[filter.start_bin + b] * filter.weights[b];
                }
                mel_spectrogram_[m][f] = mel_sum;
            }
        }

        // Power-to-dB conversion (ref = max)
        float max_power = 1e-10f;
        for (int m = 0; m < NUM_MEL_BANDS; m++) {
            for (int f = 0; f < NUM_TIME_FRAMES; f++) {
                if (mel_spectrogram_[m][f] > max_power) {
                    max_power = mel_spectrogram_[m][f];
                }
            }
        }

        float sum_db = 0.0f;
        for (int m = 0; m < NUM_MEL_BANDS; m++) {
            for (int f = 0; f < NUM_TIME_FRAMES; f++) {
                float val = mel_spectrogram_[m][f];
                if (val < 1e-10f) val = 1e-10f;
                float db = 10.0f * log10f(val / max_power);
                if (db < -80.0f) db = -80.0f;
                mel_spectrogram_[m][f] = db;
                sum_db += db;
            }
        }

        // Standardization: (db - mean) / (std + 1e-6)
        int total_elements = NUM_MEL_BANDS * NUM_TIME_FRAMES;
        float mean_db = sum_db / (float)total_elements;

        float var_sum = 0.0f;
        for (int m = 0; m < NUM_MEL_BANDS; m++) {
            for (int f = 0; f < NUM_TIME_FRAMES; f++) {
                float diff = mel_spectrogram_[m][f] - mean_db;
                var_sum += diff * diff;
            }
        }
        float std_db = sqrtf(var_sum / (float)total_elements);
        float std_denom = std_db + 1e-6f;

        // INT8 Quantization: out_tensor shape is [40, 101, 1]
        int out_idx = 0;
        for (int m = 0; m < NUM_MEL_BANDS; m++) {
            for (int f = 0; f < NUM_TIME_FRAMES; f++) {
                float normalized = (mel_spectrogram_[m][f] - mean_db) / std_denom;
                float quantized = roundf(normalized / in_scale) + (float)in_zero_point;
                if (quantized < -128.0f) quantized = -128.0f;
                if (quantized > 127.0f) quantized = 127.0f;
                out_tensor_int8[out_idx++] = (int8_t)quantized;
            }
        }
    }

private:
    RFFT512 fft_;
    float frame_[FFT_LEN];
    float power_spectrum_[FFT_OUTPUT_BINS];
    float mel_spectrogram_[NUM_MEL_BANDS][NUM_TIME_FRAMES];
};

#endif // AUDIO_PROCESSOR_H_
