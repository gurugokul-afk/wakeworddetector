#ifndef RFFT_512_H_
#define RFFT_512_H_

#include <math.h>
#include <stdint.h>
#include <string.h>

#define FFT_LEN 512
#define FFT_HALF 256
#define FFT_OUTPUT_BINS 257

class RFFT512 {
public:
    RFFT512() {
        // Precompute twiddle factors and bit-reversal table
        for (int i = 0; i < FFT_LEN; i++) {
            int j = 0;
            for (int bit = 0; bit < 9; bit++) {
                if (i & (1 << bit)) {
                    j |= (1 << (8 - bit));
                }
            }
            bit_rev_[i] = (uint16_t)j;
        }

        for (int i = 0; i < FFT_HALF; i++) {
            float angle = -2.0f * (float)M_PI * (float)i / (float)FFT_LEN;
            sin_table_[i] = sinf(angle);
            cos_table_[i] = cosf(angle);
        }
    }

    // in: 512 float samples
    // out_power: 257 float power spectrum values (re^2 + im^2)
    void compute_power(const float* in, float* out_power) {
        // Bit-reversal permutation
        for (int i = 0; i < FFT_LEN; i++) {
            re_[bit_rev_[i]] = in[i];
            im_[bit_rev_[i]] = 0.0f;
        }

        // Cooley-Tukey Radix-2
        for (int len = 2; len <= FFT_LEN; len <<= 1) {
            int half_len = len >> 1;
            int step = FFT_LEN / len;

            for (int i = 0; i < FFT_LEN; i += len) {
                int table_idx = 0;
                for (int k = 0; k < half_len; k++) {
                    float w_re = cos_table_[table_idx];
                    float w_im = sin_table_[table_idx];
                    table_idx += step;

                    int u_idx = i + k;
                    int v_idx = u_idx + half_len;

                    float v_re = re_[v_idx] * w_re - im_[v_idx] * w_im;
                    float v_im = re_[v_idx] * w_im + im_[v_idx] * w_re;

                    float u_re = re_[u_idx];
                    float u_im = im_[u_idx];

                    re_[u_idx] = u_re + v_re;
                    im_[u_idx] = u_im + v_im;
                    re_[v_idx] = u_re - v_re;
                    im_[v_idx] = u_im - v_im;
                }
            }
        }

        // Power spectrum for bins 0 to 256
        for (int k = 0; k < FFT_OUTPUT_BINS; k++) {
            out_power[k] = re_[k] * re_[k] + im_[k] * im_[k];
        }
    }

private:
    float re_[FFT_LEN];
    float im_[FFT_LEN];
    uint16_t bit_rev_[FFT_LEN];
    float cos_table_[FFT_HALF];
    float sin_table_[FFT_HALF];
};

#endif // RFFT_512_H_
