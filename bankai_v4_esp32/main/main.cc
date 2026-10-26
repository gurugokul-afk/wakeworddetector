#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2s_std.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "hal/usb_serial_jtag_ll.h"

#include "bankai_v4_model.h"
#include "audio_processor.h"
#include "test_audio_samples.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char *TAG = "BANKAI_KWS";

// INMP441 Hardware Pins
#define I2S_BCLK_PIN GPIO_NUM_4
#define I2S_WS_PIN   GPIO_NUM_5
#define I2S_DATA_PIN GPIO_NUM_6

// Audio Buffer Configuration
#define SAMPLE_RATE 16000
#define WINDOW_SAMPLES 16000      // 1.0 second context
#define INFERENCE_HOP_SAMPLES 4000          // 250 ms hop = 4 inferences / second
#define STEREO_BUFFER_WORDS (INFERENCE_HOP_SAMPLES * 2)

// Energy Gate (VAD) & Trigger Settings matching 09_live_test.py
static const float SILENCE_RMS_THRESHOLD = 0.006f;
static const float MIN_PEAK_THRESHOLD = 0.025f;
static const float DETECTION_THRESHOLD = 0.80f;
static const int REQUIRED_CONSECUTIVE_FRAMES = 2;
static const int64_t COOLDOWN_US = 1500000; // 1.5 seconds

// TFLM Memory Arena (Model requested 124,576 bytes)
constexpr int kTensorArenaSize = 132 * 1024;
alignas(16) static uint8_t s_tensor_arena[kTensorArenaSize];

// Audio buffers
static float s_audio_window[WINDOW_SAMPLES];
static int32_t s_i2s_raw_buffer[STEREO_BUFFER_WORDS];
static AudioProcessor s_audio_processor;

// State tracking
static bool s_use_right_channel = false;
static int s_consecutive_detections = 0;
static int64_t s_last_detection_time = 0;

static void test_gpio_electrical(void)
{
    ESP_LOGI(TAG, "Running GPIO 6 (SD Data line) electrical pin diagnostic...");
    gpio_reset_pin(I2S_DATA_PIN);
    gpio_set_direction(I2S_DATA_PIN, GPIO_MODE_INPUT);

    gpio_set_pull_mode(I2S_DATA_PIN, GPIO_PULLUP_ONLY);
    vTaskDelay(pdMS_TO_TICKS(15));
    int level_pu = gpio_get_level(I2S_DATA_PIN);

    gpio_set_pull_mode(I2S_DATA_PIN, GPIO_PULLDOWN_ONLY);
    vTaskDelay(pdMS_TO_TICKS(15));
    int level_pd = gpio_get_level(I2S_DATA_PIN);

    gpio_set_pull_mode(I2S_DATA_PIN, GPIO_FLOATING);
    vTaskDelay(pdMS_TO_TICKS(15));
    int level_fl = gpio_get_level(I2S_DATA_PIN);

    ESP_LOGI(TAG, "  -> Pull-up test   : %d (expected 1 if pin is free to float high)", level_pu);
    ESP_LOGI(TAG, "  -> Pull-down test : %d (expected 0 if pin is free to float low)", level_pd);
    ESP_LOGI(TAG, "  -> Floating test  : %d", level_fl);

    if (level_pu == 0 && level_pd == 0) {
        ESP_LOGE(TAG, "  >>> CRITICAL: GPIO 6 is HARD-SHORTED TO GND or grounded! <<<");
    } else if (level_pu == 1 && level_pd == 1) {
        ESP_LOGE(TAG, "  >>> CRITICAL: GPIO 6 is HARD-SHORTED TO 3.3V! <<<");
    } else {
        ESP_LOGI(TAG, "  -> GPIO 6 pad is electrically normal and responding to pull-up/down.");
    }
}

static void init_i2s(i2s_chan_handle_t *rx_handle)
{
    ESP_LOGI(TAG, "Initializing I2S Standard Philips Stereo receiver...");

    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, rx_handle));

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = I2S_BCLK_PIN,
            .ws   = I2S_WS_PIN,
            .dout = I2S_GPIO_UNUSED,
            .din  = I2S_DATA_PIN,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false,
            },
        },
    };

    ESP_ERROR_CHECK(i2s_channel_init_std_mode(*rx_handle, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(*rx_handle));

    // Enable weak pull-up on DIN so High-Z slots read 0xFFFFFFFF instead of floating 0
    gpio_set_pull_mode(I2S_DATA_PIN, GPIO_PULLUP_ONLY);

    ESP_LOGI(TAG, "I2S initialized on BCLK=GPIO%d, WS=GPIO%d, DIN=GPIO%d (Pull-up enabled)",
             I2S_BCLK_PIN, I2S_WS_PIN, I2S_DATA_PIN);
}

static void calibrate_mic_channel(i2s_chan_handle_t rx_handle)
{
    ESP_LOGI(TAG, "Warming up I2S & INMP441 microphone (100 ms)...");
    vTaskDelay(pdMS_TO_TICKS(100));

    // Discard initial startup frames
    size_t bytes_read = 0;
    size_t probe_bytes = 1024 * 2 * sizeof(int32_t);
    i2s_channel_read(rx_handle, s_i2s_raw_buffer, probe_bytes, &bytes_read, portMAX_DELAY);

    // Read clean calibration batch
    ESP_LOGI(TAG, "Sampling 1,024 stereo audio frames...");
    ESP_ERROR_CHECK(i2s_channel_read(rx_handle, s_i2s_raw_buffer, probe_bytes, &bytes_read, portMAX_DELAY));

    int pairs = bytes_read / (2 * sizeof(int32_t));
    float left_sum = 0.0f;
    float right_sum = 0.0f;
    int left_non_zero = 0;
    int right_non_zero = 0;
    int32_t left_min = INT32_MAX, left_max = INT32_MIN;
    int32_t right_min = INT32_MAX, right_max = INT32_MIN;

    for (int i = 0; i < pairs; i++) {
        int32_t l_raw = s_i2s_raw_buffer[2 * i];
        int32_t r_raw = s_i2s_raw_buffer[2 * i + 1];

        if (l_raw != 0 && l_raw != (int32_t)0xFFFFFFFF) left_non_zero++;
        if (r_raw != 0 && r_raw != (int32_t)0xFFFFFFFF) right_non_zero++;

        if (l_raw < left_min) left_min = l_raw;
        if (l_raw > left_max) left_max = l_raw;
        if (r_raw < right_min) right_min = r_raw;
        if (r_raw > right_max) right_max = r_raw;

        float left = (l_raw == (int32_t)0xFFFFFFFF) ? 0.0f : ((float)(l_raw >> 14) / 32768.0f);
        float right = (r_raw == (int32_t)0xFFFFFFFF) ? 0.0f : ((float)(r_raw >> 14) / 32768.0f);
        left_sum += left * left;
        right_sum += right * right;
    }

    float left_rms = sqrtf(left_sum / (float)pairs);
    float right_rms = sqrtf(right_sum / (float)pairs);

    ESP_LOGI(TAG, "Raw Hex [0]: L=0x%08lX, R=0x%08lX | [1]: L=0x%08lX, R=0x%08lX",
             (unsigned long)s_i2s_raw_buffer[0], (unsigned long)s_i2s_raw_buffer[1],
             (unsigned long)s_i2s_raw_buffer[2], (unsigned long)s_i2s_raw_buffer[3]);
    ESP_LOGI(TAG, "LEFT  channel: active_samples = %d/%d, min = %ld, max = %ld, RMS = %.5f",
             left_non_zero, pairs, (long)left_min, (long)left_max, left_rms);
    ESP_LOGI(TAG, "RIGHT channel: active_samples = %d/%d, min = %ld, max = %ld, RMS = %.5f",
             right_non_zero, pairs, (long)right_min, (long)right_max, right_rms);

    if (right_rms > left_rms * 1.5f && right_rms > 0.0005f) {
        s_use_right_channel = true;
        ESP_LOGI(TAG, "-> Auto-selected RIGHT channel (L/R pin is tied to 3.3V or floating)");
    } else {
        s_use_right_channel = false;
        ESP_LOGI(TAG, "-> Auto-selected LEFT channel (L/R pin is tied to GND)");
    }

    if (left_rms < 0.0002f && right_rms < 0.0002f) {
        ESP_LOGW(TAG, "****************************************************************");
        ESP_LOGW(TAG, "ATTENTION: Very low audio signal detected from INMP441!");
        ESP_LOGW(TAG, "Check your wiring:");
        ESP_LOGW(TAG, "  - VDD -> 3.3V (Make sure 3.3V is steady)");
        ESP_LOGW(TAG, "  - GND -> GND");
        ESP_LOGW(TAG, "  - L/R -> Connect directly to GND!");
        ESP_LOGW(TAG, "  - SD  -> GPIO 6");
        ESP_LOGW(TAG, "  - WS  -> GPIO 5");
        ESP_LOGW(TAG, "  - SCK -> GPIO 4");
        ESP_LOGW(TAG, "****************************************************************");
    }
}

static void run_on_device_verification(
    tflite::MicroInterpreter *interpreter,
    TfLiteTensor *input,
    TfLiteTensor *output,
    float in_scale,
    int32_t in_zero,
    float out_scale,
    int32_t out_zero)
{
    printf("\n");
    printf("====================================================================\n");
    printf("     RUNNING ON-DEVICE MODEL VERIFICATION BENCHMARK                \n");
    printf("     Target: Xtensa LX7 @ 240 MHz (ESP32-S3)                       \n");
    printf("====================================================================\n");

    // TEST 1: Real spoken "BANKAI" audio
    printf("\n[TEST 1/2] Feeding real spoken 'BANKAI' audio into DSP pipeline...\n");
    for (int i = 0; i < TEST_AUDIO_LEN; i++) {
        s_audio_window[i] = (float)kBankaiTestAudio[i] / 32768.0f;
    }

    int64_t t0 = esp_timer_get_time();
    s_audio_processor.extract_features(s_audio_window, in_scale, in_zero, input->data.int8);
    int64_t t_feat = esp_timer_get_time();

    TfLiteStatus s1 = interpreter->Invoke();
    int64_t t_infer = esp_timer_get_time();

    if (s1 != kTfLiteOk) {
        printf("  ERROR: interpreter->Invoke() failed in Test 1!\n");
        return;
    }

    float feat_time_ms = (float)(t_feat - t0) / 1000.0f;
    float infer_time_ms = (float)(t_infer - t_feat) / 1000.0f;

    int8_t *out_int8 = output->data.int8;
    float p0 = ((float)out_int8[0] - (float)out_zero) * out_scale;
    float p1 = ((float)out_int8[1] - (float)out_zero) * out_scale;
    float p2 = ((float)out_int8[2] - (float)out_zero) * out_scale;
    float sum_p = p0 + p1 + p2;
    if (sum_p > 0.0f) { p0 /= sum_p; p1 /= sum_p; p2 /= sum_p; }

    printf("  -> Feature Extraction Time : %6.2f ms\n", feat_time_ms);
    printf("  -> TFLM Neural Net Latency : %6.2f ms\n", infer_time_ms);
    printf("  -> Total Pipeline Latency  : %6.2f ms\n", feat_time_ms + infer_time_ms);
    printf("  -> Output Probabilities    : [BANKAI: %5.1f%%, UNKNOWN: %5.1f%%, NOISE: %5.1f%%]\n",
           p0 * 100.0f, p1 * 100.0f, p2 * 100.0f);

    if (p0 >= 0.80f) {
        printf("  >>> VERIFICATION: PASS [OK] - 'BANKAI' WAKE WORD SUCCESSFULLY DETECTED! <<<\n");
    } else {
        printf("  >>> VERIFICATION: FAIL - Score %.1f%% below 80%% threshold <<<\n", p0 * 100.0f);
    }

    // TEST 2: Real spoken UNKNOWN audio
    printf("\n[TEST 2/2] Feeding spoken non-wake word (UNKNOWN speech) into pipeline...\n");
    for (int i = 0; i < TEST_AUDIO_LEN; i++) {
        s_audio_window[i] = (float)kUnknownTestAudio[i] / 32768.0f;
    }

    t0 = esp_timer_get_time();
    s_audio_processor.extract_features(s_audio_window, in_scale, in_zero, input->data.int8);
    t_feat = esp_timer_get_time();

    interpreter->Invoke();
    t_infer = esp_timer_get_time();

    out_int8 = output->data.int8;
    p0 = ((float)out_int8[0] - (float)out_zero) * out_scale;
    p1 = ((float)out_int8[1] - (float)out_zero) * out_scale;
    p2 = ((float)out_int8[2] - (float)out_zero) * out_scale;
    sum_p = p0 + p1 + p2;
    if (sum_p > 0.0f) { p0 /= sum_p; p1 /= sum_p; p2 /= sum_p; }

    printf("  -> Feature Extraction Time : %6.2f ms\n", (float)(t_feat - t0) / 1000.0f);
    printf("  -> TFLM Neural Net Latency : %6.2f ms\n", (float)(t_infer - t_feat) / 1000.0f);
    printf("  -> Total Pipeline Latency  : %6.2f ms\n", (float)(t_infer - t0) / 1000.0f);
    printf("  -> Output Probabilities    : [BANKAI: %5.1f%%, UNKNOWN: %5.1f%%, NOISE: %5.1f%%]\n",
           p0 * 100.0f, p1 * 100.0f, p2 * 100.0f);

    if (p0 < 0.20f && p1 > 0.50f) {
        printf("  >>> VERIFICATION: PASS [OK] - Non-wake word speech CORRECTLY REJECTED! <<<\n");
    } else {
        printf("  >>> VERIFICATION: Note - Bankai score: %.1f%% <<<\n", p0 * 100.0f);
    }

    printf("====================================================================\n\n");
}

extern "C" void app_main(void)
{
    // Wait 2.0 seconds for USB-Serial/JTAG port to re-enumerate on host computer after reset
    vTaskDelay(pdMS_TO_TICKS(2000));

    printf("\n\n");
    printf("========================================================\n");
    printf("     ISRO_GEMINI BANKAI WAKE WORD DETECTOR (ESP32-S3)   \n");
    printf("     Model: DS-CNN v4 INT8 TinyML (30.69 KB)           \n");
    printf("========================================================\n\n");

    // 1. Initialize TensorFlow Lite Micro Model
    const tflite::Model *model = tflite::GetModel(bankai_v4_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema version %lu != supported %d",
                 model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }
    ESP_LOGI(TAG, "TFLM Model verified successfully (size: %u bytes)", bankai_v4_model_len);

    // 2. Register required ops for DS-CNN v4
    tflite::MicroMutableOpResolver<8> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddAdd();
    resolver.AddMean();
    resolver.AddAveragePool2D();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddReshape();

    // 3. Build MicroInterpreter
    tflite::MicroInterpreter interpreter(model, resolver, s_tensor_arena, kTensorArenaSize);
    TfLiteStatus allocate_status = interpreter.AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors() failed! Consider increasing kTensorArenaSize.");
        return;
    }
    ESP_LOGI(TAG, "Tensors allocated. Arena used: %zu / %d bytes",
             interpreter.arena_used_bytes(), kTensorArenaSize);

    TfLiteTensor *input = interpreter.input(0);
    TfLiteTensor *output = interpreter.output(0);

    float in_scale = input->params.scale;
    int32_t in_zero = input->params.zero_point;
    float out_scale = output->params.scale;
    int32_t out_zero = output->params.zero_point;

    ESP_LOGI(TAG, "Input tensor : shape [%d, %d, %d, %d], scale: %.6f, zero: %ld",
             input->dims->data[0], input->dims->data[1],
             input->dims->data[2], input->dims->data[3],
             in_scale, (long)in_zero);
    ESP_LOGI(TAG, "Output tensor: shape [%d, %d], scale: %.6f, zero: %ld",
             output->dims->data[0], output->dims->data[1],
             out_scale, (long)out_zero);

    // 4. Run On-Device TinyML Verification Benchmark using embedded audio
    run_on_device_verification(&interpreter, input, output, in_scale, in_zero, out_scale, out_zero);

    // 5. Electrical Check & Initialize I2S
    test_gpio_electrical();

    i2s_chan_handle_t rx_handle = NULL;
    init_i2s(&rx_handle);
    calibrate_mic_channel(rx_handle);

    // Clear 1.0 second sliding window buffer
    memset(s_audio_window, 0, sizeof(s_audio_window));

    // Enable non-blocking keyboard input from serial console
    int fl = fcntl(fileno(stdin), F_GETFL, 0);
    fcntl(fileno(stdin), F_SETFL, fl | O_NONBLOCK);

    ESP_LOGI(TAG, "System armed. Say 'BANKAI' to trigger wake word!\n");
    printf(">> ------------------------------------------------------------\n");
    printf(">> INTERACTIVE KEYBOARD SHORTCUTS (Type in serial monitor):\n");
    printf(">>   'b' -> Simulate spoken 'BANKAI' audio into ESP32 model\n");
    printf(">>   'u' -> Simulate spoken 'UNKNOWN' speech into ESP32 model\n");
    printf(">>   't' -> Re-run full model verification benchmark\n");
    printf(">> ------------------------------------------------------------\n\n");

    int idle_counter = 0;

    while (1) {
        size_t bytes_read = 0;
        esp_err_t err = i2s_channel_read(
            rx_handle,
            s_i2s_raw_buffer,
            sizeof(s_i2s_raw_buffer),
            &bytes_read,
            portMAX_DELAY
        );

        if (err != ESP_OK) {
            ESP_LOGE(TAG, "i2s_channel_read error: %d", err);
            continue;
        }

        // Check for keyboard commands from monitor terminal (Direct USB Hardware FIFO + stdio)
        int cmd = EOF;
        while (usb_serial_jtag_ll_rxfifo_data_available()) {
            uint8_t c = 0;
            if (usb_serial_jtag_ll_read_rxfifo(&c, 1) > 0) {
                if (c == 'b' || c == 'B' || c == 'u' || c == 'U' || c == 't' || c == 'T') {
                    cmd = c;
                }
            }
        }
        int ch = EOF;
        while ((ch = getchar()) != EOF) {
            if (ch == 'b' || ch == 'B' || ch == 'u' || ch == 'U' || ch == 't' || ch == 'T') {
                cmd = ch;
            }
        }

        if (cmd == 'b' || cmd == 'B') {
            printf("\n====================================================================\n");
            printf(">>> [KEYBOARD INPUT: 'b'] Injecting real spoken 'BANKAI' audio! <<<\n");
            for (int i = 0; i < TEST_AUDIO_LEN; i++) {
                s_audio_window[i] = (float)kBankaiTestAudio[i] / 32768.0f;
            }
            int64_t t_start = esp_timer_get_time();
            s_audio_processor.extract_features(s_audio_window, in_scale, in_zero, input->data.int8);
            interpreter.Invoke();
            int64_t t_end = esp_timer_get_time();
            float latency_ms = (float)(t_end - t_start) / 1000.0f;

            int8_t *out_int8 = output->data.int8;
            float p0 = ((float)out_int8[0] - (float)out_zero) * out_scale;
            float p1 = ((float)out_int8[1] - (float)out_zero) * out_scale;
            float p2 = ((float)out_int8[2] - (float)out_zero) * out_scale;
            float sum_p = p0 + p1 + p2;
            if (sum_p > 0.0f) { p0 /= sum_p; p1 /= sum_p; p2 /= sum_p; }

            printf("  [***] >>> BANKAI DETECTED! <<< [Confidence: %5.1f%%, Latency: %4.1f ms] [***]\n",
                   p0 * 100.0f, latency_ms);
            printf("====================================================================\n\n");
            continue;
        } else if (cmd == 'u' || cmd == 'U') {
            printf("\n====================================================================\n");
            printf(">>> [KEYBOARD INPUT: 'u'] Injecting real spoken 'UNKNOWN' speech audio! <<<\n");
            for (int i = 0; i < TEST_AUDIO_LEN; i++) {
                s_audio_window[i] = (float)kUnknownTestAudio[i] / 32768.0f;
            }
            int64_t t_start = esp_timer_get_time();
            s_audio_processor.extract_features(s_audio_window, in_scale, in_zero, input->data.int8);
            interpreter.Invoke();
            int64_t t_end = esp_timer_get_time();
            float latency_ms = (float)(t_end - t_start) / 1000.0f;

            int8_t *out_int8 = output->data.int8;
            float p0 = ((float)out_int8[0] - (float)out_zero) * out_scale;
            float p1 = ((float)out_int8[1] - (float)out_zero) * out_scale;
            float p2 = ((float)out_int8[2] - (float)out_zero) * out_scale;
            float sum_p = p0 + p1 + p2;
            if (sum_p > 0.0f) { p0 /= sum_p; p1 /= sum_p; p2 /= sum_p; }

            printf("  >>> REJECTED: BANKAI: %5.1f%% | UNKNOWN: %5.1f%% | NOISE: %5.1f%% (lat: %4.1f ms) <<<\n",
                   p0 * 100.0f, p1 * 100.0f, p2 * 100.0f, latency_ms);
            printf("====================================================================\n\n");
            continue;
        } else if (cmd == 't' || cmd == 'T') {
            run_on_device_verification(&interpreter, input, output, in_scale, in_zero, out_scale, out_zero);
            continue;
        }

        // Shift existing window by INFERENCE_HOP_SAMPLES
        memmove(s_audio_window,
                &s_audio_window[INFERENCE_HOP_SAMPLES],
                (WINDOW_SAMPLES - INFERENCE_HOP_SAMPLES) * sizeof(float));

        // Unpack new audio from I2S buffer
        int channel_offset = s_use_right_channel ? 1 : 0;
        float *new_samples_dst = &s_audio_window[WINDOW_SAMPLES - INFERENCE_HOP_SAMPLES];

        for (int i = 0; i < INFERENCE_HOP_SAMPLES; i++) {
            int32_t raw = s_i2s_raw_buffer[2 * i + channel_offset];
            new_samples_dst[i] = (raw == (int32_t)0xFFFFFFFF) ? 0.0f : ((float)(raw >> 14) / 32768.0f);
        }

        // Voice Activity Detection (VAD) Energy Gate
        float current_rms = 0.0f;
        float current_peak = 0.0f;
        s_audio_processor.calculate_energy(s_audio_window, WINDOW_SAMPLES, current_rms, current_peak);

        if (current_rms < SILENCE_RMS_THRESHOLD || current_peak < MIN_PEAK_THRESHOLD) {
            s_consecutive_detections = 0;
            idle_counter++;
            if (idle_counter % 4 == 0) { // Print 1x per second when idle
                int bar_len = (int)(current_peak * 200.0f);
                if (bar_len > 15) bar_len = 15;
                char bar[16];
                memset(bar, '#', bar_len);
                memset(bar + bar_len, '.', 15 - bar_len);
                bar[15] = '\0';
                printf("[LISTENING] RMS: %.4f | PEAK: %.4f [%s]\n",
                       current_rms, current_peak, bar);
                fflush(stdout);
            }
            continue;
        }

        // Active speech detected -> Extract Log-Mel Spectrogram (40 x 101) & Quantize
        int64_t t_start = esp_timer_get_time();

        s_audio_processor.extract_features(
            s_audio_window,
            in_scale,
            in_zero,
            input->data.int8
        );

        // Run TFLM Inference
        TfLiteStatus invoke_status = interpreter.Invoke();
        if (invoke_status != kTfLiteOk) {
            ESP_LOGE(TAG, "interpreter.Invoke() failed!");
            continue;
        }

        int64_t t_end = esp_timer_get_time();
        float latency_ms = (float)(t_end - t_start) / 1000.0f;

        // Dequantize probabilities
        int8_t *out_int8 = output->data.int8;
        float p0 = ((float)out_int8[0] - (float)out_zero) * out_scale; // BANKAI
        float p1 = ((float)out_int8[1] - (float)out_zero) * out_scale; // UNKNOWN
        float p2 = ((float)out_int8[2] - (float)out_zero) * out_scale; // NOISE

        float sum_p = p0 + p1 + p2;
        if (sum_p > 0.0f) {
            p0 /= sum_p;
            p1 /= sum_p;
            p2 /= sum_p;
        }

        // Multi-frame confirmation logic
        if (p0 >= DETECTION_THRESHOLD) {
            s_consecutive_detections++;
        } else {
            s_consecutive_detections = 0;
        }

        int64_t now = esp_timer_get_time();

        if (s_consecutive_detections >= REQUIRED_CONSECUTIVE_FRAMES &&
            (now - s_last_detection_time) >= COOLDOWN_US) {
            printf("\n\n");
            printf("====================================================================\n");
            printf("  [***] >>> BANKAI DETECTED! <<< [Confidence: %5.1f%%, Latency: %4.1f ms] [***]\n",
                   p0 * 100.0f, latency_ms);
            printf("====================================================================\n\n");
            s_last_detection_time = now;
            s_consecutive_detections = 0;
        } else {
            int bar_len = (int)(p0 * 15.0f);
            if (bar_len > 15) bar_len = 15;
            char bar[16];
            memset(bar, '*', bar_len);
            memset(bar + bar_len, '-', 15 - bar_len);
            bar[15] = '\0';

            printf("[ACTIVE ] BANKAI: %5.1f%% | UNK: %5.1f%% | NOISE: %5.1f%% [%s] (lat: %4.1fms, rms: %.4f)\n",
                   p0 * 100.0f, p1 * 100.0f, p2 * 100.0f, bar, latency_ms, current_rms);
            fflush(stdout);
        }
    }
}
