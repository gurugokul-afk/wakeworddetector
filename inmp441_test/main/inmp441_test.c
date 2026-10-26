#include <stdio.h>
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2s_std.h"

#define I2S_BCLK_GPIO 4
#define I2S_WS_GPIO   5
#define I2S_DATA_GPIO 6

void app_main(void)
{
    i2s_chan_handle_t rx_handle;

    i2s_chan_config_t chan_cfg =
        I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);

    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_handle));

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(16000),
        .slot_cfg = {
    .data_bit_width = I2S_DATA_BIT_WIDTH_32BIT,
    .slot_bit_width = I2S_SLOT_BIT_WIDTH_32BIT,
    .slot_mode = I2S_SLOT_MODE_MONO,
    .slot_mask = I2S_STD_SLOT_LEFT,
    .ws_width = 32,
    .ws_pol = false,
    .bit_shift = true,
    .left_align = false,
    .big_endian = false,
    .bit_order_lsb = false,
},
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = I2S_BCLK_GPIO,
            .ws   = I2S_WS_GPIO,
            .dout = I2S_GPIO_UNUSED,
            .din  = I2S_DATA_GPIO,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false,
            },
        },
    };

    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_handle, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_handle));

    printf("\n");
    printf("====================================\n");
    printf("   INMP441 MICROPHONE TEST\n");
    printf("====================================\n");
    printf("BCLK : GPIO %d\n", I2S_BCLK_GPIO);
    printf("WS   : GPIO %d\n", I2S_WS_GPIO);
    printf("DATA : GPIO %d\n", I2S_DATA_GPIO);
    printf("Rate : 16000 Hz\n");
    printf("====================================\n\n");

    int32_t samples[256];
    size_t bytes_read;

    while (1)
    {
        ESP_ERROR_CHECK(
            i2s_channel_read(
                rx_handle,
                samples,
                sizeof(samples),
                &bytes_read,
                portMAX_DELAY
            )
        );

        int sample_count = bytes_read / sizeof(int32_t);

        int32_t min = samples[0];
        int32_t max = samples[0];

        for (int i = 1; i < sample_count; i++)
        {
            if (samples[i] < min)
                min = samples[i];

            if (samples[i] > max)
                max = samples[i];
        }

        printf("samples=%d  min=%ld  max=%ld\n",
               sample_count,
               (long)min,
               (long)max);
    }
}