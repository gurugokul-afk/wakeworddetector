#include <stdio.h>
#include "esp_log.h"

static const char *TAG = "BANKAI_V4";

#ifdef __cplusplus
extern "C" {
#endif

extern const unsigned char bankai_v4_model[];
extern const unsigned int bankai_v4_model_len;

#ifdef __cplusplus
}
#endif

void app_main(void)
{
    ESP_LOGI(TAG, "BANKAI V4 ESP32 TEST");
    ESP_LOGI(TAG, "Model address: %p", bankai_v4_model);
    ESP_LOGI(TAG, "Model size: %u bytes", bankai_v4_model_len);
}
