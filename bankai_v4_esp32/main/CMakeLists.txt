idf_component_register(
    SRCS "main.cc" "bankai_v4_model.cc"
    INCLUDE_DIRS "."
    REQUIRES espressif__esp-tflite-micro esp_driver_i2s esp_driver_gpio esp_timer
)
