// gpu_telemetry.h: GPU Telemetry Interface
#pragma once

#include <string>

struct GPUTelemetrySample {
    double timestamp_s = 0.0;
    double delta_s = 0.0;
    double graphics_frequency_hz = 0.0;
    double memory_frequency_hz = 0.0;
    double amplitude_norm = 0.0;
    double voltage_v = 0.0;
    double amperage_a = 0.0;
    double power_w = 0.0;
    double temperature_c = 0.0;
    double gpu_util_norm = 0.0;
    double memory_util_norm = 0.0;
    double thermal_interference_norm = 0.0;
    bool live = false;
    std::string provider;
};

class GPUTelemetry {
public:
    GPUTelemetry();
    GPUTelemetrySample poll();

private:
    GPUTelemetrySample _last;
    bool _has_last;
    double _phase_seed;

    bool try_poll_nvidia_smi(GPUTelemetrySample& sample);
    GPUTelemetrySample build_proxy_sample(double now_s);

    static double now_seconds();
    static double clamp01(double value);
};
