#pragma once

#include <cstdint>

// Centralized DDPM defaults used by secure inference.
// Edit this file for experiments. The original values are preserved in para_backup.hpp.
namespace ddpm_para {

struct ModelConfig {
    std::uint64_t image_hw;
    std::uint64_t base_ch;
    std::uint64_t mid_ch;
    std::uint64_t temb_dim;
    std::uint64_t norm_groups;
};

// The t2i secure inference path assumes single-channel sample/output images and no cond image.
struct IoConfig {
    std::uint64_t model_input_channels;
    std::uint64_t sample_channels;
    std::uint64_t cond_channels;
    std::uint64_t output_select_channel;
    std::uint64_t saved_image_channels;
    double noise_stddev;
};

struct SchedulerConfig {
    std::uint64_t num_train_timesteps;
    const char *beta_schedule;
    const char *prediction_type;
    bool clip_sample;
    double clip_sample_range;
    bool thresholding;
    double dynamic_thresholding_ratio;
    double sample_max_value;
    const char *variance_type;
    const char *timestep_spacing;
};

struct RuntimeConfig {
    std::uint64_t steps;
    double strength;
    const char *image_path;
    const char *out_path;
    const char *cond_out_path;
};

constexpr std::uint64_t kMidChannelsFromChannelsMultiplier = 2;
constexpr std::uint64_t kTembFromChannelsMultiplier = 4;

constexpr ModelConfig kModelConfig{
    28,
    16,
    32,
    64,
    8,
};

constexpr IoConfig kIoConfig{
    1,
    1,
    0,
    0,
    1,
    1.0,
};

constexpr SchedulerConfig kSchedulerConfig{
    1000,
    "linear",
    "epsilon",
    false,
    1.0,
    false,
    0.995,
    1.0,
    "fixed_small",
    "uniform",
};

constexpr RuntimeConfig kRuntimeConfig{
    5,
    0.0,
    "",
    "ddpm_out.jpg",
    "ddpm_cond.jpg",
};

constexpr std::uint64_t mid_channels_from_channels(std::uint64_t channels) {
    return channels * kMidChannelsFromChannelsMultiplier;
}

constexpr std::uint64_t temb_from_channels(std::uint64_t channels) {
    return channels * kTembFromChannelsMultiplier;
}

}  // namespace ddpm_para
