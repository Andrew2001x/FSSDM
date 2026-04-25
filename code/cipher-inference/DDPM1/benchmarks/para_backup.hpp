#pragma once

#include <cstdint>

// Backup of the original DDPM defaults as of 2026-03-31.
namespace ddpm_para {

struct ModelConfig {
    std::uint64_t image_hw;
    std::uint64_t base_ch;
    std::uint64_t mid_ch;
    std::uint64_t temb_dim;
    std::uint64_t norm_groups;
};

struct IoConfig {
    std::uint64_t model_input_channels;
    std::uint64_t sample_channels;
    std::uint64_t cond_channels;
    std::uint64_t output_select_channel;
    std::uint64_t saved_image_channels;
    double noise_stddev;
};

struct RuntimeConfig {
    std::uint64_t steps;
    std::uint64_t num_timesteps;
    double strength;
    const char *beta_schedule;
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
    2,
    1,
    1,
    0,
    1,
    1.0,
};

constexpr RuntimeConfig kRuntimeConfig{
    5,
    1000,
    0.35,
    "linear",
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
