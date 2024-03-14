// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#pragma once
#include "cpp_stable_diffusion_ov/stable_diffusion_interpolation_pipeline.h"
#include "cpp_stable_diffusion_audio_ov/visibility.h"

namespace cpp_stable_diffusion_ov
{
    class CPP_SD_OV_AUDIO_API StableDiffusionAudioInterpolationPipeline : public StableDiffusionInterpolationPipeline
    {
    public:

        StableDiffusionAudioInterpolationPipeline(std::string model_folder,
            std::optional< std::string > unet_subdir,
            std::optional< std::string > cache = {},
            std::string text_encoder_device = "CPU",
            std::string unet_positive_device = "CPU",
            std::string unet_negative_device = "CPU",
            std::string vae_decoder_device = "CPU",
            std::string vae_encoder_device = "CPU");

        size_t EstimateTotalUnetIterations(
            float denoising_start = 0.75f,
            float denoising_end = 0.75f,
            int num_interpolation_steps = 5,
            int num_inference_steps_per_sample = 50,
            const std::string& scheduler_str = "EulerDiscreteScheduler",
            std::optional<size_t> num_output_segments = {});

        typedef bool (*CallbackFuncInterpolationIteration)(size_t interp_step_i_complete,
            size_t num_interp_steps,
            std::shared_ptr<std::vector<float>> wav,
            std::shared_ptr<std::vector<uint8_t>> img_rgb,
            size_t img_width,
            size_t img_height,
            void* user);

        std::pair< std::shared_ptr<std::vector<float>>, std::shared_ptr<std::vector<float>> > operator()(
            bool bStereo,
            const std::string prompt_start,
            std::optional< std::string > negative_prompt_start = {},
            std::optional< std::string > prompt_end = {},
            std::optional< std::string > negative_prompt_ebd = {},
            std::optional< unsigned int > seed_start = {},
            std::optional< unsigned int > seed_end = {},
            float denoising_start = 0.75f,
            float denoising_end = 0.75f,
            float guidance_scale_start = 7.5f,
            float guidance_scale_end = 7.5f,
            int num_inference_steps_per_sample = 50,
            int num_interpolation_steps = 5,
            std::optional<size_t> num_output_segments = {},
            std::string seed_image = "og_beat",
            float alpha_power = 1.0f,
            const std::string& scheduler_str = "EulerDiscreteScheduler",
            std::optional<std::pair< CallbackFuncUnetIteration, void*>> unet_iteration_callback = {},
            std::optional<std::pair< CallbackFuncInterpolationIteration, void*>> interp_iteration_callback = {}
        );

    };
}