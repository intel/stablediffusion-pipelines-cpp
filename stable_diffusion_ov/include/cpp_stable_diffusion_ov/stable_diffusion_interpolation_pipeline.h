// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#pragma once
#include "cpp_stable_diffusion_ov/stable_diffusion_pipeline.h"
#include "cpp_stable_diffusion_ov/visibility.h"

namespace cpp_stable_diffusion_ov
{
    class CPP_SD_OV_API StableDiffusionInterpolationPipeline : public StableDiffusionPipeline
    {
    public:

        StableDiffusionInterpolationPipeline(std::string model_folder,
            std::optional< std::string > cache = {},
            std::string text_encoder_device = "CPU",
            std::string unet_positive_device = "CPU",
            std::string unet_negative_device = "CPU",
            std::string vae_decoder_device = "CPU",
            std::string vae_encoder_device = "CPU");

        std::vector<std::shared_ptr<std::vector<uint8_t>>> operator()(
            const std::string start_prompt,
            std::optional< std::string > end_prompt = {},
            std::optional< std::string > negative_prompt = {},
            std::optional< unsigned int > seed_start = {},
            std::optional< unsigned int > seed_end = {},
            float guidance_scale_start = 7.5f,
            float guidance_scale_end = 7.5f,
            int num_inference_steps = 25,
            int num_interpolation_steps = 5,
            const std::string& scheduler_str = "EulerDiscreteScheduler",
            bool bGiveBGR = false,
            std::optional< InputImageParams > input_image_params = {},
            std::optional<std::pair< CallbackFuncUnetIteration, void*>> unet_iteration_callback = {}
        );

        std::shared_ptr<std::vector<uint8_t>> run_single_alpha(
            const std::string start_prompt,
            const std::string end_prompt,
            std::optional< std::string > negative_prompt,
            float alpha,
            int num_inference_steps,
            const std::string& scheduler_str,
            std::optional< unsigned int > seed_start,
            std::optional< unsigned int > seed_end,
            float guidance_scale,
            bool bGiveBGR,
            std::optional< InputImageParams > input_image_params);


    protected:

        std::shared_ptr<std::vector<uint8_t>> _run_alpha(
            std::optional< ov::Tensor > vae_encoded,
            ov::Tensor pos_prompt_embeds,
            ov::Tensor neg_prompt_embeds,
            std::shared_ptr<Scheduler> scheduler,
            int num_inference_steps,
            std::optional< unsigned int > seed_start,
            std::optional< unsigned int > seed_end,
            float guidance_scale,
            float strength,
            float alpha,
            bool bGiveBGR,
            std::optional<std::pair< CallbackFuncUnetIteration, void*>> unet_iteration_callback = {});

        ov::Tensor _prepare_latents_alpha(std::optional< ov::Tensor > vae_encoded,
            double latent_timestep,
            std::shared_ptr<Scheduler> scheduler,
            std::optional< unsigned int > seed_start,
            std::optional< unsigned int > seed_end,
            float alpha);

    };
}