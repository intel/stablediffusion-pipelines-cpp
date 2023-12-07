// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#pragma once

#include "pipelines/unet_loop.h"
#include <openvino/openvino.hpp>
#include <optional>

namespace cpp_stable_diffusion_ov
{
    class Scheduler;

    //'Standard' Stable Diffusion UNet loop that splits inference for positive and negative prompts 
    // (which adds the ability to run them across 2 separate devices)
    class UNetLoopSplit : public UNetLoop
    {
    public:

        UNetLoopSplit(std::string model_path,
            std::string device_unet_positive_prompt,
            std::string device_unet_negative_prompt,
            size_t max_tok_len, size_t img_width,
            size_t img_height,
            ov::Core &core);

        //Given timesteps and latents, produce output latents. 
        // todo: probably need some optional callback mechanism so that app can hook to UI progress bar, etc.
        virtual ov::Tensor operator()(const std::vector<double>& timesteps,
            ov::Tensor latents,
            ov::Tensor encoder_hidden_states_positive,
            ov::Tensor encoder_hidden_states_negative,
            float guidance_scale,
            std::shared_ptr<Scheduler> scheduler,
            std::optional<std::pair< CallbackFuncUnetIteration, void*>> unet_iteration_callback = {}) override;

    private:

        ov::Shape _sample_shape;
        ov::Shape _encoder_hidden_states_shape;
        ov::InferRequest _infer_request[2];
        std::string _device_unet_positive_prompt;
        std::string _device_unet_negative_prompt;

        void _run_unet_async(ov::Tensor latents, double timestep, ov::Tensor encoder_hidden_states, size_t infer_request_index);

    };
}