// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#pragma once

#include "cpp_stable_diffusion_ov/callbacks.h"
#include <optional>
#include <openvino/openvino.hpp>

namespace cpp_stable_diffusion_ov
{
    class Scheduler;

    //Base class for variations of unet loops
    class UNetLoop
    {
    public:

        //Given timesteps and latents, produce output latents. 
        virtual ov::Tensor operator()(const std::vector<double>& timesteps,
            ov::Tensor latents,
            ov::Tensor encoder_hidden_states_positive,
            ov::Tensor encoder_hidden_states_negative,
            float guidance_scale,
            std::shared_ptr<Scheduler> scheduler,
            std::optional<std::pair< CallbackFuncUnetIteration, void*>> unet_iteration_callback = {}) = 0;
    };
}