// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#pragma once
#include <openvino/openvino.hpp>

namespace cpp_stable_diffusion_ov
{
    class Scheduler
    {
    public:

        virtual void set_timesteps(size_t num_inference_steps) = 0;

        virtual size_t get_steps_offset() = 0;

        virtual ov::Tensor step(ov::Tensor model_output,
            double timestep,
            ov::Tensor sample) = 0;

        virtual ov::Tensor add_noise(ov::Tensor& original_samples,
            ov::Tensor& noise,
            double timesteps,
            bool return_as_noise = true) = 0;

        virtual void scale_model_input(ov::Tensor& sample, double t) {};

        virtual float init_noise_sigma() = 0;

        virtual std::vector<double> timesteps() = 0;

    };
}