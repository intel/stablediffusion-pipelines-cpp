// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#pragma once
#include "schedulers/scheduler.h"
#include <openvino/openvino.hpp>
#include <string>
#include <optional>
#include <vector>

namespace cpp_stable_diffusion_ov
{
    class RNG_G;
    class EulerDiscreteScheduler : public Scheduler
    {
    public:

        EulerDiscreteScheduler(size_t num_train_timesteps = 1000,
            float beta_start = 0.0001f,
            float beta_end = 0.02f,
            std::string beta_schedule = "linear",
            std::optional< std::vector<float> > trained_betas = {},
            std::string prediction_type = "epsilon",
            std::string interpolation_type = "linear",
            std::optional<bool> use_karras_sigmas = false,
            std::string timestep_spacing = "linspace",
            size_t steps_offset = 0);

        virtual void set_timesteps(size_t num_inference_steps) override;

        virtual size_t get_steps_offset() override { return steps_offset; };

        virtual ov::Tensor step(ov::Tensor model_output,
            double timestep,
            ov::Tensor sample) override;

        virtual ov::Tensor add_noise(ov::Tensor& original_samples,
            ov::Tensor& noise,
            double timesteps,
            bool return_as_noise = true) override;

        virtual float init_noise_sigma() override;

        virtual std::vector<double> timesteps() override {
            return _timesteps;
        };

        virtual void scale_model_input(ov::Tensor& sample, double t) override;

    private:

        std::vector<float> _betas;
        std::vector<float> _alphas;
        std::vector<float> _alphas_cumprod;
        std::vector<float> _sigmas;

        std::vector<double> _timesteps;

        ov::Tensor _cur_sample;

        int64_t _counter = 0;

        std::optional< size_t > _num_inference_steps = {};

        size_t _num_train_timesteps;
        size_t steps_offset;

        std::string _prediction_type;

        bool _is_scale_input_called = false;
        std::optional<bool> _use_karras_sigmas;

        std::string _interpolation_type;
        std::string _timestep_spacing;

        std::shared_ptr< RNG_G > _rng;
    };
}