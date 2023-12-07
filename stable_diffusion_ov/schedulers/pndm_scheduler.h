// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#pragma once
#include "scheduler.h"
#include <openvino/openvino.hpp>
#include <string>
#include <optional>
#include <vector>

namespace cpp_stable_diffusion_ov
{
    class PNDMScheduler : public Scheduler
    {
    public:

        PNDMScheduler(size_t num_train_timesteps = 1000,
            float beta_start = 0.0001f,
            float beta_end = 0.002f,
            std::string beta_schedule = "linear",
            std::optional< std::vector<float> > trained_betas = {},
            bool skip_prk_steps = false,
            bool set_alpha_to_one = false,
            std::string prediction_type = "epsilon",
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

        void scale_model_input(ov::Tensor& sample);

        virtual float init_noise_sigma() override { return _init_noise_sigma; };

        virtual std::vector<double> timesteps() override {

            std::vector<double> ret;
            for (auto& t : _timesteps)
                ret.push_back((double)t);

            return ret;
        };

    private:

        ov::Tensor step_plms(ov::Tensor model_output,
            int64_t timestep,
            ov::Tensor sample);

        ov::Tensor _get_prev_sample(ov::Tensor sample,
            int64_t timestep,
            int64_t prev_timestep,
            ov::Tensor model_output);

        std::vector<float> _betas;
        std::vector<float> _alphas;
        std::vector<float> _alphas_cumprod;
        float _final_alpha_cumprod = 0.f;

        std::vector<int64_t> __timesteps;
        std::vector<int64_t> _timesteps;

        std::vector<int64_t> _prk_timesteps;
        std::vector<int64_t> _plms_timesteps;

        std::vector< ov::Tensor > _ets;
        ov::Tensor _cur_sample;

        int64_t _counter = 0;

        float _init_noise_sigma = 1.0;
        int _pndm_order = 4;

        std::optional< size_t > _num_inference_steps = {};

        size_t _num_train_timesteps;
        size_t steps_offset;

        bool _skip_prk_steps;

        std::string _prediction_type;


    };
}