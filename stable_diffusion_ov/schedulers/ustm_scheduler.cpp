// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#include "schedulers/ustm_scheduler.h"
#include "cpp_stable_diffusion_ov/tokenization_utils.h"
#include "cpp_stable_diffusion_ov/openvino_model_utils.h"

namespace cpp_stable_diffusion_ov
{
    static std::vector<float> linspace(float start, float end, size_t steps)
    {
        std::vector<float> res(steps);

        for (size_t i = 0; i < steps; i++)
        {
            res[i] = (float)(start + (steps - (steps - i)) * ((end - start) / (steps - 1)));
        }

        return res;
    }

    static std::vector<float> cumprod(std::vector<float>& input)
    {
        std::vector<float> res = input;
        for (size_t i = 1; i < res.size(); i++)
        {
            res[i] = res[i] * res[i - 1];
        }

        return res;
    }

    USTMScheduler::USTMScheduler(size_t num_train_timesteps,
        float beta_start,
        float beta_end,
        std::string beta_schedule,
        std::optional< std::vector<float> > trained_betas,
        bool set_alpha_to_one,
        std::string prediction_type,
        size_t steps_offset)
        : _num_train_timesteps(num_train_timesteps), steps_offset(steps_offset), _prediction_type(prediction_type)
    {
        if (trained_betas)
        {
            _betas = *trained_betas;
        }
        else if (beta_schedule == "linear")
        {
            _betas = linspace(beta_start, beta_end, num_train_timesteps);
        }
        else if (beta_schedule == "scaled_linear")
        {
            _betas = linspace(std::sqrt(beta_start), std::sqrt(beta_end), num_train_timesteps);
            for (auto& b : _betas)
            {
                b *= b;
            }
        }
        else
        {
            throw std::invalid_argument(beta_schedule + "is not implemented for USTMScheduler");
        }

        _alphas = _betas;
        for (auto& a : _alphas)
        {
            a = 1.f - a;
        }

        _alphas_cumprod = cumprod(_alphas);

        if (set_alpha_to_one)
            _final_alpha_cumprod = 1.0f;
        else
            _final_alpha_cumprod = _alphas_cumprod[0];

        // standard deviation of the initial noise distribution
        _init_noise_sigma = 1.0f;

        // self._timesteps = np.arange(0, num_train_timesteps)[::-1].copy()
        __timesteps.resize(num_train_timesteps);
        for (size_t i = 0; i < __timesteps.size(); i++)
        {
            __timesteps[i] = (__timesteps.size() - 1) - i;
        }
    }

    void USTMScheduler::set_timesteps(size_t num_inference_steps)
    {
        _num_inference_steps = num_inference_steps;

        int step_ratio = _num_train_timesteps / num_inference_steps;

        // creates integer timesteps by multiplying by ratio
        // casting to int to avoid issues when num_inference_step is power of 3
        //self._timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()
       // self._timesteps += self.config.steps_offset
        __timesteps.resize(num_inference_steps);
        for (size_t i = 0; i < __timesteps.size(); i++)
        {
            __timesteps[i] = i * step_ratio + steps_offset;
        }


        _prk_timesteps = {};
        auto plms_timesteps = vslice(__timesteps, {}, -1);
        auto tmp = vslice(__timesteps, -2, -1);
        plms_timesteps.insert(plms_timesteps.end(), tmp.begin(), tmp.end());
        tmp = vslice(__timesteps, -1, {});
        plms_timesteps.insert(plms_timesteps.end(), tmp.begin(), tmp.end());

        _plms_timesteps.resize(plms_timesteps.size());
        for (size_t i = 0; i < plms_timesteps.size(); i++)
        {
            _plms_timesteps[i] = plms_timesteps[(plms_timesteps.size() - 1) - i];
        }

        _timesteps = {};
        _timesteps.insert(_timesteps.end(), _prk_timesteps.begin(), _prk_timesteps.end());
        _timesteps.insert(_timesteps.end(), _plms_timesteps.begin(), _plms_timesteps.end());

        _counter = 0;
        //todo:
        //self.cur_model_output = 0
    }


    ov::Tensor USTMScheduler::step(ov::Tensor model_output,
        double ts,
        ov::Tensor sample)
    {
        int64_t timestep = (int64_t)ts;

        if (!_num_inference_steps)
        {
            throw std::invalid_argument("Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler");
        }

        int64_t prev_timestep = timestep - _num_train_timesteps / *_num_inference_steps;

        if (_counter == 1)
        {
            prev_timestep = timestep;
            timestep = timestep + _num_train_timesteps / *_num_inference_steps;
        }

        if (_counter == 0)
        {
            _cur_sample = sample;
        }
        else if (_counter == 1)
        {
            sample = _cur_sample;
            _cur_sample = {};
        }

        auto prev_sample = _get_prev_sample(sample, timestep, prev_timestep, model_output);

        _counter += 1;

        return prev_sample;
    }


    ov::Tensor USTMScheduler::_get_prev_sample(ov::Tensor sample,
        int64_t timestep,
        int64_t prev_timestep,
        ov::Tensor model_output)
    {
        // See formula(9) of PNDM paper https ://arxiv.org/pdf/2202.09778.pdf
        auto alpha_prod_t = _alphas_cumprod[timestep];
        float alpha_prod_t_prev;
        if (prev_timestep >= 0)
        {
            alpha_prod_t_prev = _alphas_cumprod[prev_timestep];
        }
        else
        {
            alpha_prod_t_prev = _final_alpha_cumprod;
        }
        auto beta_prod_t = 1.f - alpha_prod_t;
        auto beta_prod_t_prev = 1.f - alpha_prod_t_prev;

        if (_prediction_type == "v_prediction")
        {
            throw std::invalid_argument("v_prediction case needs to be implemented.");
        }
        else if (_prediction_type != "epsilon")
        {
            throw std::invalid_argument("prediction_type given as " + _prediction_type + " must be one of `epsilon` or `v_prediction`");
        }

        auto sample_coeff = std::sqrt(alpha_prod_t_prev / alpha_prod_t);

        auto model_output_denom_coeff = alpha_prod_t * std::sqrt(beta_prod_t_prev) +
            std::sqrt(alpha_prod_t * beta_prod_t * alpha_prod_t_prev);

        ov::Tensor prev_sample = ov::Tensor(sample.get_element_type(),
            sample.get_shape());

        // full formula(9)
        //    prev_sample = (
        //        sample_coeff * sample - (alpha_prod_t_prev - alpha_prod_t) * model_output / model_output_denom_coeff
        //        )
        float* pPrevSample = prev_sample.data<float>();
        float* pSample = sample.data<float>();
        float* pModelOutput = model_output.data<float>();
        for (size_t i = 0; i < prev_sample.get_size(); i++)
        {
            pPrevSample[i] = sample_coeff * pSample[i] - (alpha_prod_t_prev - alpha_prod_t) * pModelOutput[i] / model_output_denom_coeff;
        }

        return prev_sample;
    }


    ov::Tensor USTMScheduler::add_noise(ov::Tensor& original_samples,
        ov::Tensor& noise,
        double ts,
        bool return_as_noise)
    {
        size_t timesteps = (size_t)ts;
        auto sqrt_alpha_prod = std::sqrt(_alphas_cumprod[timesteps]);
        auto sqrt_one_minus_alpha_prod = std::sqrt(1 - _alphas_cumprod[timesteps]);

        auto noisy_samples = ov::Tensor(noise.get_element_type(),
            noise.get_shape());

        float* pNoisySamples = noisy_samples.data<float>();
        float* pNoise = noise.data<float>();
        float* pSamples = original_samples.data<float>();

        //be cautious here, keeping in mind that pNoise may be equal to pNoisySamples
        for (size_t i = 0; i < noise.get_size(); i++)
        {
            pNoisySamples[i] = sqrt_alpha_prod * pSamples[i] + sqrt_one_minus_alpha_prod * pNoise[i];
        }

        return noisy_samples;
    }

}