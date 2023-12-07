// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#include "pndm_scheduler.h"
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

    PNDMScheduler::PNDMScheduler(size_t num_train_timesteps,
        float beta_start,
        float beta_end,
        std::string beta_schedule,
        std::optional< std::vector<float> > trained_betas,
        bool skip_prk_steps,
        bool set_alpha_to_one,
        std::string prediction_type,
        size_t steps_offset)
        : _num_train_timesteps(num_train_timesteps), steps_offset(steps_offset), _skip_prk_steps(skip_prk_steps), _prediction_type(prediction_type)
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
#if 0
            for (int i = 0; i < num_train_timesteps; i++)
            {
                std::cout << i << ": " << std::fixed << std::setprecision(16) << _betas[i] << std::endl;

            }
#endif
        }
        else
        {
            throw std::invalid_argument(beta_schedule + "is not implemented for PNDMScheduler");
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

        //For now we only support F-PNDM, i.e. the runge-kutta method
        //For more information on the algorithm please take a look at the paper: https://arxiv.org/pdf/2202.09778.pdf
        //mainly at formula (9), (12), (13) and the Algorithm 2.
        _pndm_order = 4;

        // self._timesteps = np.arange(0, num_train_timesteps)[::-1].copy()
        __timesteps.resize(num_train_timesteps);
        for (size_t i = 0; i < __timesteps.size(); i++)
        {
            __timesteps[i] = (__timesteps.size() - 1) - i;
        }
    }

    void PNDMScheduler::set_timesteps(size_t num_inference_steps)
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

        if (_skip_prk_steps)
        {
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
        }
        else
        {
            throw std::invalid_argument("set_timesteps is not yet implemented for skip_prk_steps=false config");
        }

        _timesteps = {};
        _timesteps.insert(_timesteps.end(), _prk_timesteps.begin(), _prk_timesteps.end());
        _timesteps.insert(_timesteps.end(), _plms_timesteps.begin(), _plms_timesteps.end());

        _ets = {};
        _counter = 0;
        //todo:
        //self.cur_model_output = 0
    }


    ov::Tensor PNDMScheduler::step(ov::Tensor model_output,
        double ts,
        ov::Tensor sample)
    {
        int64_t timestep = (int64_t)ts;
        if (_counter < _prk_timesteps.size() && !(_skip_prk_steps))
        {
            throw std::invalid_argument("Not yet implemented step_prk method. Set skip_prk_steps to true for now.");
        }
        else
        {
            return step_plms(model_output, timestep, sample);
        }
    }

    ov::Tensor PNDMScheduler::step_plms(ov::Tensor model_output,
        int64_t timestep,
        ov::Tensor sample)
    {
        if (!_num_inference_steps)
        {
            throw std::invalid_argument("Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler");
        }

        if (!_skip_prk_steps && _ets.size() < 3)
        {
            throw std::invalid_argument("an only be run AFTER scheduler has been run ");
        }

        int64_t prev_timestep = timestep - _num_train_timesteps / *_num_inference_steps;

        if (_counter != 1)
        {
            if (_ets.size() > 3)
            {
                _ets = std::vector<ov::Tensor>(_ets.begin() + (_ets.size() - 3), _ets.end());
            }

            // when caching model_output, we need to make a deep copy to avoid the case where
            // caller retains shallow copy, and it modified by them.
            ov::Tensor mo = ov::Tensor(model_output.get_element_type(),
                model_output.get_shape());
            model_output.copy_to(mo);

            _ets.push_back(mo);
        }
        else
        {
            prev_timestep = timestep;
            timestep = timestep + _num_train_timesteps / *_num_inference_steps;
        }

        if ((_ets.size() == 1) && _counter == 0)
        {
            _cur_sample = sample;
        }
        else if ((_ets.size() == 1) && _counter == 1)
        {
            //model_output = (model_output + self.ets[-1]) / 2
            float* pModelOutput = model_output.data<float>();
            float* pEts = _ets.back().data<float>();

            for (size_t i = 0; i < model_output.get_size(); i++)
            {
                pModelOutput[i] += pEts[i];
                pModelOutput[i] /= 2.f;
            }
            sample = _cur_sample;
            _cur_sample = {};
        }
        else if (_ets.size() == 2)
        {
            //model_output = (3 * self.ets[-1] - self.ets[-2]) / 2
            float* pEtsM1 = _ets.back().data<float>();
            float* pEtsM2 = _ets.front().data<float>();
            float* pModelOutput = model_output.data<float>();
            for (size_t i = 0; i < model_output.get_size(); i++)
            {
                pModelOutput[i] = (3.f * pEtsM1[i] - pEtsM2[i]) / 2.f;
            }
        }
        else if (_ets.size() == 3)
        {
            //model_output = (23 * self.ets[-1] - 16 * self.ets[-2] + 5 * self.ets[-3]) / 12
            float* pEtsM1 = _ets[2].data<float>();
            float* pEtsM2 = _ets[1].data<float>();
            float* pEtsM3 = _ets[0].data<float>();
            float* pModelOutput = model_output.data<float>();
            for (size_t i = 0; i < model_output.get_size(); i++)
            {
                pModelOutput[i] = (23.f * pEtsM1[i] - 16.f * pEtsM2[i] + 5.f * pEtsM3[i]) / 12.f;
            }
        }
        else
        {
            float* pEtsM1 = (_ets.end() - 1)->data<float>();
            float* pEtsM2 = (_ets.end() - 2)->data<float>();
            float* pEtsM3 = (_ets.end() - 3)->data<float>();
            float* pEtsM4 = (_ets.end() - 4)->data<float>();
            float* pModelOutput = model_output.data<float>();

            //model_output = (1 / 24) * (55 * self.ets[-1] - 59 * self.ets[-2] + 37 * self.ets[-3] - 9 * self.ets[-4])
            for (size_t i = 0; i < model_output.get_size(); i++)
            {
                pModelOutput[i] = (1.f / 24.f) * (55.f * pEtsM1[i] - 59.f * pEtsM2[i] + 37.f * pEtsM3[i] - 9.f * pEtsM4[i]);
            }
        }


        auto prev_sample = _get_prev_sample(sample, timestep, prev_timestep, model_output);

        _counter += 1;

        return prev_sample;
    }

    void PNDMScheduler::scale_model_input(ov::Tensor& sample)
    {
        //no-op in case of PNDM
    }

    ov::Tensor PNDMScheduler::_get_prev_sample(ov::Tensor sample,
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


    ov::Tensor PNDMScheduler::add_noise(ov::Tensor& original_samples,
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