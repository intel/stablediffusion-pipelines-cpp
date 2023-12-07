// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#include "schedulers/euler_discrete_scheduler.h"
#include "cpp_stable_diffusion_ov/tokenization_utils.h"
#include "cpp_stable_diffusion_ov/openvino_model_utils.h"
#include "utils/rng.h"

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

    static std::vector<float> interp(const std::vector<float>& x, const std::vector<float>& xp, const std::vector<float>& yp)
    {
        if (xp.size() != yp.size())
        {
            throw std::invalid_argument("xp size != yp size");
        }

        std::vector<float> ret;

        //for each x
        for (auto& xv : x)
        {
            //first find i0, which is the last xp where x >= that
            size_t i;
            for (i = 0; i < yp.size(); i++)
            {
                if (i > 0)
                    if (xp[i] <= xp[i - 1])
                        throw std::invalid_argument("xp must be monotonically increasing");

                if (xp[i] > xv)
                {
                    break;
                }
            }

            if (i == 0)
            {
                ret.push_back(yp[0]);
            }
            else if (i == yp.size())
            {
                ret.push_back(yp.back());
            }
            else
            {
                float x1 = xp[i - 1];
                float x2 = xp[i];
                float y1 = yp[i - 1];
                float y2 = yp[i];

                float y = y1 + ((xv - x1) / (x2 - x1)) * (y2 - y1);
                ret.push_back(y);
            }
        }

        return ret;
    }


    EulerDiscreteScheduler::EulerDiscreteScheduler(size_t num_train_timesteps,
        float beta_start,
        float beta_end,
        std::string beta_schedule,
        std::optional< std::vector<float> > trained_betas,
        std::string prediction_type,
        std::string interpolation_type,
        std::optional<bool> use_karras_sigmas,
        std::string timestep_spacing,
        size_t steps_offset)
        : _num_train_timesteps(num_train_timesteps), steps_offset(steps_offset), _prediction_type(prediction_type),
        _use_karras_sigmas(use_karras_sigmas), _timestep_spacing(timestep_spacing), _interpolation_type(interpolation_type)
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
            throw std::invalid_argument(beta_schedule + "is not implemented for EulerDiscreteScheduler");
        }

        _alphas = _betas;
        for (auto& a : _alphas)
        {
            a = 1.f - a;
        }

        _alphas_cumprod = cumprod(_alphas);

        auto sigmas = _alphas_cumprod;
        for (auto& s : sigmas)
        {
            s = std::sqrt((1.f - s) / s);
        }

        _sigmas = std::vector<float>(sigmas.size() + 1, 0.f);
        for (size_t i = 0; i < sigmas.size(); i++)
        {
            //std::cout << "sigmas[" << i << "] = " << sigmas[i] << std::endl;
            _sigmas[i] = sigmas[(sigmas.size() - 1) - i];
        }

#if 0
        for (size_t i = 0; i < _sigmas.size(); i++)
        {
            std::cout << "_sigmas[" << i << "] = " << _sigmas[i] << std::endl;
        }
#endif

        _num_inference_steps = {};
        //timesteps = np.linspace(0, num_train_timesteps - 1, num_train_timesteps, dtype = float)[:: - 1].copy()
        {
            auto tmp = linspace(0, (float)(num_train_timesteps - 1), num_train_timesteps);
            _timesteps.resize(num_train_timesteps);
            for (size_t i = 0; i < tmp.size(); i++)
            {
                _timesteps[i] = tmp[(tmp.size() - 1) - i];
                //std::cout << "_timesteps[" << i << "] = " << _timesteps[i] << std::endl;
            }
        }

        //todo: should seed be programmable?
        _rng = std::make_shared< RNG_G >(0);
    }

    float EulerDiscreteScheduler::init_noise_sigma()
    {
        if (_sigmas.empty())
        {
            throw std::invalid_argument("init_noise_sigma: _sigmas is empty!");
        }

        float sigmas_max = _sigmas[0];
        for (size_t i = 1; i < _sigmas.size(); i++)
        {
            if (_sigmas[i] > sigmas_max)
                sigmas_max = _sigmas[i];
        }

        if ((_timestep_spacing == "linspace") || (_timestep_spacing == "trailing"))
        {
            return sigmas_max;
        }

        return std::sqrt(sigmas_max * sigmas_max + 1);
    }

    void EulerDiscreteScheduler::scale_model_input(ov::Tensor& sample, double t)
    {
        if (_timesteps.empty())
        {
            throw std::invalid_argument("step: timesteps are empty!");
        }

        // step_index = (self.timesteps == timestep).nonzero().item()
        size_t step_index = 0;
        for (size_t i = 0; i < _timesteps.size(); i++)
        {
            if (_timesteps[i] == t)
            {
                step_index = i;
                break;
            }

            if ((i + 1) == _timesteps.size())
            {
                throw std::invalid_argument("step: t not found in timesteps!");
            }
        }

        float sigma = _sigmas[step_index];
        float* pSample = sample.data<float>();
        for (size_t i = 0; i < sample.get_size(); i++)
        {
            pSample[i] = pSample[i] / std::sqrt(sigma * sigma + 1);
        }

        _is_scale_input_called = true;
    }

    void EulerDiscreteScheduler::set_timesteps(size_t num_inference_steps)
    {
        _rng = std::make_shared< RNG_G >(0);

        _num_inference_steps = num_inference_steps;

        std::vector<float> timesteps(num_inference_steps);
        if (_timestep_spacing == "linspace")
        {
            auto timesteps_tmp = linspace(0, (float)(_num_train_timesteps - 1), *_num_inference_steps);
            for (size_t i = 0; i < timesteps_tmp.size(); i++)
            {
                //std::cout << "sigmas[" << i << "] = " << sigmas[i] << std::endl;
                timesteps[i] = timesteps_tmp[(timesteps_tmp.size() - 1) - i];
            }
        }
        else
        {
            throw std::invalid_argument("Only timestep spacing for 'linspace' is implemented right now.");
        }

        auto sigmas = _alphas_cumprod;
        for (auto& s : sigmas)
        {
            s = std::sqrt((1.f - s) / s);
        }

        if (_interpolation_type == "linear")
        {
            std::vector<float> arange(sigmas.size());
            for (size_t i = 0; i < arange.size(); i++)
            {
                arange[i] = (float)i;
            }
            sigmas = interp(timesteps, arange, sigmas);
        }
        else
        {
            throw std::invalid_argument("Only \"linear\" interpolation_type is implemented right now.");
        }

        if (_use_karras_sigmas && *_use_karras_sigmas)
        {
            throw std::invalid_argument("'use karras sigmas' not implemented yet.");
        }

        sigmas.push_back(0.f);
        _sigmas = sigmas;

        _timesteps.resize(timesteps.size());
        for (size_t i = 0; i < _timesteps.size(); i++)
            _timesteps[i] = timesteps[i];
    }


    ov::Tensor EulerDiscreteScheduler::step(ov::Tensor model_output,
        double timestep,
        ov::Tensor sample)
    {
        float s_churn = 0.0;
        float s_tmin = 0.0f;
        float s_tmax = std::numeric_limits<float>::max();
        float s_noise = 1.0f;

        if (_timesteps.empty())
        {
            throw std::invalid_argument("step: timesteps are empty!");
        }

        // step_index = (self.timesteps == timestep).nonzero().item()
        size_t step_index = 0;
        for (size_t i = 0; i < _timesteps.size(); i++)
        {
            if (_timesteps[i] == timestep)
            {
                step_index = i;
                break;
            }
        }

        float sigma = _sigmas[step_index];
        float gamma = 0.f;
        if ((s_tmin <= sigma) && (sigma <= s_tmax))
        {
            gamma = std::min(s_churn / ((float)_sigmas.size() - 1), std::sqrt(2.f) - 1);
        }

        //std::cout << "sigma = " << sigma << std::endl;
        //std::cout << "gamma = " << gamma << std::endl;

        //hack, remove this!
        //std::cout << "rng hack in place! Remove me!" << std::endl;
        //_rng = std::make_shared< RNG_G >(0);

        ov::Tensor eps(model_output.get_element_type(), model_output.get_shape());
        switch (eps.get_element_type())
        {
        case ov::element::f32:
        {
            float* pEPS = eps.data<float>();
            for (size_t i = 0; i < eps.get_size(); i++)
            {
                pEPS[i] = (float)(_rng->gen()) * s_noise;
            }

            break;
        }
        default:
            throw std::invalid_argument("Latent generation not supported for element type (yet).");
            break;
        }

        //save_tensor_to_disk(eps, "noise_" + std::to_string(timestep) + ".raw");

        float sigma_hat = sigma * (gamma + 1);
        //std::cout << "sigma_hat = " << sigma_hat << std::endl;

        if (gamma > 0)
        {
            float* pSample = sample.data<float>();
            float* pEPS = eps.data<float>();
            for (size_t i = 0; i < sample.get_size(); i++)
            {
                //sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5
                pSample[i] = pSample[i] + pEPS[i] * std::sqrt(sigma_hat * sigma_hat - sigma * sigma);
            }
        }

        // 1. compute predicted original sample(x_0) from sigma - scaled predicted noise
        ov::Tensor pred_original_sample;
        if (_prediction_type == "epsilon")
        {
            pred_original_sample = ov::Tensor(model_output.get_element_type(), model_output.get_shape());
            float* pPred = pred_original_sample.data<float>();
            float* pSample = sample.data<float>();
            float* pModelOutput = model_output.data<float>();
            for (size_t i = 0; i < pred_original_sample.get_size(); i++)
            {
                //pred_original_sample = sample - sigma_hat * model_output
                pPred[i] = pSample[i] - sigma_hat * pModelOutput[i];
            }
        }
        else
        {
            throw std::invalid_argument("Only epsilon prediction type is supported right now.");
        }

        //save_tensor_to_disk(sample, "sample_" + std::to_string(timestep) + ".raw");
        //save_tensor_to_disk(pred_original_sample, "pred_original_sample_" + std::to_string(timestep) + ".raw");

        // 2. Convert to an ODE derivative
        auto derivative = ov::Tensor(sample.get_element_type(), sample.get_shape());
        {
            float* pDer = derivative.data<float>();
            float* pPred = pred_original_sample.data<float>();
            float* pSample = sample.data<float>();
            for (size_t i = 0; i < derivative.get_size(); i++)
            {
                //derivative = (sample - pred_original_sample) / sigma_hat
                pDer[i] = (pSample[i] - pPred[i]) / sigma_hat;
            }
        }

        auto dt = _sigmas[step_index + 1] - sigma_hat;
        //std::cout << "dt = " << dt << std::endl;

        auto prev_sample = ov::Tensor(sample.get_element_type(), sample.get_shape());
        {
            float* pSample = sample.data<float>();
            float* pDer = derivative.data<float>();
            float* pPrev = prev_sample.data<float>();
            for (size_t i = 0; i < prev_sample.get_size(); i++)
            {
                //prev_sample = sample + derivative * dt
                pPrev[i] = pSample[i] + pDer[i] * dt;
            }
        }

        return prev_sample;
    }

    ov::Tensor EulerDiscreteScheduler::add_noise(ov::Tensor& original_samples,
        ov::Tensor& noise,
        double timesteps,
        bool return_as_noise)
    {
        auto noisy_samples = ov::Tensor(original_samples.get_element_type(), original_samples.get_shape());

        if (_timesteps.empty())
        {
            throw std::invalid_argument("step: timesteps are empty!");
        }

        size_t step_index = 0;
        for (size_t i = 0; i < _timesteps.size(); i++)
        {
            if (_timesteps[i] == timesteps)
            {
                step_index = i;
                break;
            }

            if ((i + 1) == _timesteps.size())
            {
                throw std::invalid_argument("add_noise: t not found in timesteps!");
            }
        }

        auto sigma = _sigmas[step_index];

        float* pNoisySamples = noisy_samples.data<float>();
        float* pNoise = noise.data<float>();
        float* pOrig = original_samples.data<float>();

        for (size_t i = 0; i < noisy_samples.get_size(); i++)
        {
            //noisy_samples = original_samples + noise * sigma
            pNoisySamples[i] = pOrig[i] + pNoise[i] * sigma;
        }

        return noisy_samples;
    }
}
