// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#include "cpp_stable_diffusion_ov/stable_diffusion_interpolation_pipeline.h"
#include "cpp_stable_diffusion_ov/openvino_text_encoder.h"
#include "pipelines/unet_loop.h"
#include "cpp_stable_diffusion_ov/openvino_vae_decoder.h"
#include "schedulers/scheduler.h"
#include "schedulers/scheduler_factory.h"
#include "cpp_stable_diffusion_ov/openvino_model_utils.h"
#include <chrono>

#include "utils/rng.h"

namespace cpp_stable_diffusion_ov
{
    StableDiffusionInterpolationPipeline::StableDiffusionInterpolationPipeline(std::string model_folder,
        std::optional< std::string > unet_subdir,
        std::optional< std::string > cache,
        std::string text_encoder_device,
        std::string unet_positive_device,
        std::string unet_negative_device,
        std::string vae_decoder_device,
        std::string vae_encoder_device)
        : StableDiffusionPipeline(model_folder, unet_subdir, cache, text_encoder_device, unet_positive_device,
            unet_negative_device, vae_decoder_device, vae_encoder_device)
    {

    }

    static std::vector<float> linspace(float start, float end, size_t steps)
    {
        std::vector<float> res(steps);

        for (size_t i = 0; i < steps; i++)
        {
            res[i] = (float)(start + (steps - (steps - i)) * ((end - start) / (steps - 1)));
        }

        return res;
    }

    std::vector<std::shared_ptr<std::vector<uint8_t>>> StableDiffusionInterpolationPipeline::operator()(
        const std::string start_prompt,
        std::optional< std::string > end_prompt,
        std::optional< std::string > negative_prompt,
        std::optional< unsigned int > seed_start,
        std::optional< unsigned int > seed_end,
        float guidance_scale_start,
        float guidance_scale_end,
        int num_inference_steps,
        int num_interpolation_steps,
        const std::string& scheduler_str,
        bool bGiveBGR,
        std::optional< InputImageParams > input_image_params,
        std::optional<std::pair< CallbackFuncUnetIteration, void*>> unet_iteration_callback
        )
    {

        if (num_inference_steps < 1)
        {
            throw std::invalid_argument("num_inference_steps must be >=1. It is set to " + std::to_string(num_inference_steps));
        }

        if (num_interpolation_steps < 0)
        {
            throw std::invalid_argument("num_interpolation_steps must be >=0. It is set to " + std::to_string(num_interpolation_steps));
        }

        std::vector<float> alphas;
        if (num_interpolation_steps > 1)
        {
            alphas = linspace(0.f, 1.f, num_interpolation_steps);
        }
        else
        {
            alphas = { 0.f };
        }

        const float alpha_power = 1.f;
        for (auto& a : alphas)
        {
            a = a * 2 - 1;
            //s = np.sign(a)
            float s;
            if (a > 0.)
                s = 1.f;
            else if (a < 0.f)
                s = -1.f;
            else
                s = 0.f;
            a = (std::pow(std::fabs(a), alpha_power) * s + 1) / 2.f;
        }

        for (size_t i = 0; i < alphas.size(); i++)
        {
            std::cout << "alphas[" << i << "] = " << alphas[i] << std::endl;
        }

        // Set strength. It stays at 1.0 unless input image is being used.
        float strength = 1.f;
        if (input_image_params)
        {
            strength = input_image_params->strength;
            if ((strength < 0) || (strength > 1))
            {
                throw std::invalid_argument("strength must be in the range: 0 <= strength <= 1. It is set to " + std::to_string(strength));
            }
        }

        // generate the text embeddings for the start & end prompts
        ov::Tensor pos_prompt_embeds_start = _encode_prompt(start_prompt);
        if (!end_prompt)
            end_prompt = start_prompt;

        ov::Tensor pos_prompt_embeds_end = _encode_prompt(*end_prompt);

        ov::Tensor neg_prompt_embeds;
        if ((guidance_scale_start > 1.f) || (guidance_scale_end > 1.f))
        {
            if (!negative_prompt)
                negative_prompt = "";

            neg_prompt_embeds = _encode_prompt(*negative_prompt);
        }

        //if user supplied input image, pass it through vae encoder.
        std::optional< ov::Tensor > vae_encoded;
        if (input_image_params)
        {
            vae_encoded = _vae_encode(*input_image_params);
        }

        // Create an Scheduler instance, given string (e.g. "EulerDiscreteScheduler", etc.)
        auto scheduler = SchedulerFactory::Generate(scheduler_str);

        std::vector<std::shared_ptr<std::vector<uint8_t>>> out_img_vec;
        for (auto alpha : alphas)
        {
            std::cout << "############ alpha = " << alpha << std::endl;
            // Generate guidance scale as weighted sum of start & end guidance scale.
            float guidance_scale = guidance_scale_start * (1.f - alpha) + guidance_scale_end * alpha;

            std::cout << "guidance_scale = " << guidance_scale << std::endl;

            //Interpolate between start & end prompts
            ov::Tensor pos_prompt_embeds = ov::Tensor(pos_prompt_embeds_start.get_element_type(),
                pos_prompt_embeds_start.get_shape());
            {
                float* pTextEmbedding = pos_prompt_embeds.data<float>();
                float* pStart = pos_prompt_embeds_start.data<float>();
                float* pEnd = pos_prompt_embeds_end.data<float>();
                for (size_t i = 0; i < pos_prompt_embeds.get_size(); i++)
                {
                    pTextEmbedding[i] = (1.f - alpha) * pStart[i] + alpha * pEnd[i];
                }
            }

            auto image_buf_8u = _run_alpha(vae_encoded,
                pos_prompt_embeds,
                neg_prompt_embeds,
                scheduler,
                num_inference_steps,
                seed_start,
                seed_end,
                guidance_scale,
                strength,
                alpha,
                bGiveBGR);

            if (!image_buf_8u)
            {
                return {};
            }

            out_img_vec.push_back(image_buf_8u);
        }

        return out_img_vec;
    }

    std::shared_ptr<std::vector<uint8_t>> StableDiffusionInterpolationPipeline::StableDiffusionInterpolationPipeline::run_single_alpha(
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
        std::optional< InputImageParams > input_image_params)
    {
        if (num_inference_steps < 1)
        {
            throw std::invalid_argument("num_inference_steps must be >=1. It is set to " + std::to_string(num_inference_steps));
        }

        if ((alpha < 0.) || (alpha > 1.0))
        {
            throw std::invalid_argument("alpha must be in the range: 0 <= alpha <= 1. It is set to " + std::to_string(alpha));
        }

        // Set strength. It stays at 1.0 unless input image is being used.
        float strength = 1.f;
        if (input_image_params)
        {
            strength = input_image_params->strength;
            if ((strength < 0) || (strength > 1))
            {
                throw std::invalid_argument("strength must be in the range: 0 <= strength <= 1. It is set to " + std::to_string(strength));
            }
        }

        bool do_classifier_free_guidance = (guidance_scale > 1.0f);

        // generate the text embeddings for the start & end prompts
        ov::Tensor pos_prompt_embeds_start = _encode_prompt(start_prompt);
        ov::Tensor pos_prompt_embeds_end = _encode_prompt(end_prompt);

        ov::Tensor neg_prompt_embeds;
        if (do_classifier_free_guidance)
        {
            if (!negative_prompt)
                negative_prompt = "";

            neg_prompt_embeds = _encode_prompt(*negative_prompt);
        }

        //Interpolate between start & end prompts
        ov::Tensor pos_prompt_embeds = ov::Tensor(pos_prompt_embeds_start.get_element_type(),
            pos_prompt_embeds_start.get_shape());
        {
            float* pTextEmbedding = pos_prompt_embeds.data<float>();
            float* pStart = pos_prompt_embeds_start.data<float>();
            float* pEnd = pos_prompt_embeds_end.data<float>();
            for (size_t i = 0; i < pos_prompt_embeds.get_size(); i++)
            {
                pTextEmbedding[i] = (1.f - alpha) * pStart[i] + alpha * pEnd[i];
            }
        }

        //if user supplied input image, pass it through vae encoder.
        std::optional< ov::Tensor > vae_encoded;
        if (input_image_params)
        {
            vae_encoded = _vae_encode(*input_image_params);
        }

        // Create an Scheduler instance, given string (e.g. "EulerDiscreteScheduler", etc.)
        auto scheduler = SchedulerFactory::Generate(scheduler_str);

        auto image_buf_8u = _run_alpha(vae_encoded,
            pos_prompt_embeds,
            neg_prompt_embeds,
            scheduler,
            num_inference_steps,
            seed_start,
            seed_end,
            guidance_scale,
            strength,
            alpha,
            bGiveBGR);
        return image_buf_8u;
    }

    std::shared_ptr<std::vector<uint8_t>> StableDiffusionInterpolationPipeline::_run_alpha(
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
        std::optional<std::pair< CallbackFuncUnetIteration, void*>> unet_iteration_callback)
    {
        // set timesteps
        scheduler->set_timesteps(num_inference_steps);

        // get timesteps with possible adjustment from strength
        auto timesteps = _get_timesteps(strength, scheduler);
        double latent_timestep = 0.;
        if (!timesteps.empty())
            latent_timestep = timesteps[0];

#if 0
        auto start_latents = _prepare_latents(vae_encoded, latent_timestep, scheduler, seed_start);
        auto end_latents = _prepare_latents(vae_encoded, latent_timestep, scheduler, seed_end);

        //Generate intial latents as a weighted sum between start & end latents
        ov::Tensor latents = ov::Tensor(start_latents.get_element_type(),
            start_latents.get_shape());

        {
            float* pLatents = latents.data<float>();
            float* pStart = start_latents.data<float>();
            float* pEnd = end_latents.data<float>();
            for (size_t i = 0; i < latents.get_size(); i++)
            {
                //pLatents[i] = pStart[i] + alpha * (pEnd[i] - pStart[i]);
                pLatents[i] = pStart[i] * (1 - alpha) + alpha * pEnd[i];
            }
        }

#else
        auto latents = _prepare_latents_alpha(vae_encoded, latent_timestep, scheduler, seed_start, seed_end, alpha);
#endif

        latents = (*_unet_loop)(timesteps, latents,
            pos_prompt_embeds,
            neg_prompt_embeds,
            guidance_scale,
            scheduler,
            unet_iteration_callback);

        if (!latents)
        {
            return {};
        }

        //latents = 1 / 0.18215 * latents
        {
            auto* pLatents = latents.data<float>();
            for (size_t i = 0; i < latents.get_size(); i++)
            {
                pLatents[i] = 1.f / 0.18215f * pLatents[i];
            }
        }

        // generate output image by vae decoding resultant latent.
        auto vae_decode_out = (*_vae_decoder)(latents);

        // convert vae decoded to a uint8_t RGB / BGR image (buffer)
        auto image_buf_8u = _post_proc(vae_decode_out, bGiveBGR);

        return image_buf_8u;
    }


    // will overwrite v0 with result.
    static void slerp(float t, ov::Tensor& v0, ov::Tensor& v1, float dot_threshold = 0.9995f)
    {
        std::cout << "slerp: t = " << t << std::endl;
        float norm_v0 = 0.f;
        float norm_v1 = 0.f;

        float* pV0 = v0.data<float>();
        float* pV1 = v1.data<float>();
        for (size_t i = 0; i < v0.get_size(); i++)
        {
            norm_v0 += pV0[i] * pV0[i];
            norm_v1 += pV1[i] * pV1[i];
        }

        norm_v0 = std::sqrt(norm_v0);
        norm_v1 = std::sqrt(norm_v1);

        std::cout << "norm_v0 = " << norm_v0 << std::endl;
        std::cout << "norm_v1 = " << norm_v1 << std::endl;

        float dot = 0.f;
        for (size_t i = 0; i < v0.get_size(); i++)
        {
            dot += pV0[i] * pV1[i] / (norm_v0 * norm_v1);
        }

        std::cout << "dot = " << dot << std::endl;
        if (std::fabs(dot) > dot_threshold)
        {
            for (size_t i = 0; i < v0.get_size(); i++)
            {
                if (i == 0)
                {
                    std::cout << "i = 0; v0 = " << pV0[i] << " v1 = " << pV1[i] << std::endl;
                }
                pV0[i] = (1.f - t) * pV0[i] + t * pV1[i];
            }
        }
        else
        {
            auto theta_0 = std::acos(dot);
            std::cout << "theta_0 = " << theta_0 << std::endl;
            auto sin_theta_0 = std::sin(theta_0);
            auto theta_t = theta_0 * t;
            auto sin_theta_t = std::sin(theta_t);
            auto s0 = std::sin(theta_0 - theta_t) / sin_theta_0;
            auto s1 = sin_theta_t / sin_theta_0;
            for (size_t i = 0; i < v0.get_size(); i++)
            {
                pV0[i] = s0 * pV0[i] + s1 * pV1[i];
            }
        }
    }


    ov::Tensor StableDiffusionInterpolationPipeline::_prepare_latents_alpha(std::optional< ov::Tensor > vae_encoded,
        double latent_timestep,
        std::shared_ptr<Scheduler> scheduler,
        std::optional< unsigned int > seed_start,
        std::optional< unsigned int > seed_end,
        float alpha)
    {
        // get the initial random noise unless the user supplied it
        ov::Shape latent_shape = { 1, 4, _height / 8, _width / 8 };

        ov::Tensor noise_start(ov::element::f32, latent_shape);
        ov::Tensor noise_end(ov::element::f32, latent_shape);

        // fill latents with random data
        //todo: Add ability to pass in latents (i.e. for img->img )
        RNG_G rng_start(seed_start);
        RNG_G rng_end(seed_end);

        {
            float* pNoiseS = noise_start.data<float>();
            float* pNoiseE = noise_end.data<float>();
            for (size_t i = 0; i < noise_start.get_size(); i++)
            {
                pNoiseS[i] = (float)(rng_start.gen());
                pNoiseE[i] = (float)(rng_end.gen());
            }
        }

        slerp(alpha, noise_start, noise_end);
        auto noise = noise_start;

        ov::Tensor latents;
        if (!vae_encoded)
        {
            double init_noise_sigma = scheduler->init_noise_sigma();
            switch (noise.get_element_type())
            {
            case ov::element::f32:
            {
                float* pNoise = noise.data<float>();
                for (size_t i = 0; i < noise.get_size(); i++)
                {
                    pNoise[i] *= init_noise_sigma;
                }

                break;
            }
            default:
                throw std::invalid_argument("Latent generation not supported for element type (yet).");
                break;
            }

            latents = noise;
        }
        else
        {
            auto moments_orig = *vae_encoded;
            auto moments = ov::Tensor(moments_orig.get_element_type(),
                moments_orig.get_shape());
            moments_orig.copy_to(moments);

            ov::Tensor mean, logvar;
            // Split moments into mean and var tensors.
            {
                float* pMoments = moments.data<float>();
                ov::Shape tmp_shape = moments.get_shape();
                tmp_shape[1] /= 2;


                // Given that moments is NCHW [1, 4, 512, 512], we don't need to allocate a new buffer.
                // Just wrap 2 new tensors as existing ptr
                mean = ov::Tensor(moments.get_element_type(),
                    tmp_shape, pMoments);
                logvar = ov::Tensor(moments.get_element_type(),
                    tmp_shape, pMoments + mean.get_size());
            }

            //std = np.exp(logvar * 0.5)
            {
                float* pStd = logvar.data<float>();
                for (size_t i = 0; i < logvar.get_size(); i++)
                {
                    float l = pStd[i];

                    pStd[i] = std::exp(0.5f * l);
                }
            }

            //auto tmp_rand = ov::Tensor(ov::element::f32, mean.get_shape());
            float* pMean = mean.data<float>();
            float* pStd = logvar.data<float>();
            latents = ov::Tensor(ov::element::f32, mean.get_shape());
            float* pLatents = latents.data<float>();
            for (size_t i = 0; i < latents.get_size(); i++)
            {
                pLatents[i] = (pMean[i] + pStd[i] * (float)rng_start.gen()) * 0.18215f;
            }

            latents = scheduler->add_noise(latents, noise, latent_timestep);
        }

        return latents;
    }
}

