// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#include <future>
#include "cpp_stable_diffusion_audio_ov/stable_diffusion_audio_interpolation_pipeline.h"
#include "cpp_stable_diffusion_ov/openvino_text_encoder.h"
#include "pipelines/unet_loop.h"
#include "cpp_stable_diffusion_ov/openvino_vae_encoder.h"
#include "cpp_stable_diffusion_ov/openvino_vae_decoder.h"
#include "schedulers/scheduler.h"
#include "schedulers/scheduler_factory.h"
#include "cpp_stable_diffusion_ov/openvino_model_utils.h"
#include "cpp_stable_diffusion_audio_ov/spectrogram_image_converter.h"
#include <chrono>
#include <math.h>

#ifdef WIN32
#define OS_SEP "\\"
#else
#define OS_SEP "/"
#endif

namespace cpp_stable_diffusion_ov
{

    StableDiffusionAudioInterpolationPipeline::StableDiffusionAudioInterpolationPipeline(std::string model_folder,
        std::optional< std::string > unet_subdir,
        std::optional< std::string > cache,
        std::string text_encoder_device,
        std::string unet_positive_device,
        std::string unet_negative_device,
        std::string vae_decoder_device,
        std::string vae_encoder_device)
        : StableDiffusionInterpolationPipeline(model_folder, unet_subdir, cache, text_encoder_device, unet_positive_device,
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

    static bool run_img_to_wav_routine(std::shared_ptr<std::vector<float>> output, std::shared_ptr<std::vector<uint8_t>> image, std::shared_ptr<SpectrogramImageConverter> converter, size_t chan)
    {
        std::cout << "run_img_to_wav_routine->" << std::endl;
        auto wav = converter->audio_from_spectrogram_image(image, 512, 512, chan);
        output->insert(output->end(), wav->begin(), wav->end());
        std::cout << "<-run_img_to_wav_routine" << std::endl;

        return true;
    }


    size_t StableDiffusionAudioInterpolationPipeline::EstimateTotalUnetIterations(
        float denoising_start,
        float denoising_end,
        int num_interpolation_steps,
        int num_inference_steps_per_sample,
        const std::string& scheduler_str,
        std::optional<size_t> num_output_segments)
    {
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
            a = (powf(fabsf(a), alpha_power) * s + 1) / 2.f;
        }

        auto scheduler = SchedulerFactory::Generate(scheduler_str);

        // set timesteps
        scheduler->set_timesteps(num_inference_steps_per_sample);
        auto sts = scheduler->timesteps();
        int num_inference_steps = (int)sts.size();

        if (num_output_segments && (*num_output_segments < alphas.size()))
        {
            std::cout << "Clipping alphas[] size to " << *num_output_segments << std::endl;
            alphas.resize(*num_output_segments);
        }

        if (num_output_segments && *num_output_segments > alphas.size())
        {
            throw std::invalid_argument("If num_output_segments is set, it must be <= num_interpolation_steps");
        }

        size_t unet_iterations = 0;

        for (auto alpha : alphas)
        {
            float strength = denoising_start * (1.f - alpha) + denoising_end * alpha;

            auto init_timestep = std::min((int)(num_inference_steps * strength), num_inference_steps);

            auto t_start = std::max(num_inference_steps - init_timestep, 0);

            if (t_start >= sts.size())
                continue;

            unet_iterations += sts.size() - t_start;
        }

        return unet_iterations;
    }

    std::pair< std::shared_ptr<std::vector<float>>, std::shared_ptr<std::vector<float>> > StableDiffusionAudioInterpolationPipeline::operator()(
        bool bStereo,
        const std::string prompt_start,
        std::optional< std::string > negative_prompt_start,
        std::optional< std::string > prompt_end,
        std::optional< std::string > negative_prompt_end,
        std::optional< unsigned int > seed_start,
        std::optional< unsigned int > seed_end,
        float denoising_start,
        float denoising_end,
        float guidance_scale_start,
        float guidance_scale_end,
        int num_inference_steps_per_sample,
        int num_interpolation_steps,
        std::optional<size_t> num_output_segments,
        std::string seed_image,
        float alpha_power,
        const std::string& scheduler_str,
        std::optional<std::pair< CallbackFuncUnetIteration, void*>> unet_iteration_callback,
        std::optional<std::pair< CallbackFuncInterpolationIteration, void*>> interp_iteration_callback
        )
    {
        if (num_inference_steps_per_sample < 1)
        {
            throw std::invalid_argument("num_inference_steps_per_sample must be >=1. It is set to " + std::to_string(num_inference_steps_per_sample));
        }

        if ((denoising_start < 0) || (denoising_start > 1))
        {
            throw std::invalid_argument("denoising_start must be in the range: 0 <= denoising_start <= 1. It is set to " + std::to_string(denoising_start));
        }

        if ((denoising_end < 0) || (denoising_end > 1))
        {
            throw std::invalid_argument("denoising_start must be in the range: 0 <= denoising_end <= 1. It is set to " + std::to_string(denoising_end));
        }

        if (num_interpolation_steps < 1)
        {
            throw std::invalid_argument("num_interpolation_steps must be >=1. It is set to " + std::to_string(num_interpolation_steps));
        }

        std::pair< std::shared_ptr<std::vector<float>>, std::shared_ptr<std::vector<float>> > output_pair;
        std::vector<float> alphas;
        if (num_interpolation_steps > 1)
        {
            alphas = linspace(0.f, 1.f, num_interpolation_steps);
        }
        else
        {
            alphas = { 0.f };
        }

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
            a = (powf(fabsf(a), alpha_power) * s + 1) / 2.f;
        }

        if (num_output_segments && (*num_output_segments < alphas.size()))
        {
            std::cout << "Clipping alphas[] size to " << *num_output_segments << std::endl;
            alphas.resize(*num_output_segments);
        }

        if (num_output_segments && *num_output_segments > alphas.size())
        {
            throw std::invalid_argument("If num_output_segments is set, it must be <= num_interpolation_steps");
        }

        // generate the text embeddings for the start & end prompts
        ov::Tensor pos_prompt_embeds_start = _encode_prompt(prompt_start);
        if (!prompt_end)
            prompt_end = prompt_start;

        ov::Tensor pos_prompt_embeds_end = _encode_prompt(*prompt_end);

        ov::Tensor neg_prompt_embeds;
        if ((guidance_scale_start > 1.f) || (guidance_scale_end > 1.f))
        {
            if (!negative_prompt_start)
                negative_prompt_start = "";

            neg_prompt_embeds = _encode_prompt(*negative_prompt_start);
        }

        //image latents
        auto init_image_tensor = ov::Tensor(pos_prompt_embeds_start.get_element_type(),
            ov::Shape({ 1, 3, _height, _width }));

        std::string fname = _model_folder + OS_SEP + seed_image + ".raw";
        load_tensor_from_disk(init_image_tensor, fname);

        std::optional< ov::Tensor > vae_encoded;
        vae_encoded = (*_vae_encoder)(init_image_tensor);

        // Create an Scheduler instance, given string (e.g. "EulerDiscreteScheduler", etc.)
        auto scheduler = SchedulerFactory::Generate(scheduler_str);

        auto converterL = std::make_shared<SpectrogramImageConverter>();
        auto converterR = std::make_shared<SpectrogramImageConverter>();

        auto output_L = std::make_shared<std::vector<float>>();
        std::shared_ptr<std::vector<float>> output_R;
        if (bStereo)
        {
            output_R = std::make_shared<std::vector<float>>();
        }


        std::future<bool> last_iteration_img_to_wav_L;
        std::future<bool> last_iteration_img_to_wav_R;
        size_t interp_i = 0;
        for (auto alpha : alphas)
        {
            // Generate guidance scale as weighted sum of start & end guidance scale.
            float guidance_scale = guidance_scale_start * (1.f - alpha) + guidance_scale_end * alpha;

            float strength = denoising_start * (1.f - alpha) + denoising_end * alpha;

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
                num_inference_steps_per_sample,
                seed_start,
                seed_end,
                guidance_scale,
                strength,
                alpha,
                false,
                unet_iteration_callback);

            if (!image_buf_8u)
            {
                if (last_iteration_img_to_wav_L.valid())
                    last_iteration_img_to_wav_L.wait();

                if (last_iteration_img_to_wav_R.valid())
                    last_iteration_img_to_wav_R.wait();

                return {};
            }

            if (interp_iteration_callback)
            {
                interp_iteration_callback->first(interp_i, alphas.size(), {}, image_buf_8u, _width, _height, interp_iteration_callback->second);
            }

            if (last_iteration_img_to_wav_L.valid())
                last_iteration_img_to_wav_L.wait();

            if (last_iteration_img_to_wav_R.valid())
                last_iteration_img_to_wav_R.wait();

            size_t chan_L = bStereo ? 1 : 0;
            last_iteration_img_to_wav_L = std::async(run_img_to_wav_routine, output_L, image_buf_8u, converterL, chan_L);

            if (bStereo)
            {
                last_iteration_img_to_wav_R = std::async(run_img_to_wav_routine, output_R, image_buf_8u, converterR, 2);
            }

            interp_i++;
        }
        if (last_iteration_img_to_wav_L.valid())
            last_iteration_img_to_wav_L.wait();

        if (last_iteration_img_to_wav_R.valid())
            last_iteration_img_to_wav_R.wait();

        output_pair.first = output_L;
        output_pair.second = output_R;
        return output_pair;
    }
}

