// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#include "cpp_stable_diffusion_ov/stable_diffusion_pipeline.h"
#include "cpp_stable_diffusion_ov/clip_tokenizer.h"
#include "cpp_stable_diffusion_ov/openvino_text_encoder.h"
#include "pipelines/unet_loop.h"
#include "cpp_stable_diffusion_ov/openvino_vae_decoder.h"
#include "cpp_stable_diffusion_ov/openvino_vae_encoder.h"
#include "schedulers/scheduler.h"
#include "schedulers/scheduler_factory.h"
#include "cpp_stable_diffusion_ov/openvino_model_utils.h"
#include "cpp_stable_diffusion_ov/model_collateral_cache.h"
#include <chrono>

#include "utils/rng.h"

#ifdef WIN32
#define OS_SEP "\\"
#else
#define OS_SEP "/"
#include <cmath>
#include <ctgmath>
#endif

namespace cpp_stable_diffusion_ov
{
    StableDiffusionPipeline::StableDiffusionPipeline(std::string model_folder,
        std::optional< std::string > unet_subdir,
        std::optional< std::string > cache,
        std::string text_encoder_device,
        std::string unet_positive_device,
        std::string unet_negative_device,
        std::string vae_decoder_device,
        std::string vae_encoder_device)
        : _model_folder(model_folder)
    {
        //In python, '77' comes from tokenizer config json. Seems random but it appears
        // to be a very well-defined constants across many variations of SD pipelines..
        _tok_max_length = 77;
        CLIPTokenizer::CLIPTokenizer_Params init;
        init.baseInit.baseInit.model_max_length = _tok_max_length;
        _tokenizer = std::make_shared< CLIPTokenizer >(init);

        std::string cache_dir;
        if (!cache)
        {
            cache_dir = "my_cache";
        }
        else
        {
            cache_dir = *cache;
        }

        auto m = ModelCollateralCache::instance()->GetModelCollateral(model_folder, unet_subdir, cache_dir, text_encoder_device, unet_positive_device,
            unet_negative_device, vae_decoder_device, vae_encoder_device);

        _unet_loop = m.unet_loop;
        _vae_decoder = m.vae_decoder;
        _text_encoder = m.text_encoder;
        _vae_encoder = m.vae_encoder;
    }

    std::shared_ptr<std::vector<uint8_t>> StableDiffusionPipeline::operator()(
        const std::string prompt,
        std::optional< std::string > negative_prompt,
        int num_inference_steps,
        const std::string& scheduler_str,
        std::optional< unsigned int > seed,
        float guidance_scale,
        bool bGiveBGR,
        std::optional< InputImageParams > input_image_params,
        std::optional<std::pair< CallbackFuncUnetIteration, void*>> unet_iteration_callback)
    {

        if (num_inference_steps < 1)
        {
            throw std::invalid_argument("num_inference_steps must be >=1. It is set to " + std::to_string(num_inference_steps));
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

        auto embeds = _encode_prompt(prompt,
            negative_prompt,
            do_classifier_free_guidance);

        ov::Tensor pos_prompt_embeds = embeds.first;
        ov::Tensor neg_prompt_embeds = embeds.second;

        //if user supplied input image, pass it through vae encoder.
        std::optional< ov::Tensor > vae_encoded;
        if (input_image_params)
        {
            vae_encoded = _vae_encode(*input_image_params);
        }

        // Create an Scheduler instance, given string (e.g. "EulerDiscreteScheduler", etc.)
        auto scheduler = SchedulerFactory::Generate(scheduler_str);

        // set timesteps
        scheduler->set_timesteps(num_inference_steps);

        // get timesteps with possible adjustment from strength
        auto timesteps = _get_timesteps(strength, scheduler);
        double latent_timestep = 0.;
        if (!timesteps.empty())
            latent_timestep = timesteps[0];

        // prepare initial latents. 
        // If user passed in image, it will be generated as noise added to vae encoded.
        // Otherwise, it will be set as random noise.
        auto latents = _prepare_latents(vae_encoded, latent_timestep, scheduler, seed);

        latents = (*_unet_loop)(timesteps, latents, pos_prompt_embeds, neg_prompt_embeds, guidance_scale, scheduler, unet_iteration_callback);

        if (!latents)
        {
            return {};
        }

        //latents = 1 / 0.18215 * latents
        {
            auto* pLatents = latents.data<float>();
            for (size_t i = 0; i < latents.get_size(); i++)
            {
                pLatents[i] = (1.f / 0.18215f) * pLatents[i];
            }
        }

        // generate output image by vae decoding resultant latent.
        auto vae_decode_out = (*_vae_decoder)(latents);

        // convert vae decoded to a uint8_t RGB / BGR image (buffer)
        auto image_buf_8u = _post_proc(vae_decode_out, bGiveBGR);

        return image_buf_8u;

    }

    ov::Tensor StableDiffusionPipeline::_vae_encode(InputImageParams& input_image_params)
    {

        auto buf = input_image_params.image_buffer;
        if (!buf)
        {
            throw std::invalid_argument("InputImageParams.buffer is NULL");
        }

        if (buf->size() != (_width * _height * 3))
        {
            throw std::invalid_argument("Expected image buffer within input_image_params to have size " +
                std::to_string(_width * _height * 3) + ", but it has size " + std::to_string(buf->size()));
        }


        auto init_image_tensor = _preprocess(input_image_params);
        auto moments = (*_vae_encoder)(init_image_tensor);

        //'moments' above is a ov::Tensor, but direct reference to
        // the output of vae encoder's infer request. So, return a copy of it,
        // otherwise what we return may be (implicitly) overwritten if we invoke
        // the vae encoder again.
        auto moments_copy = ov::Tensor(moments.get_element_type(),
            moments.get_shape());
        moments.copy_to(moments_copy);
        return moments_copy;
    }



    ov::Tensor StableDiffusionPipeline::_prepare_latents(std::optional< ov::Tensor > vae_encoded,
        double latent_timestep,
        std::shared_ptr<Scheduler> scheduler,
        std::optional< unsigned int > seed)
    {
        // get the initial random noise unless the user supplied it
        ov::Shape latent_shape = { 1, 4, _height / 8, _width / 8 };

        ov::Tensor noise(ov::element::f32, latent_shape);

        // fill latents with random data
        //todo: Add ability to pass in latents (i.e. for img->img )
        RNG_G rng(seed);
        switch (noise.get_element_type())
        {
        case ov::element::f32:
        {
            float* pNoise = noise.data<float>();
            for (size_t i = 0; i < noise.get_size(); i++)
            {
                pNoise[i] = (float)(rng.gen());
            }

            break;
        }
        default:
            throw std::invalid_argument("Latent generation not supported for element type (yet).");
            break;
        }

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
#ifdef WIN32
                    pStd[i] = std::expf(0.5f * l);
#else
                    pStd[i] = expf(0.5f * l);
#endif
                }
            }

            //auto tmp_rand = ov::Tensor(ov::element::f32, mean.get_shape());
            float* pMean = mean.data<float>();
            float* pStd = logvar.data<float>();
            latents = ov::Tensor(ov::element::f32, mean.get_shape());
            float* pLatents = latents.data<float>();
            for (size_t i = 0; i < latents.get_size(); i++)
            {
                pLatents[i] = (pMean[i] + pStd[i] * (float)rng.gen()) * 0.18215f;
            }

            latents = scheduler->add_noise(latents, noise, latent_timestep);
        }

        return latents;
    }

    ov::Tensor StableDiffusionPipeline::_encode_prompt(const std::string prompt)
    {
        // tokenize the prompt
        BatchEncoding text_inputs = _tokenizer->call(
            prompt,  //text 
            {},      //text_pair
            {},      //text_target
            {},      //text_pair_target
            true,      //add_special_tokens 
            "max_length",  //padding
            true,         //truncation
            _tok_max_length);  //max_length

        auto text_input_ids = text_inputs["input_ids"];

        //invoke the text encoder, producing the positive prompt embeddings
        auto prompt_embeddings = (*_text_encoder)(text_input_ids);

        //'prompt_embeddings' above is a ov::Tensor, but direct reference to
        // the output of text encoder's infer request. So, return a copy of it,
        // otherwise what we return may be (implicitly) overwritten if we invoke
        // the text encoder again.
        auto prompt_embeds_copy = ov::Tensor(prompt_embeddings.get_element_type(),
            prompt_embeddings.get_shape());
        prompt_embeddings.copy_to(prompt_embeds_copy);

        return prompt_embeds_copy;
    }

    std::pair<ov::Tensor, ov::Tensor>  StableDiffusionPipeline::_encode_prompt(const std::string prompt,
        std::optional< std::string > negative_prompt,
        bool do_classifier_free_guidance)
    {
        ov::Tensor pos_prompt_embeds;
        ov::Tensor neg_prompt_embeds;

        pos_prompt_embeds = _encode_prompt(prompt);

        if (do_classifier_free_guidance)
        {
            if (!negative_prompt)
                negative_prompt = "";

            neg_prompt_embeds = _encode_prompt(*negative_prompt);
        }

        return { pos_prompt_embeds, neg_prompt_embeds };
    }

    std::vector<double> StableDiffusionPipeline::_get_timesteps(float strength, std::shared_ptr<Scheduler> scheduler)
    {
        auto sts = scheduler->timesteps();
        int num_inference_steps = (int)sts.size();

        auto init_timestep = std::min((int)(num_inference_steps * strength), num_inference_steps);

        auto t_start = std::max(num_inference_steps - init_timestep, 0);

        if (t_start >= sts.size())
            return {};

        std::vector<double> timesteps = std::vector<double>(sts.begin() + t_start, sts.end());

        return timesteps;
    }

    ov::Tensor StableDiffusionPipeline::_preprocess(InputImageParams& image_params)
    {
        ov::Tensor preprocessed(ov::element::f32, { 1, 3, _height, _width });

        float* pTensor = preprocessed.data<float>();
        uint8_t* pImg = image_params.image_buffer->data();

        if (image_params.isNHWC)
        {
            float* pTensorR;
            float* pTensorG;
            float* pTensorB;

            if (image_params.isBGR)
            {
                pTensorB = pTensor;
                pTensorG = pTensor + _width * _height;
                pTensorR = pTensor + _width * _height * 2;
            }
            else
            {
                pTensorR = pTensor;
                pTensorG = pTensor + _width * _height;
                pTensorB = pTensor + _width * _height * 2;
            }

            uint8_t* pU8 = pImg;
            for (size_t h = 0; h < _height; h++)
            {

                for (size_t w = 0; w < _width; w++)
                {
                    pTensorR[w] = pU8[w * 3 + 0];
                    pTensorG[w] = pU8[w * 3 + 1];
                    pTensorB[w] = pU8[w * 3 + 2];
                }

                pTensorR += _width;
                pTensorG += _width;
                pTensorB += _width;
                pU8 += _width * 3;
            }
        }
        else
        {
            uint8_t* pU8R;
            uint8_t* pU8G;
            uint8_t* pU8B;

            if (image_params.isBGR)
            {
                pU8B = pImg;
                pU8G = pImg + _width * _height;
                pU8R = pImg + _width * _height * 2;
            }
            else
            {
                pU8R = pImg;
                pU8G = pImg + _width * _height;
                pU8B = pImg + _width * _height * 2;
            }

            float* pTensorR = pTensor;
            float* pTensorG = pTensor + _width * _height;
            float* pTensorB = pTensor + _width * _height * 2;

            for (size_t i = 0; i < _width * _height; i++)
            {
                pTensorR[i] = (float)pU8R[i];
                pTensorG[i] = (float)pU8G[i];
                pTensorB[i] = (float)pU8B[i];
            }
        }

        for (size_t i = 0; i < preprocessed.get_size(); i++)
        {
            pTensor[i] /= 255.f;
            pTensor[i] = 2.f * pTensor[i] - 1.f;
        }


        return preprocessed;
    }

    std::shared_ptr<std::vector<uint8_t>> StableDiffusionPipeline::_post_proc(ov::Tensor vae_decode_out,
        bool bGiveBGR)
    {
        //normalize to [0,1]
        //image = np.clip(image / 2 + 0.5, 0, 1)
        {
            auto* pImage = vae_decode_out.data<float>();
            for (size_t i = 0; i < vae_decode_out.get_size(); i++)
            {
                pImage[i] = pImage[i] / 2.f + 0.5f;
                pImage[i] = pImage[i] > 1.f ? 1.f : pImage[i];
                pImage[i] = pImage[i] < 0 ? 0 : pImage[i];
            }
        }

        std::shared_ptr<std::vector<uint8_t>> image_buf_8u = std::make_shared<std::vector<uint8_t>>(_width * _height * 3);

        //convert to nchw, 8 bit
        float* pRF = vae_decode_out.data<float>();
        float* pGF = pRF + _width * _height;
        float* pBF = pGF + _width * _height;

        uint8_t* pImagebuf = image_buf_8u->data();
        if (bGiveBGR)
        {
            for (int i = 0; i < _width * _height; i++)
            {
                pImagebuf[i * 3 + 0] = (uint8_t)(pBF[i] * 255.f);
                pImagebuf[i * 3 + 1] = (uint8_t)(pGF[i] * 255.f);
                pImagebuf[i * 3 + 2] = (uint8_t)(pRF[i] * 255.f);
            }
        }
        else
        {
            for (int i = 0; i < _width * _height; i++)
            {
                pImagebuf[i * 3 + 0] = (uint8_t)(pRF[i] * 255.f);
                pImagebuf[i * 3 + 1] = (uint8_t)(pGF[i] * 255.f);
                pImagebuf[i * 3 + 2] = (uint8_t)(pBF[i] * 255.f);
            }
        }

        return image_buf_8u;
    }
}