// Copyright(C) 2024 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#include "pipelines/unet_loop_batch2.h"
#include "cpp_stable_diffusion_ov/openvino_model_utils.h"
#include <chrono>

//todo: this should change to header for base scheduler class once it exists.
#include "schedulers/scheduler.h"

namespace cpp_stable_diffusion_ov
{
    UNetLoopBatch2::UNetLoopBatch2(std::shared_ptr<ov::Model>  unet_model,
        std::string device,
        size_t max_tok_len,
        size_t img_width,
        size_t img_height,
        ov::Core& core)
        : _device(device)
    {
        logBasicModelInfo(unet_model);

        //we'll run batch size 2, but these are the shapes that we expect the 'sample'latent' and encoder 
        // hidden states tensors to be passed into operator() below. 
        _sample_shape = ov::Shape{ 1, 4, img_height / 8, img_width / 8 };
        _encoder_hidden_states_shape = ov::Shape{ 1, max_tok_len, 768 };

        std::cout << "UNet model info:" << std::endl;
        logBasicModelInfo(unet_model);

        {
            std::cout << "Compiling batch2 unet model for " << device << "..." << std::endl;
            auto compiled_model = core.compile_model(unet_model, device);
            std::cout << "Compiling batch2 unet model for " << device << " done" << std::endl;
            _infer_request = compiled_model.create_infer_request();
        }
    }

    ov::Tensor UNetLoopBatch2::operator()(const std::vector<double>& timesteps, ov::Tensor latents, ov::Tensor encoder_hidden_states_positive,
        ov::Tensor encoder_hidden_states_negative, float guidance_scale, std::shared_ptr<Scheduler> scheduler,
        std::optional<std::pair< CallbackFuncUnetIteration, void*>> unet_iteration_callback)
    {
        if (latents.get_shape() != _sample_shape)
        {
            throw std::invalid_argument("invalid sample shape");
        }

        if (encoder_hidden_states_positive.get_shape() != _encoder_hidden_states_shape)
        {
            throw std::invalid_argument("invalid encoder_hidden_states shape");
        }

        if (encoder_hidden_states_negative.get_shape() != _encoder_hidden_states_shape)
        {
            throw std::invalid_argument("invalid encoder_hidden_states shape");
        }

        //copy the encoder hidden states to the unet batch2 encoder_hidden_states tensor.
        // This can be done once up-front, as this tensor doesn't change per unet iteration.
        {
            auto encoder_hidden_states_batch2 = _infer_request.get_tensor("encoder_hidden_states");

            auto* pEncoderHiddenStatesBatch2 = encoder_hidden_states_batch2.data<float>();
            auto* pEncoderHiddenStatesPos = encoder_hidden_states_positive.data<float>();
            auto* pEncoderHiddenStatesNeg = encoder_hidden_states_negative.data<float>();

            std::memcpy(pEncoderHiddenStatesBatch2, pEncoderHiddenStatesPos, (encoder_hidden_states_batch2.get_size()/2) * sizeof(float));
            std::memcpy(pEncoderHiddenStatesBatch2 + (encoder_hidden_states_batch2.get_size() / 2), pEncoderHiddenStatesNeg, (encoder_hidden_states_batch2.get_size()/2) * sizeof(float));
        }

        using namespace std::chrono;
        using Clock = std::chrono::high_resolution_clock;
        uint64_t  unet_start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
        for (size_t i = 0; i < timesteps.size(); i++)
        {
            uint64_t  ts = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
            auto t = timesteps[i];
            std::cout << "i =  " << i << ", t = " << t << std::endl;

            auto latent_model_input = ov::Tensor(latents.get_element_type(), latents.get_shape());
            
            //the scheduler scale input modifies latent_model_input in-place. We need to maintain pre-scaled
            // version for some calculations later on, this is why we make a copy of 'latents' here.
            latents.copy_to(latent_model_input);

            scheduler->scale_model_input(latent_model_input, t);

            uint64_t  tus = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

            //expand to batch size 2
            {
                auto latent_model_input_batch2 = _infer_request.get_tensor("latent_model_input");

                auto* pLatentModelInputBatch2 = latent_model_input_batch2.data<float>();
                auto* pLatentModelInputBatch1 = latent_model_input.data<float>();
                std::memcpy(pLatentModelInputBatch2, pLatentModelInputBatch1, latent_model_input.get_size() * sizeof(float));
                std::memcpy(pLatentModelInputBatch2 + latent_model_input.get_size(), pLatentModelInputBatch1, latent_model_input.get_size() * sizeof(float));
            }

            //set the timestamp
            auto timestep_tensor = _infer_request.get_tensor("t");
            auto* pTimestep = timestep_tensor.data<double>();
            *pTimestep = t;

            // run unet
            _infer_request.infer();

            uint64_t  tue = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
            std::cout << "unet portion =  " << (double)(tue - tus) / 1000.0 << " seconds." << std::endl;

            auto noise_pred = _infer_request.get_output_tensor();

            //noise_pred is shape 2,4,64,64, so split this batch2 tensor to produce noise_pred_text & noise_pred_uncond.
            // We just wrap existing tensor buffer here.
            ov::Tensor noise_pred_text, noise_pred_uncond;
            {
                float* pNoisePred = noise_pred.data<float>();
                auto noise_pred_shape = noise_pred.get_shape();
                noise_pred_shape[0] = 1;

                noise_pred_text = ov::Tensor(noise_pred.get_element_type(), noise_pred_shape, pNoisePred);
                noise_pred_uncond = ov::Tensor(noise_pred.get_element_type(), noise_pred_shape, pNoisePred + noise_pred.get_size()/2);
            }

            //noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            auto* pNoisePredUncond = noise_pred_uncond.data<float>();
            auto* pNoisePredText = noise_pred_text.data<float>();
            for (size_t i = 0; i < noise_pred_text.get_size(); i++)
            {
                pNoisePredText[i] = pNoisePredUncond[i] + guidance_scale * (pNoisePredText[i] - pNoisePredUncond[i]);
            }
            
            //run scheduler
            latents = scheduler->step(noise_pred_text, t, latents);
            uint64_t  te = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
            std::cout << "UNet iteration " << i << " took " << (double)(te - ts) / 1000.0 << " seconds. " <<
                "it/s = " << (double)(i + 1) / ((double)(te - unet_start) / 1000.0) << std::endl;

            if (unet_iteration_callback)
            {
                if (!unet_iteration_callback->first(i, timesteps.size(), unet_iteration_callback->second))
                {
                    std::cout << "Cancelled!" << std::endl;
                    return {};
                }
            }
        }
        uint64_t  unet_end = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
        std::cout << "Overall UNet Loop (" << timesteps.size() << ") iterations took " << (double)(unet_end - unet_start) / 1000.0 << " seconds." << std::endl;
        std::cout << "it/s = " << (double)timesteps.size() / ((double)(unet_end - unet_start) / 1000.0) << std::endl;

        return latents;
    }
}