// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#include "pipelines/unet_loop_split.h"
#include "cpp_stable_diffusion_ov/openvino_model_utils.h"
#include <chrono>

//todo: this should change to header for base scheduler class once it exists.
#include "schedulers/scheduler.h"

namespace cpp_stable_diffusion_ov
{
    UNetLoopSplit::UNetLoopSplit(std::string model_path,
        std::string device_unet_positive_prompt,
        std::string device_unet_negative_prompt,
        size_t max_tok_len,
        size_t img_width,
        size_t img_height,
        ov::Core& core)
        : _device_unet_positive_prompt(device_unet_positive_prompt), _device_unet_negative_prompt(device_unet_negative_prompt)
    {
        //Read the OpenVINO encoder IR (.xml/.bin) from disk, producing an ov::Model object.
        auto model = core.read_model(model_path);

        logBasicModelInfo(model);

        _sample_shape = ov::Shape{ 1, 4, img_height / 8, img_width / 8 };
        _encoder_hidden_states_shape = ov::Shape{ 1, max_tok_len, 768 };
        std::map<size_t, ov::PartialShape> reshape_map;
        reshape_map[0] = _sample_shape;
        reshape_map[2] = _encoder_hidden_states_shape;
        model->reshape(reshape_map);

        //ov::preprocess::PrePostProcessor ppp(model);
        //ppp.input("timestep").tensor().set_element_type(ov::element::f64);
        //model = ppp.build();

        std::cout << "UNet model info:" << std::endl;
        logBasicModelInfo(model);

        {
            std::cout << "Compiling unet model for " << device_unet_positive_prompt << "..." << std::endl;
            auto compiled_model = core.compile_model(model, device_unet_positive_prompt);
            std::cout << "Compiling unet model for " << device_unet_positive_prompt << " done" << std::endl;
            _infer_request[0] = compiled_model.create_infer_request();

            //if our negative prompt is equal to positive prompt. Just use same compiled model and create
            // new infer request.
            if (device_unet_negative_prompt == device_unet_positive_prompt)
            {
                _infer_request[1] = compiled_model.create_infer_request();
            }
        }

        if (device_unet_negative_prompt != device_unet_positive_prompt)
        {
            std::cout << "Compiling unet model for " << device_unet_negative_prompt << "..." << std::endl;
            auto compiled_model = core.compile_model(model, device_unet_negative_prompt);
            std::cout << "Compiling unet model for " << device_unet_negative_prompt << " done" << std::endl;

            _infer_request[1] = compiled_model.create_infer_request();
        }
    }

    ov::Tensor UNetLoopSplit::operator()(const std::vector<double>& timesteps, ov::Tensor latents, ov::Tensor encoder_hidden_states_positive,
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

        if (encoder_hidden_states_positive.get_shape() != _encoder_hidden_states_shape)
        {
            throw std::invalid_argument("invalid encoder_hidden_states shape");
        }

        using namespace std::chrono;
        using Clock = std::chrono::high_resolution_clock;
        uint64_t  unet_start = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
        for (size_t i = 0; i < timesteps.size(); i++)
        {
            uint64_t  ts = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
            auto t = timesteps[i];
            std::cout << "i =  " << i << ", t = " << t << std::endl;

            // we don't expand the latents to batch size 2, as it does in pipeline_stable_diffusion.py,
            // as we will run unet twice (for +/- prompt). This allows us to split across two different 
            // devices.

            auto latent_model_input = ov::Tensor(latents.get_element_type(), latents.get_shape());
            latents.copy_to(latent_model_input);

            //save_tensor_to_disk(latent_model_input, "latent_model_input" + std::to_string(i) + ".raw");

            scheduler->scale_model_input(latent_model_input, t);

            uint64_t  tus = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

            // run positive prompt
            _run_unet_async(latent_model_input, t, encoder_hidden_states_positive, 0);


            //if + and - prompt devices are equal, don't kick off both simultaneously.
            // It could be in the future on something like dGPU there could be an advantage
            // to doing that, but there hasn't been any advantage discovered yet -- and probably
            // just complicates the scheduling a bit? Could be wrong.
            if (_device_unet_positive_prompt == _device_unet_negative_prompt)
            {
                _infer_request[0].wait();
            }

            // run negative prompt
            _run_unet_async(latent_model_input, t, encoder_hidden_states_negative, 1);

            if (_device_unet_positive_prompt != _device_unet_negative_prompt)
            {
                _infer_request[0].wait();
            }

            _infer_request[1].wait();
            uint64_t  tue = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
            std::cout << "unet portion =  " << (double)(tue - tus) / 1000.0 << " seconds." << std::endl;

            auto noise_pred_text = _infer_request[0].get_output_tensor();
            auto noise_pred_uncond = _infer_request[1].get_output_tensor();

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

    void UNetLoopSplit::_run_unet_async(ov::Tensor sample, double timestep, ov::Tensor encoder_hidden_states, size_t infer_request_index)
    {
        if (sample.get_shape() != _sample_shape)
        {
            throw std::invalid_argument("invalid sample shape");
        }

        if (encoder_hidden_states.get_shape() != _encoder_hidden_states_shape)
        {
            throw std::invalid_argument("invalid encoder_hidden_states shape");
        }

        auto timestep_tensor = _infer_request[infer_request_index].get_tensor("t");
        auto* pTimestep = timestep_tensor.data<double>();
        *pTimestep = timestep;

        _infer_request[infer_request_index].set_input_tensor(0, sample);
        _infer_request[infer_request_index].set_input_tensor(2, encoder_hidden_states);

        _infer_request[infer_request_index].start_async();

        //ov::Tensor out_tensor = _infer_request[infer_request_index].get_tensor("out_sample");

        //std::cout << "out_tensor ptr = " << (void *)out_tensor.data<float>() << std::endl;
        //save_tensor_to_disk(out_tensor, "out_tensor.raw");

        //return out_tensor;
    }
}