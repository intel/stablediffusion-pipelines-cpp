// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#include "pipelines/unet_loop_sd15_internal_blobs.h"
#include "cpp_stable_diffusion_ov/openvino_model_utils.h"
#include <chrono>

//todo: this should change to header for base scheduler class once it exists.
#include "schedulers/scheduler.h"

#ifdef WIN32
#define OS_SEP "\\"
#else
#define OS_SEP "/"
#endif

namespace cpp_stable_diffusion_ov
{
    UNetLoopSD15InternalBlobs::UNetLoopSD15InternalBlobs(std::string model_dir,
        std::string device_unet_positive_prompt,
        std::string device_unet_negative_prompt,
        size_t max_tok_len,
        size_t img_width,
        size_t img_height,
        ov::Core& core)
        : _device_unet_positive_prompt(device_unet_positive_prompt), _device_unet_negative_prompt(device_unet_negative_prompt)
    {
        using namespace std::chrono;
        using Clock = std::chrono::high_resolution_clock;

        {
            //std::string unet_version = "unet_int8_sq_0.15_tp_input";
            std::string unet_version = "unet_int8_sq_0.15_sym_tp_input";
            std::string model_xml_path = model_dir + OS_SEP + unet_version + ".xml";
            std::string model_blob_path = model_dir + OS_SEP + unet_version + ".blob";

            std::cout << "unet_version" << unet_version << std::endl;

            std::cout << "UNet model info:" << std::endl;

            _sample_shape = ov::Shape{ 1, 4, img_height / 8, img_width / 8 };
            _encoder_hidden_states_shape = ov::Shape{ 1, max_tok_len, 768 };

            {
                ov::CompiledModel compiled_model;

                //if (device_unet_positive_prompt != "NPU")
                if (device_unet_positive_prompt != "NPU")
                {
                    std::cout << "Compiling unet model for " << device_unet_positive_prompt << "..." << std::endl;
                    uint64_t  start_ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

                    compiled_model = core.compile_model(model_xml_path, device_unet_positive_prompt);

                    uint64_t  end_ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
                    std::cout << "Compiling unet model for " << device_unet_positive_prompt 
                        << " took " << end_ms - start_ms << " ms." << std::endl;
                }
                else
                {
                    std::ifstream modelStream(model_blob_path, std::ios_base::binary | std::ios_base::in);
                    if (!modelStream.is_open()) {
                        throw std::runtime_error("Cannot open model file " + model_blob_path);
                    }
                    std::cout << "Importing unet model for " << device_unet_positive_prompt << "..." << std::endl;
                    uint64_t  start_ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

                    compiled_model = core.import_model(modelStream, device_unet_positive_prompt);

                    uint64_t  end_ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
                    std::cout << "Importing unet model for " << device_unet_positive_prompt
                        << " took " << end_ms - start_ms << " ms." << std::endl;
                    modelStream.close();
                }

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
                ov::CompiledModel compiled_model;

                if (device_unet_negative_prompt != "NPU")
                {
                    std::cout << "Compiling unet model for " << device_unet_negative_prompt << "..." << std::endl;
                    uint64_t  start_ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

                    compiled_model = core.compile_model(model_xml_path, device_unet_negative_prompt);

                    uint64_t  end_ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
                    std::cout << "Compiling unet model for " << device_unet_negative_prompt
                        << " took " << end_ms - start_ms << " ms." << std::endl;
                }
                else
                {
                    std::ifstream modelStream(model_blob_path, std::ios_base::binary | std::ios_base::in);
                    if (!modelStream.is_open()) {
                        throw std::runtime_error("Cannot open model file " + model_blob_path);
                    }

                    std::cout << "Importing unet model for " << device_unet_negative_prompt << "..." << std::endl;
                    uint64_t  start_ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

                    compiled_model = core.import_model(modelStream, device_unet_negative_prompt);

                    uint64_t  end_ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
                    std::cout << "Importing unet model for " << device_unet_negative_prompt
                        << " took " << end_ms - start_ms << " ms." << std::endl;
                    modelStream.close();
                }

                _infer_request[1] = compiled_model.create_infer_request();
            }
        }


        {
            std::string model_path = model_dir + OS_SEP + "unet_time_proj.xml";

            //Read the OpenVINO encoder IR (.xml/.bin) from disk, producing an ov::Model object.
            auto model = core.read_model(model_path);

            std::cout << "UNet time proj info:" << std::endl;
            logBasicModelInfo(model);

            if (model->inputs().size() == 1)
            {
                _bneeds_time_constants = false;
            }
            else if (model->inputs().size() == 2)
            {
                _bneeds_time_constants = true;
            }
            else
            {
                throw std::runtime_error("Expected unet_time_proj model to have either 1 or 2 outputs, but it has "
                    + std::to_string(model->inputs().size()));
            }


            std::cout << "Compiling unet time proj model for CPU ..." << std::endl;
            auto compiled_model = core.compile_model(model, "CPU");
            std::cout << "Compiling unet time proj model for CPU done" << std::endl;

            _unet_time_proj_request = compiled_model.create_infer_request();
        }

        if (_bneeds_time_constants)
        {
            ov::Shape time_constants_shape = { 1, 160 };
            _time_constants = ov::Tensor(ov::element::f32, time_constants_shape);
            load_tensor_from_disk(_time_constants, model_dir + OS_SEP + "time_proj_constants.raw");
            _sin_t = ov::Tensor(ov::element::f32, _time_constants.get_shape());
            _cos_t = ov::Tensor(ov::element::f32, _time_constants.get_shape());
        }
    }

    ov::Tensor UNetLoopSD15InternalBlobs::operator()(const std::vector<double>& timesteps, ov::Tensor latents, ov::Tensor encoder_hidden_states_positive,
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

        //save_tensor_to_disk(encoder_hidden_states_positive, "encoder_hidden_states_positive.raw");
        //save_tensor_to_disk(encoder_hidden_states_negative, "encoder_hidden_states_negative.raw");

        _infer_request[0].set_tensor("encoder_hidden_states", encoder_hidden_states_positive);
        _infer_request[1].set_tensor("encoder_hidden_states", encoder_hidden_states_negative);

        for (size_t i = 0; i < timesteps.size(); i++)
        {
            uint64_t  ts = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
            auto t = timesteps[i];
            std::cout << "i =  " << i << ", t = " << t << std::endl;

            if (_bneeds_time_constants)
            {
                //fill sin_t and cos_t
                float* pTimeConstants = _time_constants.data<float>();
                float* pSin = _sin_t.data<float>();
                float* pCos = _cos_t.data<float>();
                for (size_t p = 0; p < _time_constants.get_size(); p++)
                {
                    float t_scaled = pTimeConstants[p] * (float)t;
                    pSin[p] = std::sin(t_scaled);
                    pCos[p] = std::cos(t_scaled);
                }

                _unet_time_proj_request.set_tensor("sine_t", _sin_t);
                _unet_time_proj_request.set_tensor("cosine_t", _cos_t);
            }
            else
            {
                auto t_tensor = _unet_time_proj_request.get_input_tensor();
                float* pT = t_tensor.data<float>();
                *pT = (float)t;
            }

            _unet_time_proj_request.infer();
            auto time_proj = _unet_time_proj_request.get_output_tensor();

            //save_tensor_to_disk(time_proj, "time_proj" + std::to_string(i) + ".raw");

            auto latent_model_input = ov::Tensor(latents.get_element_type(), latents.get_shape());
            latents.copy_to(latent_model_input);

            // we don't expand the latents to batch size 2, as it does in pipeline_stable_diffusion.py,
            // as we will run unet twice (for +/- prompt). This allows us to split across two different
            // devices.

            //save_tensor_to_disk(latent_model_input, "input_latents_latents" + std::to_string(i) + ".raw");

            scheduler->scale_model_input(latent_model_input, t);

            //save_tensor_to_disk(latent_model_input, "input_latents_latents_scaled" + std::to_string(i) + ".raw");

            uint64_t  tus = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

            // run positive prompt
            _run_unet_async(time_proj, latent_model_input, t, encoder_hidden_states_positive, 0);


            //if + and - prompt devices are equal, don't kick off both simultaneously.
            // It could be in the future on something like dGPU there could be an advantage
            // to doing that, but there hasn't been any advantage discovered yet -- and probably
            // just complicates the scheduling a bit? Could be wrong.
            if (_device_unet_positive_prompt == _device_unet_negative_prompt)
            {
                _infer_request[0].wait();
            }

            // run negative prompt
            _run_unet_async(time_proj, latent_model_input, t, encoder_hidden_states_negative, 1);

            if (_device_unet_positive_prompt != _device_unet_negative_prompt)
            {
                _infer_request[0].wait();
            }

            _infer_request[1].wait();
            uint64_t  tue = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
            std::cout << "unet portion =  " << (double)(tue - tus) / 1000.0 << " seconds." << std::endl;

            auto noise_pred_text = _infer_request[0].get_output_tensor();
            auto noise_pred_uncond = _infer_request[1].get_output_tensor();

            //save_tensor_to_disk(noise_pred_text, "noise_pred_text_" + std::to_string(i) + ".raw");
            //save_tensor_to_disk(noise_pred_uncond, "noise_pred_uncond_" + std::to_string(i) + ".raw");

            //noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            auto* pNoisePredUncond = noise_pred_uncond.data<float>();
            auto* pNoisePredText = noise_pred_text.data<float>();
            for (size_t i = 0; i < noise_pred_text.get_size(); i++)
            {
                pNoisePredText[i] = pNoisePredUncond[i] + guidance_scale * (pNoisePredText[i] - pNoisePredUncond[i]);
            }

            //save_tensor_to_disk(noise_pred_text, "latents_sched_in_" + std::to_string(i) + ".raw");

            //run scheduler
            latents = scheduler->step(noise_pred_text, t, latents);

            //save_tensor_to_disk(latents, "latents_sched_out_" + std::to_string(i) + ".raw");

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

    void UNetLoopSD15InternalBlobs::_run_unet_async(ov::Tensor time_proj, ov::Tensor sample, int64_t timestep, ov::Tensor encoder_hidden_states, size_t infer_request_index)
    {
        if (sample.get_shape() != _sample_shape)
        {
            throw std::invalid_argument("invalid sample shape");
        }

        if (encoder_hidden_states.get_shape() != _encoder_hidden_states_shape)
        {
            throw std::invalid_argument("invalid encoder_hidden_states shape");
        }

#if 1
        auto latent_model_input = _infer_request[infer_request_index].get_tensor("latent_model_input");
        auto time_proj_input = _infer_request[infer_request_index].get_tensor("time_proj");

        time_proj.copy_to(time_proj_input);
        sample.copy_to(latent_model_input);
#else

        _infer_request[infer_request_index].set_tensor("latent_model_input", sample);
        _infer_request[infer_request_index].set_tensor("time_proj", time_proj);
#endif
        //_infer_request[infer_request_index].set_tensor("encoder_hidden_states", encoder_hidden_states);

        _infer_request[infer_request_index].start_async();

        //ov::Tensor out_tensor = _infer_request[infer_request_index].get_tensor("out_sample");

        //std::cout << "out_tensor ptr = " << (void *)out_tensor.data<float>() << std::endl;
        //save_tensor_to_disk(out_tensor, "out_tensor.raw");

        //return out_tensor;
    }
}