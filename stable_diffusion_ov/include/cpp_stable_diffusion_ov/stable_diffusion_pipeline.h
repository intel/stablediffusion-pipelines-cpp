// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#pragma once

#include <memory>
#include <vector>
#include <string>
#include <optional>
#include <openvino/openvino.hpp>
#include "cpp_stable_diffusion_ov/callbacks.h"
#include "cpp_stable_diffusion_ov/visibility.h"

namespace cpp_stable_diffusion_ov
{
    class CLIPTokenizer;
    class OpenVINOTextEncoder;
    class UNetLoop;
    class OpenVINOVAEDecoder;
    class Scheduler;
    class OpenVINOVAEEncoder;


    class CPP_SD_OV_API StableDiffusionPipeline
    {
    public:

        StableDiffusionPipeline(std::string model_folder,
            std::optional< std::string > unet_subdir,
            std::optional< std::string > cache = {},
            std::string text_encoder_device = "CPU",
            std::string unet_positive_device = "CPU",
            std::string unet_negative_device = "CPU",
            std::string vae_decoder_device = "CPU",
            std::string vae_encoder_device = "CPU");

        struct InputImageParams
        {
            std::shared_ptr<std::vector<uint8_t>> image_buffer = {};
            float strength = 1.f;
            bool isBGR = false; //is buffer BGR order?
            bool isNHWC = true; //is buffer NHWC layout?
        };

        //Generate an image. This will return a 512x512 image buffer.
        // For example, you could wrap the result as cv::Mat like this:
        // (assuming image_buf is the return of this function)
        // cv::Mat image = cv::Mat(512, 512, CV_8UC3, image_buf->data());
        std::shared_ptr<std::vector<uint8_t>> operator()(
            const std::string prompt,
            std::optional< std::string > negative_prompt = {},
            int num_inference_steps = 50,
            const std::string& scheduler_str = "EulerDiscreteScheduler",
            std::optional< unsigned int > seed = {},
            float guidance_scale = 7.5f,
            bool bGiveBGR = true,
            std::optional< InputImageParams > input_image_params = {},
            std::optional<std::pair< CallbackFuncUnetIteration, void*>> unet_iteration_callback = {});

    protected:

        std::shared_ptr<CLIPTokenizer> _tokenizer;
        std::shared_ptr<OpenVINOTextEncoder> _text_encoder;
        std::shared_ptr<UNetLoop> _unet_loop;
        std::shared_ptr<OpenVINOVAEDecoder> _vae_decoder;
        std::shared_ptr<OpenVINOVAEEncoder> _vae_encoder;

        // returns a pair of embeddings. 
        // pair.first is positive prompt embedding
        // pair.second is negative prompt embedding
        // if do_classifier_free_guidance is false, pair.second
        // will be an empty tensor.
        std::pair<ov::Tensor, ov::Tensor> _encode_prompt(const std::string prompt,
            std::optional< std::string > negative_prompt,
            bool do_classifier_free_guidance);

        ov::Tensor _encode_prompt(const std::string prompt);

        ov::Tensor _prepare_latents(std::optional< ov::Tensor > vae_encoded,
            double latent_timestep,
            std::shared_ptr<Scheduler> scheduler,
            std::optional< unsigned int > seed);

        ov::Tensor _vae_encode(InputImageParams& input_image_params);

        std::shared_ptr<std::vector<uint8_t>> _post_proc(ov::Tensor latents,
            bool bGiveBGR = false);

        size_t _width = 512;
        size_t _height = 512;

        int64_t _tok_max_length = 0;

        std::vector<double> _get_timesteps(float strength, std::shared_ptr<Scheduler> scheduler);

        ov::Tensor _preprocess(InputImageParams& image_params);

        std::string _model_folder;
    };
}