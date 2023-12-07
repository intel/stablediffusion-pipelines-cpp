// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#pragma once
#include <memory>
#include <vector>
#include <string>
#include <optional>

#include <openvino/openvino.hpp>
#include "cpp_stable_diffusion_ov/callbacks.h"
#include "cpp_stable_diffusion_audio_ov/visibility.h"

namespace cpp_stable_diffusion_ov
{
    class StableDiffusionPipeline;
    class SpectrogramImageConverter;

    class CPP_SD_OV_AUDIO_API RiffusionAudioToAudioPipeline
    {
    public:

        RiffusionAudioToAudioPipeline(std::string model_folder,
            std::optional<std::string> cache_folder = {},
            std::string text_encoder_device = "CPU",
            std::string unet_positive_device = "CPU",
            std::string unet_negative_device = "CPU",
            std::string vae_decoder_device = "CPU",
            std::string vae_encoder_device = "CPU");

        typedef bool (*CallbackFuncAudioSegmentComplete)(size_t num_segments_complete,
            size_t num_total_segments,
            std::shared_ptr<std::vector<float>> wav,
            std::shared_ptr<std::vector<uint8_t>> img_rgb,
            size_t img_width,
            size_t img_height,
            void* user);

        std::pair< std::shared_ptr<std::vector<float>>, std::shared_ptr<std::vector<float>> > operator()(
            float* pInput_44100_wav_L, //<- this one is required
            float* pInput_44100_wav_R, //<- this one can be nullptr
            size_t nsamples_to_riffuse,  //i.e. desired # of samples to modify, this is # of samples that will be output
            size_t ntotal_samples_allowed_to_read, // total # of input samples. It will only read more than 'nsamples_to_riffuse'
                                                   // for the last audio chunk -- which will produce better results than 
                                                   // having pipeline pad remainder with 0's.
            const std::string prompt = {},
            std::optional< std::string > negative_prompt = {},
            int num_inference_steps = 50,
            const std::string& scheduler_str = "EulerDiscreteScheduler",
            std::optional< unsigned int > seed = {},
            float guidance_scale = 7.5f,
            float denoising_strength = 0.55f,
            float crossfade_overlap_seconds = 0.2f,
            std::optional<std::pair< CallbackFuncUnetIteration, void*>> unet_iteration_callback = {},
            std::optional<std::pair< CallbackFuncAudioSegmentComplete, void*>> segment_callback = {});


    private:

        std::shared_ptr<StableDiffusionPipeline> _stable_diffusion_pipeline;
        std::shared_ptr<SpectrogramImageConverter> _spec_img_converterL;
        std::shared_ptr<SpectrogramImageConverter> _spec_img_converterR;
    };
}