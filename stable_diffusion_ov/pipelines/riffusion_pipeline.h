#pragma once

#include <memory>
#include <vector>
#include <string>
#include <optional>
#include <openvino/openvino.hpp>
#include "utils/callbacks.h"

class CLIPTokenizer;
class OpenVINOTextEncoder;
class UNetLoop;
class OpenVINOVAEDecoder;
class OpenVINOVAEEncoder;
class PNDMScheduler;


//todo: There should be a base class that includes alot of the basics (like prompt encode, unet loop, etc.)
// Right now this is very redundant as compared with StableDiffusionPipeline class.
class RiffusionPipeline
{
public:

    RiffusionPipeline(std::string model_folder,
        std::optional< std::string > cache = {},
        std::string text_encoder_device = "CPU",
        std::string unet_positive_device = "CPU",
        std::string unet_negative_device = "CPU",
        std::string vae_decoder_device = "CPU",
        std::string vae_encoder_device = "CPU");

    typedef bool (*CallbackFuncInterpolationIteration)(size_t interp_step_i_complete,
        size_t num_interp_steps,
        std::shared_ptr<std::vector<float>> wav,
        std::shared_ptr<std::vector<uint8_t>> img_rgb,
        size_t img_width,
        size_t img_height,
        void* user);

    std::pair< std::shared_ptr<std::vector<float>>, std::shared_ptr<std::vector<float>> > operator()(
        bool bStereo,
        const std::string prompt_start,
        std::optional< std::string > negative_prompt_start = {},
        std::optional< std::string > prompt_end = {},
        std::optional< std::string > negative_prompt_ebd = {},
        std::optional< unsigned int > seed_start = {},
        std::optional< unsigned int > seed_end = {},
        float denoising_start = 0.75f,
        float denoising_end = 0.75f,
        float guidance_scale_start = 7.5f,
        float guidance_scale_end = 7.5f,
        int num_inference_steps_per_sample = 50,
        int num_interpolation_steps = 5,
        std::string seed_image = "og_beat",
        float alpha_power = 1.0f,
        const std::string& scheduler_str = "USTMScheduler",
        std::optional<std::pair< CallbackFuncUnetIteration, void*>> unet_iteration_callback = {},
        std::optional<std::pair< CallbackFuncInterpolationIteration, void*>> interp_iteration_callback = {}
        );


private:

    std::shared_ptr<CLIPTokenizer> _tokenizer;
    std::shared_ptr<OpenVINOTextEncoder> _text_encoder;
    std::shared_ptr<UNetLoop> _unet_loop;
    std::shared_ptr<OpenVINOVAEDecoder> _vae_decoder;
    std::shared_ptr<OpenVINOVAEEncoder> _vae_encoder;

    void _embed_text(const std::string prompt,
        ov::Tensor& text_embeds);

    std::shared_ptr<std::vector<uint8_t>> _interpolate_img2img(ov::Tensor& text_embedding,
        ov::Tensor init_latents,
        ov::Tensor& uncond_embeddings,
        std::optional< unsigned int > seed_start,
        std::optional< unsigned int > seed_end,
        float interpolate_alpha,
        float strength_a,
        float strength_b,
        int num_inference_steps,
        float guidance_scale,
        const std::string& scheduler_str,
        std::optional<std::pair< CallbackFuncUnetIteration, void*>> unet_iteration_callback
    );

    size_t _width = 512;
    size_t _height = 512;

    int64_t _tok_max_length = 0;

    std::string _model_folder;
};