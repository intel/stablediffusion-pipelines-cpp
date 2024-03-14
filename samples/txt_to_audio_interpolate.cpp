// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#include <iostream>
#include "cpp_stable_diffusion_audio_ov/stable_diffusion_audio_interpolation_pipeline.h"
#include "simple_cmdline_parser.h"
#include <fstream>
#include "cpp_stable_diffusion_audio_ov/wav_util.h"
#include "cpp_stable_diffusion_ov/model_collateral_cache.h"

bool MyInterpolationCallback(size_t interp_step_i_complete,
    size_t num_interp_steps,
    std::shared_ptr<std::vector<float>> wav,
    std::shared_ptr<std::vector<uint8_t>> img_rgb,
    size_t img_width,
    size_t img_height,
    void* user)
{
    std::cout << "Interpolation iteration " << interp_step_i_complete << " / " << num_interp_steps << " complete." << std::endl;
    if( wav )
       std::cout << "nsamples = " << wav->size() << std::endl;
    return true;
}

struct CancelAfter
{
    bool bCancelAfter = false;
    size_t cancel_after_n_unet_its = 0;
};

bool MyUnetCallback(size_t unet_step_i_complete,
    size_t num_unet_steps,
    void* user)
{
    std::cout << "unet iteration " << unet_step_i_complete << " / " << num_unet_steps << " complete." << std::endl;

    CancelAfter* pCancelAfter = (CancelAfter*)user;
    if (pCancelAfter)
    {
        if (pCancelAfter->bCancelAfter)
        {
            if (unet_step_i_complete >= pCancelAfter->cancel_after_n_unet_its)
            {
                std::cout << "MyUnetCallback: Triggering cancel!" << std::endl;
                return false;
            }
        }
    }

    return true;
}

void print_usage()
{
    std::cout << "txt_to_audio_interpolate usage: " << std::endl;
    std::cout << "--output_wav=output.wav" << std::endl;
    std::cout << "--prompt_start=\"bubblegum eurodance\" " << std::endl;
    std::cout << "--prompt_end=\"acoustic folk violin jam\" " << std::endl;
    std::cout << "--negative_prompt=\"some negative prompt\" " << std::endl;
    std::cout << "--seed_start=12345 " << std::endl;
    std::cout << "--seed_end=23456 " << std::endl;
    std::cout << "--strength_start=0.75" << std::endl;
    std::cout << "--strength_end=0.75" << std::endl;
    std::cout << "--guidance_scale_start=7.0 " << std::endl;
    std::cout << "--guidance_scale_end=7.0 " << std::endl;
    std::cout << "--num_inference_steps=20 " << std::endl;
    std::cout << "--model_dir=\"C:\\Path\\To\\Some\\Model_Dir\" " << std::endl;
    std::cout << "--unet_subdir=\"INT8\" or \"FP16\"" << std::endl;
    std::cout << "--text_encoder_device=CPU" << std::endl;
    std::cout << "--unet_positive_device=CPU" << std::endl;
    std::cout << "--unet_negative_device=CPU" << std::endl;
    std::cout << "--vae_decoder_device=CPU" << std::endl;
    std::cout << "--vae_encoder_device=CPU" << std::endl;
    std::cout << "--scheduler=[\"EulerDiscreteScheduler\", \"PNDMScheduler\", \"USTMScheduler\"]" << std::endl;
    std::cout << "--interpolation_steps=5" << std::endl;
    std::cout << "--num_output_segments=2" << std::endl;
    std::cout << "--mono" << std::endl;
    std::cout << "--cancel_after=N (only useful for cancellation feature testing)" << std::endl;
}

int main(int argc, char* argv[])
{
    //RiffusionPipeline riffusion_pipeline("C:\\Users\\rdmetca\\riffusion_models");
    try
    {
        SimpleCmdLineParser cmdline_parser(argc, argv);
        if (cmdline_parser.is_help_needed())
        {
            print_usage();
            return -1;
        }

        std::optional<std::string> prompt_start;
        std::optional<std::string> prompt_end;
        std::optional<std::string> negative_prompt;
        std::optional<std::string> seed_start_str;
        std::optional<std::string> seed_end_str;
        std::optional<std::string> model_dir;
        std::optional<std::string> unet_subdir;
        std::optional<std::string> guidance_scale_start_str;
        std::optional<std::string> guidance_scale_end_str;
        std::optional<std::string> num_inference_steps_str;
        std::optional<std::string> num_output_segments_str;

        std::optional<std::string> text_encoder_device;
        std::optional<std::string> unet_positive_device;
        std::optional<std::string> unet_negative_device;
        std::optional<std::string> vae_decoder_device;
        std::optional<std::string> vae_encoder_device;

        std::optional<std::string> scheduler;
        std::optional<std::string> strength_start_str;
        std::optional<std::string> strength_end_str;
        std::optional<std::string> interpolation_steps_str;
        std::optional<std::string> output_wav;

        std::optional<std::string> cancel_after;

        prompt_start = cmdline_parser.get_value_for_key("prompt_start");
        prompt_end = cmdline_parser.get_value_for_key("prompt_end");
        negative_prompt = cmdline_parser.get_value_for_key("negative_prompt");
        seed_start_str = cmdline_parser.get_value_for_key("seed_start");
        seed_end_str = cmdline_parser.get_value_for_key("seed_end");
        guidance_scale_start_str = cmdline_parser.get_value_for_key("guidance_scale_start");
        guidance_scale_end_str = cmdline_parser.get_value_for_key("guidance_scale_end");
        num_inference_steps_str = cmdline_parser.get_value_for_key("num_inference_steps");
        model_dir = cmdline_parser.get_value_for_key("model_dir");
        unet_subdir = cmdline_parser.get_value_for_key("unet_subdir");
        text_encoder_device = cmdline_parser.get_value_for_key("text_encoder_device");
        unet_positive_device = cmdline_parser.get_value_for_key("unet_positive_device");
        unet_negative_device = cmdline_parser.get_value_for_key("unet_negative_device");
        vae_decoder_device = cmdline_parser.get_value_for_key("vae_decoder_device");
        vae_encoder_device = cmdline_parser.get_value_for_key("vae_decoder_device");
        scheduler = cmdline_parser.get_value_for_key("scheduler");
        strength_start_str = cmdline_parser.get_value_for_key("strength_start");
        strength_end_str = cmdline_parser.get_value_for_key("strength_end");
        output_wav = cmdline_parser.get_value_for_key("output_wav");
        interpolation_steps_str = cmdline_parser.get_value_for_key("interpolation_steps");
        num_output_segments_str = cmdline_parser.get_value_for_key("num_output_segments");
        cancel_after = cmdline_parser.get_value_for_key("cancel_after");

        bool bMono = cmdline_parser.was_key_given("mono");

        if (!prompt_start)
        {
            std::cout << "Error! --prompt_start argument is required" << std::endl;
            print_usage();
            return 1;
        }

        if (!model_dir)
        {
            std::cout << "Error! --model_dir argument is required" << std::endl;
            print_usage();
            return 1;
        }

        if (!output_wav)
        {
            std::cout << "warning, --output_wav wasn't specified, so output will not be saved." << std::endl;
        }

        if (!text_encoder_device)
            text_encoder_device = "CPU";

        if (!unet_positive_device)
            unet_positive_device = "CPU";

        if (!unet_negative_device)
            unet_negative_device = "CPU";

        if (!vae_decoder_device)
            vae_decoder_device = "CPU";

        if (!vae_encoder_device)
            vae_encoder_device = "CPU";

        if (!scheduler)
            scheduler = "EulerDiscreteScheduler";

        int num_inference_steps = 20;
        if (num_inference_steps_str)
        {
            num_inference_steps = std::stoi(*num_inference_steps_str);
        }

        std::optional<unsigned int> seed_start;
        if (seed_start_str)
        {
            seed_start = std::stoul(*seed_start_str);
        }

        std::optional<unsigned int> seed_end;
        if (seed_end_str)
        {
            seed_end = std::stoul(*seed_end_str);
        }

        float guidance_scale_start = 7.5f;
        if (guidance_scale_start_str)
        {
            guidance_scale_start = std::stof(*guidance_scale_start_str);
        }

        float guidance_scale_end = 7.5f;
        if (guidance_scale_end_str)
        {
            guidance_scale_end = std::stof(*guidance_scale_end_str);
        }

        float strength_start = 0.75f;
        if (strength_start_str)
        {
            strength_start = std::stof(*strength_start_str);
        }

        float strength_end = 0.75f;
        if (strength_end_str)
        {
            strength_end = std::stof(*strength_end_str);
        }

        int interpolation_steps = 5;
        if (interpolation_steps_str)
        {
            interpolation_steps = std::stoi(*interpolation_steps_str);
        }

        std::optional<int> num_output_segments;
        if (num_output_segments_str)
        {
            num_output_segments = std::stoi(*num_output_segments_str);
        }

        CancelAfter cancelAfter;
        if (cancel_after)
        {
            cancelAfter.bCancelAfter = true;
            cancelAfter.cancel_after_n_unet_its = std::stoi(*cancel_after);
        }
  
        cpp_stable_diffusion_ov::StableDiffusionAudioInterpolationPipeline riffusion_pipeline(*model_dir, unet_subdir, 
            {},
            *text_encoder_device,
            *unet_positive_device,
            *unet_negative_device,
            *vae_decoder_device,
            *vae_encoder_device);

        std::pair< cpp_stable_diffusion_ov::CallbackFuncUnetIteration, void*> mycallback = { MyUnetCallback, &cancelAfter };
        std::pair< cpp_stable_diffusion_ov::StableDiffusionAudioInterpolationPipeline::CallbackFuncInterpolationIteration, void*> myIntcallback = { MyInterpolationCallback, nullptr };

        auto out_samples = riffusion_pipeline(!bMono,
            *prompt_start,
            negative_prompt,
            prompt_end,
            {},
            seed_start,
            seed_end,
            strength_start, //denoising start
            strength_end, //denosing end
            guidance_scale_start,
            guidance_scale_end,
            num_inference_steps, //inference steps per sample
            interpolation_steps, //interpolation steps
            num_output_segments,
            "og_beat",
            1.0f,
            *scheduler,
            mycallback,
            myIntcallback);

        if (!out_samples.first)
        {
            std::cout << "outsamples.first is NULL. Probably cancel was triggered?" << std::endl;
            return 0;
        }

        if (output_wav)
        {
            cpp_stable_diffusion_ov::WriteWav(*output_wav, out_samples);
        }

        //TODO: Ideally we shouldn't need to do this, but it seems that in certain 
        // device driver versions, allowing the model collateral cache to destruct
        // during application shutdown causes some kind of race condition. So, do it 
        // explicitly here. Need to dig deeper into this.
        cpp_stable_diffusion_ov::ModelCollateralCache::instance()->Reset();


    }
    catch (const std::exception& error) {
        std::cout << "in RiffusionPipeline routine: exception: " << error.what() << std::endl;
    }

    return 0;
}


