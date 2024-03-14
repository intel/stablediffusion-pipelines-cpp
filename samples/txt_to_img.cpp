// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#include <iostream>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "cpp_stable_diffusion_ov/model_collateral_cache.h"
#include "cpp_stable_diffusion_ov/stable_diffusion_pipeline.h"
#include "simple_cmdline_parser.h"

void print_usage()
{
    std::cout << "txt_to_image_interpolate usage: " << std::endl;
    std::cout << "--prompt=\"some positive prompt\" " << std::endl;
    std::cout << "--negative_prompt=\"some negative prompt\" " << std::endl;
    std::cout << "--seed=12345 " << std::endl;
    std::cout << "--guidance_scale=8.0 " << std::endl;
    std::cout << "--num_inference_steps=20 " << std::endl;
    std::cout << "--model_dir=\"C:\\Path\\To\\Some\\Model_Dir\" " << std::endl;
    std::cout << "--unet_subdir=\"INT8\" or \"FP16\"" << std::endl;
    std::cout << "--text_encoder_device=CPU" << std::endl;
    std::cout << "--unet_positive_device=CPU" << std::endl;
    std::cout << "--unet_negative_device=CPU" << std::endl;
    std::cout << "--vae_decoder_device=CPU" << std::endl;
    std::cout << "--scheduler=[\"EulerDiscreteScheduler\", \"PNDMScheduler\", \"USTMScheduler\"]" << std::endl;
    std::cout << "--input_image=C:\\SomePath\\img.png" << std::endl;
    std::cout << "--strength=0.75" << std::endl;
    std::cout << "--cancel_after=N (only useful for cancellation feature testing)" << std::endl;
}

struct CancelAfter
{
    bool bCancelAfter = false;
    size_t cancel_after_n_unet_its = 0;
};

static bool UNetCallback(size_t unet_step_i_complete, size_t num_unet_steps, void* user)
{
    CancelAfter* pCancelAfter = (CancelAfter*)user;
    if (pCancelAfter)
    {
        if (pCancelAfter->bCancelAfter)
        {
            if (unet_step_i_complete >= pCancelAfter->cancel_after_n_unet_its)
            {
                std::cout << "UNetCallback: Triggering cancel!" << std::endl;
                return false;
            }
        }
    }

    return true;
}

int main(int argc, char* argv[])
{
    try
    {
        SimpleCmdLineParser cmdline_parser(argc, argv);

        if (cmdline_parser.is_help_needed())
        {
            print_usage();
            return -1;
        }

        std::optional<std::string> prompt;
        std::optional<std::string> negative_prompt;
        std::optional<std::string> seed_str;
        std::optional<std::string> model_dir;
        std::optional<std::string> unet_subdir;
        std::optional<std::string> guidance_scale_str;
        std::optional<std::string> num_inference_steps_str;

        std::optional<std::string> text_encoder_device;
        std::optional<std::string> unet_positive_device;
        std::optional<std::string> unet_negative_device;
        std::optional<std::string> vae_decoder_device;

        std::optional<std::string> scheduler;

        std::optional<std::string> input_image;
        std::optional<std::string> strength;

        std::optional<std::string> cancel_after;

        prompt = cmdline_parser.get_value_for_key("prompt");
        negative_prompt = cmdline_parser.get_value_for_key("negative_prompt");
        seed_str = cmdline_parser.get_value_for_key("seed");
        guidance_scale_str = cmdline_parser.get_value_for_key("guidance_scale");
        num_inference_steps_str = cmdline_parser.get_value_for_key("num_inference_steps");
        model_dir = cmdline_parser.get_value_for_key("model_dir");
        unet_subdir = cmdline_parser.get_value_for_key("unet_subdir");

        text_encoder_device = cmdline_parser.get_value_for_key("text_encoder_device");
        unet_positive_device = cmdline_parser.get_value_for_key("unet_positive_device");
        unet_negative_device = cmdline_parser.get_value_for_key("unet_negative_device");
        vae_decoder_device = cmdline_parser.get_value_for_key("vae_decoder_device");
        scheduler = cmdline_parser.get_value_for_key("scheduler");
        input_image = cmdline_parser.get_value_for_key("input_image");
        strength = cmdline_parser.get_value_for_key("strength");
        cancel_after = cmdline_parser.get_value_for_key("cancel_after");

        if (!text_encoder_device)
            text_encoder_device = "CPU";

        if (!unet_positive_device)
            unet_positive_device = "CPU";

        if (!unet_negative_device)
            unet_negative_device = "CPU";

        if (!vae_decoder_device)
            vae_decoder_device = "CPU";

        if (!scheduler)
            scheduler = "EulerDiscreteScheduler";

        if (!prompt)
        {
            std::cout << "Error! --prompt argument is required" << std::endl;
            print_usage();
            return 1;
        }

        if (!model_dir)
        {
            std::cout << "Error! --model_dir argument is required" << std::endl;
            print_usage();
            return 1;
        }

        int num_inference_steps = 20;
        if (num_inference_steps_str)
        {
            num_inference_steps = std::stoi(*num_inference_steps_str);
        }

        std::optional<unsigned int> seed;
        if (seed_str)
        {
            seed = std::stoul(*seed_str);
        }

        float guidance_scale = 7.5f;
        if (guidance_scale_str)
        {
            guidance_scale = std::stof(*guidance_scale_str);
        }

        CancelAfter cancelAfter;
        if (cancel_after)
        {
            cancelAfter.bCancelAfter = true;
            cancelAfter.cancel_after_n_unet_its = std::stoi(*cancel_after);
        }

        std::optional<std::pair< cpp_stable_diffusion_ov::CallbackFuncUnetIteration, void*>> unet_iteration_callback =
        {
            {UNetCallback, &cancelAfter}
        };

        std::cout << "prompt = \"" << *prompt << "\"" << std::endl;
        if (negative_prompt)
            std::cout << "negative_prompt = \"" << *negative_prompt << "\"" << std::endl;
        else
            std::cout << "negative_prompt = \"\"" << std::endl;
        std::cout << "num_inference_steps = " << num_inference_steps << std::endl;
        if (seed)
            std::cout << "seed = " << *seed << std::endl;
        else
            std::cout << "seed = (not given)" << std::endl;

        std::cout << "guidance_scale = " << guidance_scale << std::endl;
        std::cout << "scheduler = " << *scheduler << std::endl;
    
        using namespace std::chrono;
        using Clock = std::chrono::high_resolution_clock;
        uint64_t  t0 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
        cpp_stable_diffusion_ov::StableDiffusionPipeline sd_pipeline(*model_dir, unet_subdir, {},
            *text_encoder_device, *unet_positive_device, *unet_negative_device, *vae_decoder_device);
        uint64_t  t1 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
        std::cout << "Created StableDiffusionPipeline object in " << (double)(t1 - t0) / 1000.0 << " seconds." << std::endl;


        std::optional< cpp_stable_diffusion_ov::StableDiffusionPipeline::InputImageParams > input_image_params = {};
        if (input_image)
        {
            std::cout << "Using input image: " << *input_image << std::endl;
            auto cvimg = cv::imread(*input_image, cv::IMREAD_COLOR);
            if ((cvimg.rows != 512) || (cvimg.cols != 512))
            {
                throw std::invalid_argument("input image must be 512x512 (for now)");
            }

            auto buf = std::make_shared< std::vector<uint8_t> >(512 * 512 * 3);
            std::memcpy(buf->data(), cvimg.ptr(), 512 * 512 * 3);

            cpp_stable_diffusion_ov::StableDiffusionPipeline::InputImageParams params;
            params.image_buffer = buf;
            params.isBGR = true;
            params.isNHWC = true;
            
            if (strength)
            {
                params.strength = std::stof(*strength);
            }
            else
            {
                params.strength = 0.75f;
            }

            std::cout << "strength =  " << params.strength << std::endl;

            input_image_params = params;
        }
        
        for (int i = 0; i > -1; i++)
        {
            if (i != 0)
            {
                seed = {};
            }

            t0 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

            auto image_buf = sd_pipeline(*prompt,
                negative_prompt,
                num_inference_steps, //num 
                *scheduler,
                seed, //seed
                guidance_scale, //guidance scale
                true, // give us BGR back
                input_image_params,
                unet_iteration_callback);

            if (!image_buf)
            {
                std::cout << "Looks like sd_pipeline was cancelled. Quitting.." << std::endl;
                return 1;
            }

            t1 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
            std::cout << "E2E txt-to-image in " << (double)(t1 - t0) / 1000.0 << " seconds." << std::endl;

            cv::Mat outu8 = cv::Mat(512, 512, CV_8UC3, image_buf->data());
            cv::imwrite("img.png", outu8);
            cv::imshow("Generated Image", outu8);
            std::cout << "Showing Generated image. Also saved as 'img.png'" << std::endl;
            std::cout << "To quit, press 'q'. To generate another image (with a random seed), press 'c'" << std::endl;
            int key = cv::waitKey(0);
            if (key == 'q')
                break;
        }

        //TODO: Ideally we shouldn't need to do this, but it seems that in certain 
        // device driver versions, allowing the model collateral cache to destruct
        // during application shutdown causes some kind of race condition. So, do it 
        // explicitly here. Need to dig deeper into this.
        cpp_stable_diffusion_ov::ModelCollateralCache::instance()->Reset();


    }
    catch (const std::exception& error) {
        std::cout << "in StableDiffusionPipeline routine: exception: " << error.what() << std::endl;
    }

    std::cout << "sd_pipeline done!" << std::endl;

    return 0;
}

