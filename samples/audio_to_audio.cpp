#include <iostream>
#include <vector>
#include <memory>
#include <fstream>
#include "cpp_stable_diffusion_audio_ov/riffusion_audio_to_audio_pipeline.h"
#include "simple_cmdline_parser.h"
#include "cpp_stable_diffusion_audio_ov/wav_util.h"
#include "cpp_stable_diffusion_ov/model_collateral_cache.h"

void print_usage()
{
    std::cout << "audio_to_audio usage: " << std::endl;
    std::cout << "--input_wav=input_1channel_44khz_16bitPCM.wav" << std::endl;
    std::cout << "--output_wav=output.wav" << std::endl;
    std::cout << "--prompt=\"some positive prompt\" " << std::endl;
    std::cout << "--negative_prompt=\"some negative prompt\" " << std::endl;
    std::cout << "--seed=12345 " << std::endl;
    std::cout << "--guidance_scale=7.0 " << std::endl;
    std::cout << "--num_inference_steps=20 " << std::endl;
    std::cout << "--model_dir=\"C:\\Path\\To\\Some\\Model_Dir\" " << std::endl;
    std::cout << "--text_encoder_device=CPU" << std::endl;
    std::cout << "--unet_positive_device=CPU" << std::endl;
    std::cout << "--unet_negative_device=CPU" << std::endl;
    std::cout << "--vae_decoder_device=CPU" << std::endl;
    std::cout << "--vae_encoder_device=CPU" << std::endl;
    std::cout << "--scheduler=[\"EulerDiscreteScheduler\", \"PNDMScheduler\", \"USTMScheduler\"]" << std::endl;
    std::cout << "--strength=0.55" << std::endl;
    std::cout << "--duration_secs=7.0" << std::endl;

}

static bool SegmentCompleteCallback(size_t num_segments_complete,
    size_t num_total_segments,
    std::shared_ptr<std::vector<float>> wav,
    std::shared_ptr<std::vector<uint8_t>> img_rgb,
    size_t img_width,
    size_t img_height,
    void* user)
{
    std::cout << "SegmentCompleteCallback: " <<
        ((double)num_segments_complete / (double)num_total_segments) * 100 << "% complete." << std::endl;

    return true;
}

static bool UnetCallback(size_t unet_step_i_complete,
    size_t num_unet_steps,
    void* user)
{
    std::cout << "unet iteration " << unet_step_i_complete + 1 << " / " << num_unet_steps << " complete." << std::endl;
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
        std::optional<std::string> guidance_scale_str;
        std::optional<std::string> num_inference_steps_str;

        std::optional<std::string> text_encoder_device;
        std::optional<std::string> unet_positive_device;
        std::optional<std::string> unet_negative_device;
        std::optional<std::string> vae_decoder_device;
        std::optional<std::string> vae_encoder_device;

        std::optional<std::string> scheduler;
        std::optional<std::string> input_wav;
        std::optional<std::string> strength_str;

        std::optional<std::string> output_wav;
        std::optional<std::string> duration_str;

        prompt = cmdline_parser.get_value_for_key("prompt");
        negative_prompt = cmdline_parser.get_value_for_key("negative_prompt");
        seed_str = cmdline_parser.get_value_for_key("seed");
        guidance_scale_str = cmdline_parser.get_value_for_key("guidance_scale");
        num_inference_steps_str = cmdline_parser.get_value_for_key("num_inference_steps");
        model_dir = cmdline_parser.get_value_for_key("model_dir");
        text_encoder_device = cmdline_parser.get_value_for_key("text_encoder_device");
        unet_positive_device = cmdline_parser.get_value_for_key("unet_positive_device");
        unet_negative_device = cmdline_parser.get_value_for_key("unet_negative_device");
        vae_decoder_device = cmdline_parser.get_value_for_key("vae_decoder_device");
        vae_encoder_device = cmdline_parser.get_value_for_key("vae_decoder_device");
        scheduler = cmdline_parser.get_value_for_key("scheduler");
        input_wav = cmdline_parser.get_value_for_key("input_wav");
        strength_str = cmdline_parser.get_value_for_key("strength");
        output_wav = cmdline_parser.get_value_for_key("output_wav");
        duration_str = cmdline_parser.get_value_for_key("duration_secs");

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

        if (!input_wav)
        {
            std::cout << "Error! --input_wav argument is required" << std::endl;
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

        float strength = 0.55f;
        if( strength_str )
        {
            strength = std::stof(*strength_str);
        }
    
        auto wav_sample_pair = cpp_stable_diffusion_ov::ReadWav(*input_wav);
        auto input_L = wav_sample_pair.first;
        auto input_R = wav_sample_pair.second;

        if (input_R)
        {
            std::cout << "Stereo Input." << std::endl;
        }
        else
        {
            std::cout << "Mono Input." << std::endl;
        }


        size_t nsamples_to_riffuse = input_L->size();
        if (duration_str)
        {
            size_t duration_in_samples = std::stoul(*duration_str) * 44100;
            if (duration_in_samples > nsamples_to_riffuse)
            {
                std::cout << "Error! Specified --duration_secs has longer duration than given input wav file." << std::endl;
                return 1;
            }

            nsamples_to_riffuse = duration_in_samples;
        }

        std::cout << "nsamples_to_riffuse = " << nsamples_to_riffuse << std::endl;
        std::cout << "samples allowed to read = " << input_L->size() << std::endl;

        std::cout << "creating pipeline using model_dir = " << *model_dir << std::endl;
        cpp_stable_diffusion_ov::RiffusionAudioToAudioPipeline pipeline(*model_dir, {},
            *text_encoder_device,
            *unet_positive_device,
            *unet_negative_device,
            *vae_decoder_device,
            *vae_encoder_device);

        std::pair< cpp_stable_diffusion_ov::CallbackFuncUnetIteration, void*> unet_callback = { UnetCallback, nullptr };
        std::pair< cpp_stable_diffusion_ov::RiffusionAudioToAudioPipeline::CallbackFuncAudioSegmentComplete, void*> seg_callback = { SegmentCompleteCallback, nullptr };
        auto out_samples = pipeline(input_L->data(),
            input_R ? input_R->data() : nullptr,
            nsamples_to_riffuse,
            input_L->size(),
            *prompt, 
            negative_prompt, 
            num_inference_steps,
            *scheduler, 
            seed, 
            guidance_scale, 
            strength, 
            0.2f, //0.2 is overlap of individual riffused segments that are cross-faded.
            unet_callback,
            seg_callback);

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
        std::cout << "in audio-to-audio routine: exception: " << error.what() << std::endl;
    }

    return 0;
}