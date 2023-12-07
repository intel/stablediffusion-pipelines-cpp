// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#include <mutex>
#include <thread>
#include <future>
#include <iostream>
#include <condition_variable>
#include "cpp_stable_diffusion_audio_ov/riffusion_audio_to_audio_pipeline.h"
#include "cpp_stable_diffusion_ov/stable_diffusion_pipeline.h"
#include "cpp_stable_diffusion_audio_ov/spectrogram_image_converter.h"

namespace cpp_stable_diffusion_ov
{

    static const size_t SAMPLE_RATE = 44100;

    //'magic' number for how many samples equals a 512x512 spectrogram.
    static const size_t NSAMPLES_512x512 = 225351;


    RiffusionAudioToAudioPipeline::RiffusionAudioToAudioPipeline(std::string model_folder,
        std::optional<std::string> cache_folder,
        std::string text_encoder_device,
        std::string unet_positive_device,
        std::string unet_negative_device,
        std::string vae_decoder_device,
        std::string vae_encoder_device)
    {
        _stable_diffusion_pipeline = std::make_shared< StableDiffusionPipeline >(model_folder, cache_folder, text_encoder_device,
            unet_positive_device, unet_negative_device, vae_decoder_device, vae_encoder_device);

        _spec_img_converterL = std::make_shared < SpectrogramImageConverter >();
        _spec_img_converterR = std::make_shared < SpectrogramImageConverter >();

    }

    static bool run_img_chan_to_wav_routine(std::shared_ptr<std::vector<uint8_t>> img, size_t chan, float* pOutput, size_t segmenti, size_t overlap_samples, size_t valid_samples, size_t valid_sample_offset, std::shared_ptr< SpectrogramImageConverter > spec_img_converter)
    {
        std::cout << "run_img_chan_to_wav_routine(" << chan << ")->" << std::endl;

        {
            auto riffused_samples = spec_img_converter->audio_from_spectrogram_image(img, 512, 512, chan);
            float* pRiffused = riffused_samples->data();
            if (segmenti == 0)
            {
                std::memcpy(pOutput, pRiffused, valid_samples * sizeof(float));
            }
            else
            {
                std::cout << "cross-fading " << std::min(overlap_samples, valid_samples) << " channel " << chan << " samples..." << std::endl;
                //cross-fade
                float fade_step = 1.f / (float)(overlap_samples);

                size_t outputi;
                for (outputi = 0; (outputi < std::min(overlap_samples, valid_samples)); outputi++)
                {
                    float fade = pOutput[outputi] * (1 - outputi * fade_step) +
                        pRiffused[outputi + valid_sample_offset] * (outputi * fade_step);

                    pOutput[outputi] = fade;
                }

                size_t samples_left = valid_samples - outputi;
                if (samples_left)
                {
                    std::memcpy(pOutput + outputi, pRiffused + outputi + valid_sample_offset, samples_left * sizeof(float));
                }
            }
        }

        std::cout << "<-run_img_chan_to_wav_routine(" << chan << ")" << std::endl;
        return true;
    }


    static bool run_img_to_wav_routine(std::shared_ptr<std::vector<uint8_t>> img, float* pOutputL, float* pOutputR, size_t segmenti, size_t overlap_samples, size_t valid_samples, size_t valid_sample_offset, std::shared_ptr< SpectrogramImageConverter > _spec_img_converter)
    {
        std::cout << "run_img_to_wav_routine->" << std::endl;

        {
            auto riffused_samples = _spec_img_converter->audio_from_spectrogram_image(img, 512, 512, 1);
            float* pRiffused = riffused_samples->data();
            if (segmenti == 0)
            {
                std::memcpy(pOutputL, pRiffused, valid_samples * sizeof(float));
            }
            else
            {
                std::cout << "cross-fading " << std::min(overlap_samples, valid_samples) << " L/Mono samples..." << std::endl;
                //cross-fade
                float fade_step = 1.f / (float)(overlap_samples);

                size_t outputi;
                for (outputi = 0; (outputi < std::min(overlap_samples, valid_samples)); outputi++)
                {
                    float fade = pOutputL[outputi] * (1 - outputi * fade_step) +
                        pRiffused[outputi + valid_sample_offset] * (outputi * fade_step);

                    pOutputL[outputi] = fade;
                }

                size_t samples_left = valid_samples - outputi;
                if (samples_left)
                {
                    std::memcpy(pOutputL + outputi, pRiffused + outputi + valid_sample_offset, samples_left * sizeof(float));
                }
            }
        }

        if (pOutputR)
        {
            auto riffused_samples = _spec_img_converter->audio_from_spectrogram_image(img, 512, 512, 2);
            float* pRiffused = riffused_samples->data();
            if (segmenti == 0)
            {
                std::memcpy(pOutputR, pRiffused, valid_samples * sizeof(float));
            }
            else
            {
                std::cout << "cross-fading " << std::min(overlap_samples, valid_samples) << " R samples..." << std::endl;
                //cross-fade
                float fade_step = 1.f / (float)(overlap_samples);

                size_t outputi;
                for (outputi = 0; (outputi < std::min(overlap_samples, valid_samples)); outputi++)
                {
                    float fade = pOutputR[outputi] * (1 - outputi * fade_step) +
                        pRiffused[outputi + valid_sample_offset] * (outputi * fade_step);

                    pOutputR[outputi] = fade;
                }

                size_t samples_left = valid_samples - outputi;
                if (samples_left)
                {
                    std::memcpy(pOutputR + outputi, pRiffused + outputi + valid_sample_offset, samples_left * sizeof(float));
                }
            }
        }

        std::cout << "<-run_img_to_wav_routine" << std::endl;
        return true;
    }

    std::pair< std::shared_ptr<std::vector<float>>, std::shared_ptr<std::vector<float>> > RiffusionAudioToAudioPipeline::operator()(
        float* pInput_44100_wav,
        float* pInput_44100_wav_R,
        size_t nsamples_to_riffuse,
        size_t ntotal_samples_allowed_to_read,
        const std::string prompt,
        std::optional< std::string > negative_prompt,
        int num_inference_steps,
        const std::string& scheduler_str,
        std::optional< unsigned int > seed,
        float guidance_scale,
        float denoising_strength,
        float crossfade_overlap_seconds,
        std::optional<std::pair< CallbackFuncUnetIteration, void*>> unet_iteration_callback,
        std::optional<std::pair< CallbackFuncAudioSegmentComplete, void*>> segment_callback)
    {
        if (!pInput_44100_wav)
        {
            throw std::invalid_argument("pInput_44100_wav is null. It is required to be set.");
        }

        if (nsamples_to_riffuse == 0)
        {
            throw std::invalid_argument("nsamples_to_riffuse is 0");
        }
        

        if (ntotal_samples_allowed_to_read < nsamples_to_riffuse)
        {
            throw std::invalid_argument("ntotal_samples_allowed_to_read must be >= nsamples_to_riffuse");
        }

        if ((crossfade_overlap_seconds < 0) || (crossfade_overlap_seconds > 1.0))
        {
            throw std::invalid_argument("crossfade_overlap_seconds must be in the range [0, 1]. It is set to " 
                + std::to_string(crossfade_overlap_seconds));
        }

        size_t overlap_samples = (size_t)((double)SAMPLE_RATE * (double)crossfade_overlap_seconds);

        std::vector< std::pair<size_t, size_t> > segments;

        {
            size_t current_sample = 0;
            while (current_sample < nsamples_to_riffuse)
            {
                if (current_sample != 0)
                {
                    current_sample -= overlap_samples;
                }

                std::pair<size_t, size_t> segment = { current_sample,
                    std::min(nsamples_to_riffuse - current_sample, NSAMPLES_512x512) };

                segments.push_back(segment);

                current_sample += NSAMPLES_512x512;
            }
        }


        std::shared_ptr<std::vector<float>> outputL = std::make_shared< std::vector<float> >(nsamples_to_riffuse);
        std::shared_ptr<std::vector<float>> outputR;

        if (pInput_44100_wav_R)
        {
            outputR = std::make_shared< std::vector<float> >(nsamples_to_riffuse);
        }

        if (segment_callback)
        {
            segment_callback->first(0, segments.size(), {},
                {}, 512, 512, segment_callback->second);
        }

        std::future<bool> last_iteration_img_to_wavL;
        std::future<bool> last_iteration_img_to_wavR;


        for (size_t segmenti = 0; segmenti < segments.size(); segmenti++)
        {
            std::shared_ptr<std::vector<float>> this_segment_L = std::make_shared< std::vector<float> >(NSAMPLES_512x512, 0.f);
            std::shared_ptr<std::vector<float>> this_segment_R;
            if (pInput_44100_wav_R)
            {
                this_segment_R = std::make_shared< std::vector<float> >(NSAMPLES_512x512, 0.f);
            }

            size_t valid_samples = NSAMPLES_512x512;
            size_t valid_sample_offset = 0;
            if (segmenti != (segments.size() - 1))
            {
                //todo: make this zero-copy
                std::memcpy(this_segment_L->data(), pInput_44100_wav + segments[segmenti].first, NSAMPLES_512x512 * sizeof(float));
                if (pInput_44100_wav_R)
                {
                    std::memcpy(this_segment_R->data(), pInput_44100_wav_R + segments[segmenti].first, NSAMPLES_512x512 * sizeof(float));
                }
            }
            else
            {
                {
                    size_t thisi = 0;
                    float* pThisSeg = this_segment_L->data();
                    float* pInput = pInput_44100_wav + segments[segmenti].first;
                    valid_samples = nsamples_to_riffuse - segments[segmenti].first;
                    std::memcpy(pThisSeg, pInput, valid_samples * sizeof(float));

                    size_t unfilled_samples = NSAMPLES_512x512 - valid_samples;
                    if (unfilled_samples)
                    {
                        // So we have a bunch of 'unfilled samples' that we've padded to 0. This could make the output sound
                        // kind of bizarre. We have a few potential options (in order of preference).
                        // 1. If 'ntotal_samples_allowed_to_read' > 'nsamples_to_riffuse', then we can fill (at least some of)
                        // the segment with these. These extra samples are called 'readonly' samples.
                        // 2. If we don't have enough 'readonly' samples, and segmenti is not 0, it means we have a bunch of samples
                        // to the left. We could fill a segment with 'unfilled samples' residing at the end.
                        //check if there are readonly samples. Which are samples *past* the end of samples that we are outputting.
                        size_t readonly_samples = std::min(unfilled_samples, ntotal_samples_allowed_to_read - nsamples_to_riffuse);

                        if ((readonly_samples < unfilled_samples) && (segmenti != 0))
                        {
                            //okay, we want 'valid samples' to actually show up at the *end*, so copy 'unfilled' amount of samples
                            // from previous segment.
                            std::memcpy(pThisSeg, pInput - unfilled_samples, unfilled_samples * sizeof(float));

                            //now copy our valid samples
                            std::memcpy(pThisSeg + unfilled_samples, pInput, valid_samples * sizeof(float));

                            std::cout << "unfilled_samples = " << unfilled_samples << std::endl;
                            std::cout << "valid_samples = " << valid_samples << std::endl;

                            valid_sample_offset = unfilled_samples;
                        }
                        else if (readonly_samples)
                        {
                            pInput += valid_samples;
                            pThisSeg += valid_samples;
                            std::memcpy(pThisSeg, pInput, readonly_samples * sizeof(float));
                        }
                        //otherwise, we do nothing and leave padded samples there...
                    }
                }

                if (pInput_44100_wav_R)
                {
                    size_t thisi = 0;
                    float* pThisSeg = this_segment_R->data();
                    float* pInput = pInput_44100_wav_R + segments[segmenti].first;
                    valid_samples = nsamples_to_riffuse - segments[segmenti].first;
                    std::memcpy(pThisSeg, pInput, valid_samples * sizeof(float));

                    size_t unfilled_samples = NSAMPLES_512x512 - valid_samples;
                    if (unfilled_samples)
                    {
                        // So we have a bunch of 'unfilled samples' that we've padded to 0. This could make the output sound
                        // kind of bizarre. We have a few potential options (in order of preference).
                        // 1. If 'ntotal_samples_allowed_to_read' > 'nsamples_to_riffuse', then we can fill (at least some of)
                        // the segment with these. These extra samples are called 'readonly' samples.
                        // 2. If we don't have enough 'readonly' samples, and segmenti is not 0, it means we have a bunch of samples
                        // to the left. We could fill a segment with 'unfilled samples' residing at the end.
                        //check if there are readonly samples. Which are samples *past* the end of samples that we are outputting.
                        size_t readonly_samples = std::min(unfilled_samples, ntotal_samples_allowed_to_read - nsamples_to_riffuse);

                        if ((readonly_samples < unfilled_samples) && (segmenti != 0))
                        {
                            //okay, we want 'valid samples' to actually show up at the *end*, so copy 'unfilled' amount of samples
                            // from previous segment.
                            std::memcpy(pThisSeg, pInput - unfilled_samples, unfilled_samples * sizeof(float));

                            //now copy our valid samples
                            std::memcpy(pThisSeg + unfilled_samples, pInput, valid_samples * sizeof(float));

                            std::cout << "unfilled_samples = " << unfilled_samples << std::endl;
                            std::cout << "valid_samples = " << valid_samples << std::endl;

                            valid_sample_offset = unfilled_samples;
                        }
                        else if (readonly_samples)
                        {
                            pInput += valid_samples;
                            pThisSeg += valid_samples;
                            std::memcpy(pThisSeg, pInput, readonly_samples * sizeof(float));
                        }
                        //otherwise, we do nothing and leave padded samples there...
                    }
                }
            }

            auto spec_img = _spec_img_converterL->spectrogram_image_from_audio(this_segment_L, this_segment_R);
            StableDiffusionPipeline::InputImageParams img_params;
            img_params.image_buffer = spec_img.image_buf;
            img_params.isBGR = false;
            img_params.isNHWC = true;
            img_params.strength = denoising_strength;

            auto riffused_img = (*_stable_diffusion_pipeline)(
                prompt,
                negative_prompt,
                num_inference_steps,
                scheduler_str,
                seed,
                guidance_scale,
                false,
                img_params,
                unet_iteration_callback);

            //in case it was cancelled.
            if (!riffused_img)
            {
                if (last_iteration_img_to_wavL.valid())
                    last_iteration_img_to_wavL.wait();

                if (last_iteration_img_to_wavR.valid())
                    last_iteration_img_to_wavR.wait();

                return {};
            }

            //asynchronous img-to-wav
            if (segment_callback)
            {
                segment_callback->first(segmenti + 1, segments.size(), {},
                    riffused_img, 512, 512, segment_callback->second);
            }

            if (last_iteration_img_to_wavL.valid())
                last_iteration_img_to_wavL.wait();

            if (last_iteration_img_to_wavR.valid())
                last_iteration_img_to_wavR.wait();

            //last_iteration_img_to_wavL = std::async(run_img_to_wav_routine, riffused_img, pOutputL, pOutputR, segmenti, overlap_samples, valid_samples, valid_sample_offset, _spec_img_converterL);

            float* pOutputL = outputL->data() + segments[segmenti].first;
            last_iteration_img_to_wavL = std::async(run_img_chan_to_wav_routine, riffused_img, 1, pOutputL, segmenti, overlap_samples, valid_samples, valid_sample_offset, _spec_img_converterL);

            if (outputR)
            {
                auto pOutputR = outputR->data() + segments[segmenti].first;
                last_iteration_img_to_wavR = std::async(run_img_chan_to_wav_routine, riffused_img, 2, pOutputR, segmenti, overlap_samples, valid_samples, valid_sample_offset, _spec_img_converterR);
            }
        }

        if (last_iteration_img_to_wavL.valid())
            last_iteration_img_to_wavL.wait();

        if (last_iteration_img_to_wavR.valid())
            last_iteration_img_to_wavR.wait();


        return { outputL, outputR };
    }
}
