// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#pragma once
#include <string>
#include <optional>
#include <vector>
#include <memory>
#include <thread>
#include "cpp_stable_diffusion_audio_ov/visibility.h"

namespace cpp_stable_diffusion_ov
{
    struct SpectrogramParams
    {
        // Whether the audio of stereo or mono
        bool stereo = false;

        // FFT parameters
        int sample_rate = 44100;
        int step_size_ms = 10;
        int window_duration_ms = 100;
        int padded_duration_ms = 400;

        // Mel scale parameters
        int num_frequencies = 512;
        int min_frequency = 0;
        int max_frequency = 10000;
        std::optional<std::string> mel_scale_norm = {};
        std::string mel_scale_type = "htk";
        int max_mel_iters = 200;

        // Griffin Lim parameters
        int num_griffin_lim_iters = 32;

        // Image parameterization
        float power_for_image = 0.25f;

        //not sure if we are going to do anything with ExifTags. Remove if we dont' end up using this.
        enum class ExifTags
        {
            // Custom EXIF tags for the spectrogram image.
            SAMPLE_RATE = 11000,
            STEREO = 11005,
            STEP_SIZE_MS = 11010,
            WINDOW_DURATION_MS = 11020,
            PADDED_DURATION_MS = 11030,

            NUM_FREQUENCIES = 11040,
            MIN_FREQUENCY = 11050,
            MAX_FREQUENCY = 11060,

            POWER_FOR_IMAGE = 11070,
            MAX_VALUE = 11080,
        };

        // The number of samples in each STFT window, with padding.
        int n_fft()
        {
            return (int)((float)padded_duration_ms / 1000.f * (float)sample_rate);
        }

        // The number of samples in each STFT window.
        int win_length()
        {
            return (int)((float)window_duration_ms / 1000.f * (float)sample_rate);
        }

        // The number of samples between each STFT window.
        int hop_length()
        {
            return (int)((float)step_size_ms / 1000.f * (float)sample_rate);
        }

    };

    class GriffinLim;
    class MelScale;
    class InverseMelScale;
    class Spectrogram;

    class CPP_SD_OV_AUDIO_API SpectrogramImageConverter
    {
    public:

        SpectrogramImageConverter(SpectrogramParams params = {});

        struct Image
        {
            std::shared_ptr<std::vector<uint8_t>> image_buf;
            size_t width;
            size_t height;
        };

        std::shared_ptr<std::vector<float>> audio_from_spectrogram_image(
            std::shared_ptr<std::vector<uint8_t>> image_buf_8u,
            int image_width = 512, int image_height = 512,
            size_t chan = 0, //0=mono, 1=left, 2=right
            bool apply_filters = true,
            float max_value = 30e6f);


        Image spectrogram_image_from_audio(
            std::shared_ptr<std::vector<float>> audio_wav,  //mono, or stereo 'L'
            std::shared_ptr<std::vector<float>> audio_wavR = {} //optional, only set if stereo to 'R'
        );

    private:

        SpectrogramParams _params;

        std::shared_ptr<std::vector<float>> _spec_from_image(std::shared_ptr<std::vector<uint8_t>> image_buf_8u,
            int image_width,
            int image_height,
            float max_value,
            size_t chan);

        std::shared_ptr<std::vector<float>> _audio_from_spectrogram(std::shared_ptr<std::vector<float>> spectrogram,
            bool apply_filters = true);

        std::shared_ptr< Spectrogram > _spectrogram;
        std::shared_ptr< GriffinLim > _griffin_lim;
        std::shared_ptr< InverseMelScale > _inverse_mel_scale;
        std::shared_ptr< MelScale > _mel_scale;
    };
}