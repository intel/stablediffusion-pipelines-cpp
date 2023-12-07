// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#include "cpp_stable_diffusion_audio_ov/wav_util.h"
#include <fstream>
#include <iostream>
#include <climits>

namespace cpp_stable_diffusion_ov
{
    struct RiffWaveHeader {
        unsigned int riff_tag; // "RIFF" string
        int riff_length;       // Total length of the file minus 8
        unsigned int wave_tag; // "WAVE"
        unsigned int fmt_tag;  // "fmt " string (note space after 't')
        int fmt_length;        // Remaining length
        short data_format;     // Data format tag, 1 = PCM
        short num_of_channels; // Number of channels in file
        int sampling_freq;     // Sampling frequency
        int bytes_per_sec;     // Average bytes/sec
        short block_align;     // Block align
        short bits_per_sample;
        unsigned int data_tag; // "data" string
        int data_length;       // Raw data length
    };

    static const unsigned int fourcc(const char c[4]) {
        return (c[3] << 24) | (c[2] << 16) | (c[1] << 8) | (c[0]);
    }

    std::pair< std::shared_ptr<std::vector<float>>, std::shared_ptr<std::vector<float>> > ReadWav(std::string wav_filename)
    {
        std::ifstream inp_wave(wav_filename, std::ios::in | std::ios::binary);
        if (!inp_wave.is_open())
            throw std::runtime_error("fail to open " + wav_filename);

        RiffWaveHeader wave_header;
        inp_wave.read((char*)&wave_header, sizeof(RiffWaveHeader));

        std::string error_msg = "";
#define CHECK_IF(cond) if(cond){ error_msg = error_msg + #cond + ", "; }

        // make sure it is actually a RIFF file with WAVE
        CHECK_IF(wave_header.riff_tag != fourcc("RIFF"));
        CHECK_IF(wave_header.wave_tag != fourcc("WAVE"));
        CHECK_IF(wave_header.fmt_tag != fourcc("fmt "));
        // only PCM
        CHECK_IF(wave_header.data_format != 1);
        //only 2 channel
        CHECK_IF((wave_header.num_of_channels != 1) && (wave_header.num_of_channels != 2));
        // only 16 bit
        CHECK_IF(wave_header.bits_per_sample != 16);
        // make sure that data chunk follows file header
        CHECK_IF(wave_header.data_tag != fourcc("data"));
#undef CHECK_IF

        if (!error_msg.empty()) {
            throw std::runtime_error(error_msg + "for '" + wav_filename + "' file.");
        }

        std::cout << "header.bits_per_sample = " << wave_header.bits_per_sample << std::endl;
        std::cout << "header.data_length = " << wave_header.data_length << std::endl;
        std::cout << "header.sampling_freq = " << wave_header.sampling_freq << std::endl;
        std::cout << "header.bytes_per_sec = " << wave_header.bytes_per_sec << std::endl;

        size_t wave_size = wave_header.data_length / sizeof(int16_t);

        std::vector<int16_t> wave(wave_size);
        inp_wave.read((char*)&(wave.front()), wave_size * sizeof(int16_t));

        size_t nsamples = wave_size / wave_header.num_of_channels;

        if (wave_header.num_of_channels == 2)
        {
            std::shared_ptr < std::vector<float> > l_samples = std::make_shared< std::vector<float> >(nsamples);
            std::shared_ptr < std::vector<float> > r_samples = std::make_shared< std::vector<float> >(nsamples);

            float* pL = l_samples->data();
            float* pR = r_samples->data();

            for (size_t si = 0; si < nsamples; si++)
            {
                pL[si] = (float)wave[si * 2] / SHRT_MAX;
                pR[si] = (float)wave[si * 2 + 1] / SHRT_MAX;
            }

            return { l_samples, r_samples };
        }
        else
        {
            //1 channel
            std::shared_ptr < std::vector<float> > l_samples = std::make_shared< std::vector<float> >(nsamples);
            float* pL = l_samples->data();
            for (size_t si = 0; si < nsamples; si++)
            {
                pL[si] = (float)wave[si] / SHRT_MAX;
            }

            return { l_samples, {} };
        }
    }

    void WriteWav(std::string wav_filename, std::pair< std::shared_ptr<std::vector<float>>, std::shared_ptr<std::vector<float>> > wav_pair)
    {
        std::cout << "WriteWav->" << std::endl;
        std::shared_ptr<std::vector<float>> l_samples = wav_pair.first;
        std::shared_ptr<std::vector<float>> r_samples = wav_pair.second;

        if (!l_samples)
        {
            throw std::invalid_argument("WriteWav: wav_pair.first is required to be set");
        }

        if (r_samples && r_samples->size() != l_samples->size())
        {
            throw std::invalid_argument("WriteWav: If wav_pair.second is set, it must have same size (number of samples) as wav_pair.first");
        }

        size_t nsamples = l_samples->size();
        short nchannels = r_samples ? 2 : 1;

        size_t wave_size = nsamples * nchannels;
        std::vector<int16_t> wave(wave_size);

        if (nchannels == 1)
        {
            float* pMono = l_samples->data();
            for (size_t i = 0; i < nsamples; i++)
            {
                wave[i] = (int16_t)(pMono[i] * SHRT_MAX);
            }
        }
        else
        {
            float* pLeft = l_samples->data();
            float* pRight = r_samples->data();
            for (size_t i = 0; i < nsamples; i++)
            {
                wave[i * 2] = (int16_t)(pLeft[i] * SHRT_MAX);
                wave[i * 2 + 1] = (int16_t)(pRight[i] * SHRT_MAX);
            }
        }


        RiffWaveHeader out_header;
        out_header.riff_tag = fourcc("RIFF");
        out_header.wave_tag = fourcc("WAVE");
        out_header.fmt_tag = fourcc("fmt ");
        out_header.fmt_length = 16;
        out_header.data_format = 1;
        out_header.num_of_channels = nchannels;
        out_header.block_align = 2;
        out_header.bits_per_sample = 16;
        out_header.data_tag = fourcc("data");
        out_header.data_length = nsamples * sizeof(int16_t) * nchannels;
        out_header.sampling_freq = 44100;
        out_header.bytes_per_sec = out_header.sampling_freq * 2 * nchannels;
        out_header.riff_length = out_header.data_length + sizeof(RiffWaveHeader) - 8;

        std::ofstream out_wave(wav_filename, std::ios::out | std::ios::binary);
        if (!out_wave.is_open())
            throw std::runtime_error("fail to open " + wav_filename);

        out_wave.write((char*)&out_header, sizeof(RiffWaveHeader));
        out_wave.write((char*)&(wave.front()), wave.size() * sizeof(int16_t));
        std::cout << "<-WriteWav" << std::endl;
    }
}