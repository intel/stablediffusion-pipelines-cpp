#include <iostream>
#include "cpp_stable_diffusion_audio_ov/spectrogram_image_converter.h"
#include <vector>
#include <memory>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "simple_cmdline_parser.h"
#include "cpp_stable_diffusion_audio_ov/wav_util.h"
#include <chrono>

void print_usage()
{
    std::cout << "img_to_wav usage: " << std::endl;
    std::cout << "--input_image=input_512x512.png" << std::endl;
    std::cout << "--output_wav=output.wav" << std::endl;
    std::cout << "--mono (if set, will produce a mono .wav instead of stereo)" << std::endl;
}

int main(int argc, char *argv[])
{
    try
    {
        SimpleCmdLineParser cmdline_parser(argc, argv);
        if (cmdline_parser.is_help_needed())
        {
            print_usage();
            return -1;
        }

        auto input_image = cmdline_parser.get_value_for_key("input_image");
        if (!input_image)
        {
            std::cout << "Error! --input_image argument is required" << std::endl;
            print_usage();
            return 1;
        }

        auto output_wav = cmdline_parser.get_value_for_key("output_wav");
        if (!output_wav)
        {
            std::cout << "Error! --output_wav argument is required" << std::endl;
            print_usage();
            return 1;
        }

        cv::Mat init_image = cv::imread(*input_image, cv::IMREAD_COLOR);

        if ((init_image.cols != 512) || (init_image.rows != 512))
        {
            throw std::invalid_argument("input image must be 512x512");
        }

        std::shared_ptr<std::vector<uint8_t>> image_buf_8u = std::make_shared<std::vector<uint8_t>>(512 * 512 * 3);

        uint8_t* pBGR = init_image.ptr();
        uint8_t* pRGB = image_buf_8u->data();
        for (size_t p = 0; p < 512 * 512; p++)
        {
            pRGB[p * 3 + 0] = pBGR[p * 3 + 2];
            pRGB[p * 3 + 1] = pBGR[p * 3 + 1];
            pRGB[p * 3 + 2] = pBGR[p * 3 + 0];
        }
    
        //RiffWaveHeader in_header;
        //read_wav("output.wav", in_header);
        using namespace std::chrono;
        using Clock = std::chrono::high_resolution_clock;
        uint64_t  t0 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
        cpp_stable_diffusion_ov::SpectrogramImageConverter converter;
        uint64_t  t1 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
        std::cout << "Created SpectrogramImageConverter in " << t1 - t0 << " ms." << std::endl;

        std::pair< std::shared_ptr<std::vector<float>>, std::shared_ptr<std::vector<float>> > out;

        if (cmdline_parser.was_key_given("mono"))
        {
            out.first = converter.audio_from_spectrogram_image(image_buf_8u, 512, 512, 0);
        }
        else
        {
            t0 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
            out.first = converter.audio_from_spectrogram_image(image_buf_8u, 512, 512, 1);
            t1 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
            std::cout << "Converted channel 1 to L wav in " << t1 - t0 << " ms." << std::endl;
            t0 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
            out.second = converter.audio_from_spectrogram_image(image_buf_8u, 512, 512, 2);
            t1 = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
            std::cout << "Converted channel 2 to R wav in " << t1 - t0 << " ms." << std::endl;
        }

        cpp_stable_diffusion_ov::WriteWav(*output_wav, out);

    }
    catch (const std::exception& error) {
        std::cout << "in audio_from_spectrogram_image routine: exception: " << error.what() << std::endl;
    }


	return 0;
}