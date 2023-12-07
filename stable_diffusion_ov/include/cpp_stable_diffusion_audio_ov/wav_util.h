// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#pragma once
#include <vector>
#include <memory>
#include <string>
#include <optional>
#include <utility>
#include "cpp_stable_diffusion_audio_ov/visibility.h"

namespace cpp_stable_diffusion_ov
{
	//return a pair of sample vectors
	//.first = mono if wav file is mono, left is wav file is stereo
	//.second = {} if wav file is mono, right if wav file is stereo.
	std::pair< std::shared_ptr<std::vector<float>>, std::shared_ptr<std::vector<float>> > CPP_SD_OV_AUDIO_API ReadWav(std::string wav_filename);

	void CPP_SD_OV_AUDIO_API WriteWav(std::string wav_filename, std::pair< std::shared_ptr<std::vector<float>>, std::shared_ptr<std::vector<float>> > wav_pair);
}