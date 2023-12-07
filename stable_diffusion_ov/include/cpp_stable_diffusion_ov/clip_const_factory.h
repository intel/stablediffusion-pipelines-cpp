// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#pragma once
#include <unordered_map>
#include <map>
#include <string>
#include <vector>

namespace cpp_stable_diffusion_ov
{
	const std::unordered_map< std::string, int64_t > gen_vocab_encoder_map();
	const std::map< std::vector<std::string>, int64_t> gen_bpe_ranks();
	const std::unordered_map<char32_t, std::string> bytes_to_unicode();
}