// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#include <codecvt>
#include <cwctype>
#include <algorithm>
#include <locale>
#include "cpp_stable_diffusion_ov/clip_const_factory.h"
#include "tokenizers/clip-vocab.h"
#include "tokenizers/clip-bpe-merges.h"

namespace cpp_stable_diffusion_ov
{

    const std::unordered_map<char32_t, std::string> bytes_to_unicode()
    {
        std::vector< char32_t > bs;
        for (char32_t c = U'!'; c < (U'~' + 1); c++)
        {
            bs.push_back(c);
        }

        for (char32_t c = U'¡'; c < (U'¬' + 1); c++)
        {
            bs.push_back(c);
        }

        for (char32_t c = U'®'; c < (U'ÿ' + 1); c++)
        {
            bs.push_back(c);
        }

        std::vector< char32_t > cs = bs;

        uint32_t n = 0;
        for (uint32_t b = 0; b < 256; b++)
        {
            if (std::find(bs.begin(), bs.end(), b) == bs.end())
            {
                bs.push_back(b);
                cs.push_back(256 + n);
                n++;
            }
        }

        std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
        std::unordered_map<char32_t, std::string> ret;
        for (size_t i = 0; i < bs.size(); i++)
        {
            std::u32string tmp;
            tmp.push_back(cs[i]);
            std::string tmp_s = conv.to_bytes(tmp);
            ret.insert({ bs[i], tmp_s });
        }

        return ret;
    }

    const std::unordered_map< std::string, int64_t > gen_vocab_encoder_map()
    {
        std::unordered_map< std::string, int64_t > encoder;
        for (size_t i = 0; i < TOTAL_CLIP_VOCAB_PAIRS; i++)
        {
            encoder.insert({ CLIP_VOCAB_KEY[i], CLIP_VOCAB_VAL[i] });
        }

        return encoder;
    }

    const std::map< std::vector<std::string>, int64_t> gen_bpe_ranks()
    {
        std::map< std::vector<std::string>, int64_t> bpe_ranks;
        int64_t i = 0;
        for (i = 0; i < (int64_t)TOTAL_BPE_PAIRS; i++)
        {
            bpe_ranks.insert({ {CLIP_BPE_PAIR0[i], CLIP_BPE_PAIR1[i]}, i });
        }

        return bpe_ranks;
    }
}