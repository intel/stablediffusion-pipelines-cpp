// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#pragma once

#include <string>
#include "cpp_stable_diffusion_ov/tokenization_utils.h"
#include "cpp_stable_diffusion_ov/visibility.h"

namespace cpp_stable_diffusion_ov
{
    // transformers/models/clip/tokenization_clip.py
    class CPP_SD_OV_API BasicTokenizer
    {
    public:

        BasicTokenizer(bool do_lower_case = true,
            std::list< std::string > never_split = {},
            bool tokenize_chinese_characters = true,
            std::optional<bool> strip_accents = {});

        std::list< std::string > tokenize(std::string text_utf8, std::optional< std::list< std::string > > never_splitu8 = {});

    private:

        std::wstring _run_strip_accents(std::wstring& text)
        {
            //todo implement this. Got stuck at this call:
            // text = unicodedata.normalize("NFD", text)
            return text;
        }
        std::list< std::wstring > _run_split_on_punc(std::wstring& text, std::set< std::wstring >& never_split);
        std::wstring _clean_text(std::wstring text);
        std::wstring _tokenize_chinese_chars(std::wstring text);
        bool _is_chinese_char(wchar_t cp);

        std::set< std::wstring > _never_split;
        bool _do_lower_case = false;
        bool _btokenize_chinese_chars = false;
        std::optional<bool> _strip_accents = {};
    };
}