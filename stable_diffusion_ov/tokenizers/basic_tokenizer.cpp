// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#include <codecvt>
#include <cwctype>
#include <cstdint>
#include <iostream>
#include <locale>
#include <sstream>
#include "cpp_stable_diffusion_ov/basic_tokenizer.h"


namespace cpp_stable_diffusion_ov
{
    static bool _is_whitespace(wchar_t c)
    {
        // \t, \n, and \r are technically control characters but we treat them
        // as whitespace since they are generally considered as such.
        if (c == L' ' || c == L'\t' || c == L'\n' || c == L'\r')
            return true;

        return (std::iswspace(c) != 0);
    }

    static bool _is_control(wchar_t c)
    {
        // These are technically control characters but we count them as whitespace
        // characters.
        if (c == L'\t' || c == L'\n' || c == L'\r')
        {
            return false;
        }

        return (std::iswcntrl(c) != 0);
    }

    static std::list< std::wstring > whitespace_tokenize(std::wstring text)
    {
        // strip leading whitespace
        text.erase(0, text.find_first_not_of(L' '));
        // strip trailing whitespace
        text.erase(text.find_last_not_of(L' ') + 1);

        if (text.length() == 0)
        {
            return {};
        }

        std::wstringstream wss(text);
        std::wstring temp;
        std::list< std::wstring > output;
        while (std::getline(wss, temp, L' '))
            output.push_back(temp);

        return output;
    }

    static bool _is_punctuation(wchar_t c)
    {
        int cp = (int)c;

        // We treat all non - letter / number ASCII as punctuation.
        // Characters such as "^", "$", and "`" are not in the Unicode
        // Punctuation class but we treat them as punctuation anyways, for
        // consistency.
        if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126))
            return true;

        return (std::iswpunct(c) != 0);
    }

    BasicTokenizer::BasicTokenizer(bool do_lower_case, std::list< std::string > never_split, bool tokenize_chinese_characters,
        std::optional<bool> strip_accents)
        : _do_lower_case(do_lower_case), _btokenize_chinese_chars(tokenize_chinese_characters), _strip_accents(strip_accents)
    {
        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
        for (auto& str : never_split)
        {
            std::wstring str_utf16 = converter.from_bytes(str);
            _never_split.insert(str_utf16);
        }
    }

    std::list< std::string > BasicTokenizer::tokenize(std::string text_utf8, std::optional< std::list< std::string > > never_splitu8)
    {
        //first, convert to text_utf8 and never_split to utf16
        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
        std::set< std::wstring > never_split = _never_split;
        if (never_splitu8)
        {
            for (auto& str : *never_splitu8)
            {
                std::wstring str_utf16 = converter.from_bytes(str);
                never_split.insert(str_utf16);
            }
        }

        std::wstring text = converter.from_bytes(text_utf8);
        text = _clean_text(text);

        // This was added on November 1st, 2018 for the multilingualand Chinese
        // models.This is also applied to the English models now, but it doesn't
        // matter since the English models were not trained on any Chinese data
        // and generally don't have any Chinese data in them (there are Chinese
        // characters in the vocabulary because Wikipedia does have some Chinese
        // words in the English Wikipedia.).
        if (_btokenize_chinese_chars)
        {
            text = _tokenize_chinese_chars(text);
        }

        auto orig_tokens = whitespace_tokenize(text);
        std::vector< std::wstring > split_tokens;
        for (auto& token : orig_tokens)
        {
            if (never_split.count(token) == 0)
            {
                if (_do_lower_case)
                {
                    std::transform(token.begin(), token.end(), token.begin(),
                        [](wchar_t c) { return std::tolower(c); });

                    if (!_strip_accents || !(*_strip_accents))
                    {
                        token = _run_strip_accents(token);
                    }
                }
                else if (_strip_accents && *_strip_accents)
                {
                    token = _run_strip_accents(token);
                }
            }

            auto punc_split = _run_split_on_punc(token, never_split);
            split_tokens.insert(split_tokens.end(), punc_split.begin(), punc_split.end());
        }

        std::wstring output;
        for (size_t i = 0; i < split_tokens.size(); i++)
        {
            auto token = split_tokens[i];
            output.insert(output.end(), token.begin(), token.end());
            if (i < (split_tokens.size() - 1))
            {
                output.push_back(L' ');
            }
        }

        auto output_tokens = whitespace_tokenize(output);

        std::list< std::string > output_tokens_u8;
        for (auto& token : output_tokens)
        {
            output_tokens_u8.push_back(converter.to_bytes(token));
        }

        return output_tokens_u8;
    }

    std::list< std::wstring > BasicTokenizer::_run_split_on_punc(std::wstring& text, std::set< std::wstring >& never_split)
    {
        if (never_split.count(text))
            return { text };

        size_t i = 0;
        bool start_new_word = true;
        std::list< std::wstring > output;
        while (i < text.length())
        {
            auto c = text[i];
            if (_is_punctuation(c))
            {
                output.push_back({ c });
                start_new_word = true;
            }
            else
            {
                if (start_new_word)
                {
                    output.push_back({});
                }
                start_new_word = false;
                output.back().push_back(c);
            }

            i++;
        }

        return output;
    }

    std::wstring BasicTokenizer::_clean_text(std::wstring text)
    {
        std::wstring output;
        for (auto& c : text)
        {
            if ((c == 0) || (c == 0xFFFD) || _is_control(c))
            {
                continue;
            }
            if (_is_whitespace(c))
            {
                output.push_back(L' ');
            }
            else
            {
                output.push_back(c);
            }
        }
        return output;
    }

    std::wstring BasicTokenizer::_tokenize_chinese_chars(std::wstring text)
    {
        // Adds whitespace around any CJK character.
        std::wstring output;
        for (auto& c : text)
        {
            if (_is_chinese_char(c))
            {
                output.push_back(L' ');
                output.push_back(c);
                output.push_back(L' ');
            }
            else
            {
                output.push_back(c);
            }
        }

        return output;
    }

    bool BasicTokenizer::_is_chinese_char(wchar_t cp)
    {
        //Checks whether CP is the codepoint of a CJK character.
        // This defines a "chinese character" as anything in the CJK Unicode block :
        //   https ://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        //
        // Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        // despite its name.The modern Korean Hangul alphabet is a different block,
        // as is Japanese Hiraganaand Katakana.Those alphabets are used to write
        // space - separated words, so they are not treated speciallyand handled
        // like the all of the other languages.

        //TODO: This function was ported from python, but some of the comparisons here
        // make no sense, given that cp is a wchar_t (16-bit) value. Reworking this is 
        // probably tied into the desired of reworking the overall UTF-8 stuff.
        if ((cp >= 0x4E00 && cp <= 0x9FFF)
            || (cp >= 0x3400 && cp <= 0x4DBF)
            || (cp >= 0x20000 && cp <= 0x2A6DF)
            || (cp >= 0x2A700 && cp <= 0x2B73F)
            || (cp >= 0x2B740 && cp <= 0x2B81F)
            || (cp >= 0x2B820 && cp <= 0x2CEAF)
            || (cp >= 0xF900 && cp <= 0xFAFF)
            || (cp >= 0x2F800 && cp <= 0x2FA1F))
        {
            return true;
        }

        return false;
    }
}