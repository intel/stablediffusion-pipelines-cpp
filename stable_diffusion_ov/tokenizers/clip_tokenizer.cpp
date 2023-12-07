// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#include <codecvt>
#include <cwctype>
#include <vector>
#include <sstream>
#include "cpp_stable_diffusion_ov/clip_tokenizer.h"
#include "cpp_stable_diffusion_ov/clip_const_factory.h"

namespace cpp_stable_diffusion_ov
{
    CLIPTokenizer::CLIPTokenizer(CLIPTokenizer_Params params)
        : PreTrainedTokenizer(params.baseInit), _errors(params.errors)
    {
        _encoder = gen_vocab_encoder_map();
        for (auto& v : _encoder)
        {
            _decoder.insert({ v.second, v.first });
        }

        _bpe_ranks = gen_bpe_ranks();

        //self.pat = re.compile(
        //    r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
        //    re.IGNORECASE,
        //    )
        _pat = std::regex("<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+",
            std::regex_constants::ECMAScript | std::regex_constants::icase);

        _model_input_names = { "input_ids", "attention_mask" };
        _model_max_length = 77;

        _byte_encoder = bytes_to_unicode();

        unk_token(params.unk_token);
        bos_token(params.bos_token);
        eos_token(params.eos_token);
        pad_token(params.pad_token);
    }

    std::list< std::string > CLIPTokenizer::_tokenize(std::string& text)
    {
        //std::cout << "CLIPTokenizer::_tokenize: text = " << text << std::endl;
        auto text_tokenized = _nlp.tokenize(text);

#if 0
        std::cout << "CLIPTokenizer::_tokenize: text_tokenized = " << std::endl;
        for (auto& token : text_tokenized)
        {
            std::cout << "     " << token << std::endl;
        }
#endif

        text.clear();
        auto it = text_tokenized.begin();
        while (it != text_tokenized.end())
        {
            text += *it;
            it++;
            if (it != text_tokenized.end())
                text += " ";
        }

        std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;

        std::list< std::string > bpe_tokens;
        //std::cout << "creating refex iterator from text = \"" << text << "\"" << std::endl;
        //for (std::sregex_iterator i = std::sregex_iterator(text.begin(), text.end(), _pat);
        //    i != std::sregex_iterator();
        //    ++i)
        for (auto& token : text_tokenized)
        {
            //std::smatch m = *i;
            auto token32 = conv.from_bytes(token);

            std::string token_encoded;
            for (auto& c32 : token32)
            {
                //todo: add check for existence here
                token_encoded += _byte_encoder[c32];
            }
            // bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))

            std::list< std::string > tmp_list;
            {
                auto bpe_ret = bpe(token_encoded);
                std::stringstream ss(bpe_ret);
                std::string tmp;
                while (std::getline(ss, tmp, ' '))
                    tmp_list.push_back(tmp);
            }
            bpe_tokens.insert(bpe_tokens.end(), tmp_list.begin(), tmp_list.end());
        }


        return bpe_tokens;
    }

    static std::set<std::pair<std::string, std::string>> get_pairs(std::vector<std::string> word)
    {
        if (word.empty())
            return {};

        std::set<std::pair<std::string, std::string>> pairs;
        std::string prev_char = word[0];
        for (size_t i = 1; i < word.size(); i++)
        {
            auto c = word[i];
            //std::cout << "get_pairs adding " << prev_char << " " << c << std::endl;
            pairs.insert({ prev_char, c });
            prev_char = c;
        }

#if 0
        std::cout << "get_pairs: returning pairs = " << std::endl;
        for (auto& p : pairs)
        {
            std::cout << "  (" << p.first << "," << p.second << ")" << std::endl;
        }
#endif

        return pairs;
    }

    std::string CLIPTokenizer::bpe(std::string token)
    {
        //std::cout << "bpe: token = " << token << std::endl;
        if (_cache.count(token))
        {
            return _cache[token];
        }

        //bpe: token = chords
        //bpe : word = ('c', 'h', 'o', 'r', 'd', 's</w>')
        //bpe : pairs = { ('o', 'r'), ('h', 'o'), ('d', 's</w>'), ('r', 'd'), ('c', 'h') }
        std::vector< std::string > word;

        //std::cout << "word = ";
        for (size_t i = 0; i < token.length(); i++)
        {
            if (i != (token.length() - 1))
            {
                word.push_back(token.substr(i, 1));
            }
            else
            {
                word.push_back(token.substr(i, 1) + "</w>");
            }
            //std::cout << word.back() << " ";
        }
        //std::cout << std::endl;

        auto pairs = get_pairs(word);

        if (pairs.empty())
            return token + "</w>";

        while (true)
        {
            //std::cout << "start loop it" << std::endl;
            std::optional < std::pair<std::string, std::string >> min_pair = {};
            int64_t min_pair_val = std::numeric_limits< int64_t >::max();
            for (auto& p : pairs)
            {
                std::vector<std::string> k = { p.first, p.second };
                auto it = _bpe_ranks.find(k);
                if (it != _bpe_ranks.end())
                {
                    //std::cout << "_bpe_ranks[" << p.first << ", " << p.second << "] = " << it->second << std::endl;
                    if (it->second < min_pair_val)
                    {
                        min_pair_val = it->second;
                        min_pair = p;
                    }

                }
                else
                {
                    //std::cout << "_bpe_ranks[" << p.first << ", " << p.second << "] not present" << std::endl;
                }
            }

            if (!min_pair)
                break;

            auto first = min_pair->first;
            auto second = min_pair->second;

            //std::cout << "bigram = " << first << " " << second << std::endl;

            std::vector<std::string> new_word;
            size_t i = 0;
            while (i < word.size())
            {
                std::optional< size_t > j;
                // std::cout << "searching for " << first << " starting at index " << i << std::endl;
                for (auto searchi = i; searchi < word.size(); searchi++)
                {
                    if (word[searchi] == first)
                    {
                        j = searchi;
                        break;
                    }
                }

                if (!j)
                {
                    //std::cout << "not found. Extending " << i << " till end" << std::endl;
                    new_word.insert(new_word.end(), word.begin() + i, word.end());
                    break;
                }
                else
                {
                    //std::cout << "found. Extending " << i << " till " << *j << std::endl;
                    new_word.insert(new_word.end(), word.begin() + i, word.begin() + *j);
                    i = *j;
                }

                if ((word[i] == first) && (i < (word.size() - 1)) && (word[i + 1] == second))
                {
                    //std::cout << "appending first + second" << std::endl;
                    new_word.push_back(first + second);
                    i += 2;
                }
                else
                {
                    //std::cout << "appending word[i]" << std::endl;
                    new_word.push_back(word[i]);
                    i += 1;
                }

#if 0
                std::cout << "new_word = " << std::endl;
                for (auto& n : new_word)
                {
                    std::cout << "(" << n << "), ";
                }
                std::cout << std::endl;
#endif
            }


#if 0
            std::cout << "word = new_word = " << std::endl;
            for (auto& n : new_word)
            {
                std::cout << "(" << n << "), ";
            }
            std::cout << std::endl;
#endif

            word = new_word;
            if (word.size() == 1)
            {
                break;
            }
            else
            {
                pairs = get_pairs(word);
            }
        }

        std::string ret;
        for (size_t i = 0; i < word.size(); i++)
        {
            ret += word[i];

            if (i != (word.size() - 1))
                ret += " ";
        }

        _cache[token] = ret;

        //std::cout << "bpe returning " << ret << std::endl;
        return ret;
    }
}