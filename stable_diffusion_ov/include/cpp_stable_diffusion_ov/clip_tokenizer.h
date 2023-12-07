// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#pragma once

#include "cpp_stable_diffusion_ov/pretrained_tokenizer.h"
#include "cpp_stable_diffusion_ov/basic_tokenizer.h"
#include "cpp_stable_diffusion_ov/visibility.h"

namespace cpp_stable_diffusion_ov
{
    class CPP_SD_OV_API CLIPTokenizer final : public PreTrainedTokenizer
    {
    public:

        struct CLIPTokenizer_Params
        {
            PreTrainedTokenizer_Params baseInit;
            AddedToken unk_token = AddedToken("<|endoftext|>");
            AddedToken bos_token = AddedToken("<|startoftext|>");
            AddedToken eos_token = AddedToken("<|endoftext|>");
            AddedToken pad_token = AddedToken("<|endoftext|>");
            std::string errors = "replace";
        };

        CLIPTokenizer(CLIPTokenizer_Params params);

        virtual int64_t vocab_size() override {
            return _encoder.size();
        };


    protected:

        virtual std::unordered_map<std::string, int64_t> get_vocab() override
        {
            return _encoder;
        }

        std::vector<int64_t> virtual build_inputs_with_special_tokens(
            std::vector<int64_t>& token_ids_0, std::optional<std::vector<int64_t>> token_ids_1 = {}) override
        {
            std::vector<int64_t> ret;

            int64_t btoken_id = *bos_token_id();
            int64_t etoken_id = *eos_token_id();

            ret.push_back(btoken_id);
            ret.insert(ret.end(), token_ids_0.begin(), token_ids_0.end());
            ret.push_back(etoken_id);

            if (!token_ids_1)
            {
                return ret;
            }

            ret.push_back(etoken_id);
            ret.insert(ret.end(), token_ids_1->begin(), token_ids_1->end());
            ret.push_back(etoken_id);

            return ret;
        }

        virtual std::vector< int64_t > get_special_tokens_mask(
            std::vector< int64_t>& token_ids_0, std::optional<std::vector<int64_t>> token_ids_1 = {}, bool already_has_special_tokens = false) override
        {
            if (already_has_special_tokens)
            {
                return PreTrainedTokenizer::get_special_tokens_mask(token_ids_0, token_ids_1, already_has_special_tokens);
            }

            std::vector< int64_t > ret;

            ret.push_back(1);
            for (size_t i = 0; i < token_ids_0.size(); i++)
            {
                ret.push_back(0);
            }
            ret.push_back(1);

            if (!token_ids_1)
            {
                return ret;
            }

            ret.push_back(1);
            for (size_t i = 0; i < token_ids_1->size(); i++)
            {
                ret.push_back(0);
            }
            ret.push_back(1);

            return ret;
        }

        virtual std::vector< int64_t > create_token_type_ids_from_sequences(
            std::vector<int64_t>& token_ids_0, std::optional<std::vector<int64_t>> token_ids_1 = {}) override
        {
            int64_t btoken_id = *bos_token_id();
            int64_t etoken_id = *eos_token_id();

            size_t len = 0;
            len += token_ids_0.size() + 2;

            if (token_ids_1)
            {
                len += token_ids_1->size() + 2;
            }

            return std::vector< int64_t >(len, 0);
        }

        virtual std::list< std::string > _tokenize(std::string& text) override;

        std::string bpe(std::string token);

        virtual int64_t _convert_token_to_id(const std::string& token) override
        {
            auto it = _encoder.find(token);
            if (it == _encoder.end())
            {
                it = _encoder.find(unk_token()->content);
                if (it == _encoder.end())
                {
                    throw std::invalid_argument("unk token is not found as encoder key");
                }
            }

            return it->second;
        }



    private:

        BasicTokenizer _nlp;

        std::unordered_map< std::string, int64_t > _encoder;
        std::unordered_map< int64_t, std::string> _decoder;
        std::regex _pat;
        std::string _errors;
        std::map< std::vector<std::string>, int64_t> _bpe_ranks;
        std::unordered_map<std::string, std::string> _cache = { {"<|startoftext|>", "<|startoftext|>"},{"<|endoftext|>", "<|endoftext|>"} };
        std::unordered_map<char32_t, std::string> _byte_encoder;

    };
}