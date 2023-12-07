// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#pragma once

#include "cpp_stable_diffusion_ov/pretrained_tokenizer_base.h"
#include "cpp_stable_diffusion_ov/visibility.h"

namespace cpp_stable_diffusion_ov
{
    class CPP_SD_OV_API PreTrainedTokenizer : public PreTrainedTokenizerBase
    {
    public:

        struct PreTrainedTokenizer_Params
        {
            PreTrainedTokenizerBase_Params baseInit;
        };

        PreTrainedTokenizer(PreTrainedTokenizer_Params init = {})
            : PreTrainedTokenizerBase(init.baseInit)
        {

        }

        virtual bool is_fast() { return false; };

        virtual int64_t vocab_size() = 0;

        std::unordered_map<std::string, uint64_t>& get_added_vocab() { return _added_tokens_encoder; };

        int64_t len() { return vocab_size() + _added_tokens_encoder.size(); };


    protected:

        // Add a list of new tokens to the tokenizer class. If the new tokens are not in the vocabulary, they are added to
        // it with indices starting from length of the current vocabulary.
        virtual int64_t _add_tokens(std::list<AddedToken>& new_tokens_orig, bool special_tokens = false) override;

        void _create_trie(std::list<std::string>& unique_no_split_tokens);

        // Returns the number of added tokens when encoding a sequence with special tokens.
        virtual int64_t num_special_tokens_to_add(bool pair) override;


        // Converts a string in a sequence of tokens, using the tokenizer.
        virtual std::list< std::string > tokenize(std::string text, std::optional<std::string> pair = {}, bool add_special_tokens = false) override;

        virtual std::list< std::string > _tokenize(std::string& text) = 0;

        virtual std::string prepare_for_tokenization(std::string& text, bool is_split_into_words = false)
        {
            return text;
        }

        virtual int64_t convert_tokens_to_ids(std::string token)
        {
            return _convert_token_to_id_with_added_voc(token);
        }


        std::list<int64_t> convert_tokens_to_ids(std::list<std::string>& tokens)
        {
            //std::cout << "convert_tokens_to_ids->" << std::endl;
            std::list<int64_t> ids;
            for (auto& token : tokens)
            {
                ids.push_back(convert_tokens_to_ids(token));
            }
            //std::cout << "<-convert_tokens_to_ids" << std::endl;
            return ids;
        }


        int64_t _convert_token_to_id_with_added_voc(const std::string& token)
        {
            std::unordered_map<std::string, uint64_t>::iterator it = _added_tokens_encoder.find(token);
            if (it != _added_tokens_encoder.end())
            {
                return it->second;
            }

            return _convert_token_to_id(token);
        }

        virtual int64_t _convert_token_to_id(const std::string& token) = 0;

        virtual BatchEncoding _encode_plus(std::optional<std::string> text = {},
            std::optional<std::string> text_pair = {},
            bool add_special_tokens = true,
            PaddingStrategy padding_strategy = PaddingStrategy::DO_NOT_PAD,
            TruncationStrategy truncation_strategy = TruncationStrategy::DO_NOT_TRUNCATE,
            std::optional<int64_t> max_length = {},
            int64_t stride = 0,
            bool is_split_into_words = false,
            std::optional<int64_t> pad_to_multiple_of = {},
            std::optional<bool> return_token_type_ids = {},
            std::optional<bool> return_attention_mask = {},
            bool return_overflowing_tokens = false,
            bool return_special_tokens_mask = false,
            bool return_offsets_mapping = false,
            bool return_length = false,
            bool verbose = true) override;

        virtual BatchEncoding _batch_encode_plus(std::optional<std::list<std::string>> text = {},
            std::optional<std::list<std::string>> text_pair = {},
            bool add_special_tokens = true,
            PaddingStrategy padding_strategy = PaddingStrategy::DO_NOT_PAD,
            TruncationStrategy truncation_strategy = TruncationStrategy::DO_NOT_TRUNCATE,
            std::optional<int64_t> max_length = {},
            int64_t stride = 0,
            bool is_split_into_words = false,
            std::optional<int64_t> pad_to_multiple_of = {},
            std::optional<bool> return_token_type_ids = {},
            std::optional<bool> return_attention_mask = {},
            bool return_overflowing_tokens = false,
            bool return_special_tokens_mask = false,
            bool return_offsets_mapping = false,
            bool return_length = false,
            bool verbose = true) override;

        BatchEncoding _batch_prepare_for_model(
            std::list< std::pair<std::vector<int64_t>, std::optional<std::vector<int64_t>>> >& batch_ids_pairs,
            bool add_special_tokens = false,
            PaddingStrategy padding_strategy = PaddingStrategy::DO_NOT_PAD,
            TruncationStrategy truncation_strategy = TruncationStrategy::DO_NOT_TRUNCATE,
            std::optional<int64_t> max_length = {},
            int64_t stride = 0,
            std::optional<int64_t> pad_to_multiple_of = {},
            std::optional<bool> return_token_type_ids = {},
            std::optional<bool> return_attention_mask = {},
            bool return_overflowing_tokens = false,
            bool return_special_tokens_mask = false,
            bool return_offsets_mapping = false,
            bool return_length = false,
            bool verbose = true);

        virtual std::vector< int64_t > get_special_tokens_mask(
            std::vector< int64_t>& token_ids_0, std::optional<std::vector<int64_t>> token_ids_1 = {}, bool already_has_special_tokens = false) override;

        //todo: implement this when needed.
        virtual std::string convert_ids_to_tokens(std::int64_t) override
        {
            return "blah!!";
        }

        /*
        def _convert_id_to_token(self, index: int) -> str:
            raise NotImplementedError

        */

        //todo: implement when needed
        /*
        def _decode(
            self,
            token_ids: List[int],
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: bool = None,
            spaces_between_special_tokens: bool = True,
            **kwargs,
        ) -> str:*/

        std::unordered_map<std::string, uint64_t> _added_tokens_encoder;
        std::unordered_map<uint64_t, std::string> _added_tokens_decoder;
        std::list<std::string> _unique_no_split_tokens;

        Trie _tokens_trie;
        bool _decode_use_source_tokenizer = false;
        bool _do_lower_case = false;
    };
}