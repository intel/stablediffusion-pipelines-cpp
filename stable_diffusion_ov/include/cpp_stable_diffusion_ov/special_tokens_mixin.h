// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
# pragma once

#include <unordered_map>
#include <string>
#include <list>
#include <vector>
#include <stdio.h>
#include <optional>
#include <stdexcept>
#include <iostream>
#include "cpp_stable_diffusion_ov/visibility.h"

namespace cpp_stable_diffusion_ov
{
    struct CPP_SD_OV_API AddedToken
    {
        AddedToken(std::string c, bool sw = false, bool l = false, bool r = false, bool n = true)
            : content(c), single_word(sw), lstrip(l), rstrip(r), normalized(n) {}

        //AddedToken(const AddedToken& at)
        //    : content(at.content), single_word(at.single_word), lstrip(at.lstrip), rstrip(at.rstrip), normalized(at.normalized) {}

        std::string content;
        bool single_word;
        bool lstrip;
        bool rstrip;
        bool normalized;
    };

    class CPP_SD_OV_API SpecialTokensMixin
    {

    public:

        struct SpecialTokensMixin_Params
        {
            std::unordered_map<std::string, AddedToken> init_tokens = {};
            std::list<AddedToken> additional_special_tokens = {};
            bool verbose = false;
        };

        SpecialTokensMixin(SpecialTokensMixin_Params& init);

        //TODO: (didn't see it called for riffusion case, so skip for now)
        //int64_t add_special_tokens(std::unordered_map<std::string, AddedToken> special_tokens_dict,
        //                           bool replace_additional_special_tokens=true);

        int64_t add_tokens(AddedToken new_tokens, bool special_tokens = false);
        int64_t add_tokens(std::list<AddedToken>& new_tokens, bool special_tokens = false);

        //getters
        std::optional< AddedToken > bos_token();
        std::optional< AddedToken > eos_token();
        std::optional< AddedToken > unk_token();
        std::optional< AddedToken > sep_token();
        std::optional< AddedToken > pad_token();
        std::optional< AddedToken > cls_token();
        std::optional< AddedToken > mask_token();
        std::list< std::string >  additional_special_tokens();

        //setters
        void bos_token(std::optional< AddedToken > value);
        void eos_token(std::optional< AddedToken > value);
        void unk_token(std::optional< AddedToken > value);
        void sep_token(std::optional< AddedToken > value);
        void pad_token(std::optional< AddedToken > value);
        void cls_token(std::optional< AddedToken > value);
        void mask_token(std::optional< AddedToken > value);
        void additional_special_tokens(std::list<AddedToken> value);

        //id 'getters'

        //`Optional[int]`: Id of the beginning of sentence token in the vocabulary. Returns `None` if the token has not
        // been set.
        std::optional<int64_t> bos_token_id();

        //`Optional[int]`: Id of the end of sentence token in the vocabulary. Returns `None` if the token has not been
        // set.
        std::optional<int64_t> eos_token_id();

        //`Optional[int]`: Id of the unknown token in the vocabulary.Returns `None` if the token has not been set.
        std::optional<int64_t> unk_token_id();

        //`Optional[int]`: Id of the separation token in the vocabulary, to separate context and query in an input
        // sequence.Returns `None` if the token has not been set.
        std::optional<int64_t> sep_token_id();

        //`Optional[int]`: Id of the padding token in the vocabulary. Returns `None` if the token has not been set.
        std::optional<int64_t> pad_token_id();

        // `int`: Id of the padding token type in the vocabulary.
        int64_t pad_token_type_id();

        //`Optional[int]`: Id of the classification token in the vocabulary, to extract a summary of an input sequence
        // leveraging self - attention along the full depth of the model.
        int64_t cls_token_id();

        //`Optional[int]`: Id of the mask token in the vocabulary, used when training a model with masked-language
        // modeling.Returns `None` if the token has not been set.
        int64_t mask_token_id();

        std::list<int64_t> additional_special_tokens_ids();

        //id setters
        void bos_token_id(std::optional<int64_t> value);
        void eos_token_id(std::optional<int64_t> value);
        void unk_token_id(std::optional<int64_t> value);
        void sep_token_id(std::optional<int64_t> value);
        void pad_token_id(std::optional<int64_t> value);
        void cls_token_id(std::optional<int64_t> value);
        void mask_token_id(std::optional<int64_t> value);
        void additional_special_tokens_ids(std::list<int64_t> values);

        //little bit different than python version, as we can't
        // have a single map with AddedToken and list<AddedToken>
        // values. So we just return a pair.
        std::pair<std::unordered_map<std::string, AddedToken>,
            std::list<AddedToken>> special_tokens_map_extended();

        std::list< std::string> all_special_tokens()
        {
            std::list< std::string> ret;

            auto added_token_list = all_special_tokens_extended();
            for (auto& at : added_token_list)
            {
                ret.push_back(at.content);
            }

            return ret;
        }
        std::list< AddedToken > all_special_tokens_extended();
        std::list<int64_t> all_special_ids();

        std::list<int64_t> convert_tokens_to_ids(std::list< std::string>& sl);

    protected:

        //these to be defined in PreTrainedTokenizer class
        virtual int64_t convert_tokens_to_ids(std::string token) = 0;
        virtual std::string convert_ids_to_tokens(std::int64_t) = 0;

        // Add a list of new tokens to the tokenizer class. If the new tokens are not in the vocabulary, they are added to
        // it with indices starting from length of the current vocabulary.
        virtual int64_t _add_tokens(std::list<AddedToken>& new_tokens, bool special_tokens = false) = 0;

        std::list<int64_t> convert_tokens_to_ids(std::list< AddedToken> atl);

        std::unordered_map<std::string, std::optional< AddedToken >> _tokens =
        {
            {"bos_token", {}},
            {"eos_token", {}},
            {"unk_token", {}},
            {"sep_token", {}},
            {"pad_token", {}},
            {"cls_token", {}},
            {"mask_token", {}},
        };

        std::list< AddedToken> _additional_special_tokens;
        bool _verbose;
        int64_t _pad_token_type_id = 0;



    };
}