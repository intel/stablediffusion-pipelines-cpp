// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#include "cpp_stable_diffusion_ov/special_tokens_mixin.h"

namespace cpp_stable_diffusion_ov
{
    SpecialTokensMixin::SpecialTokensMixin(SpecialTokensMixin_Params& init)
        : _additional_special_tokens(init.additional_special_tokens), _verbose(init.verbose)
    {
        for (auto& t : init.init_tokens)
        {
            if (_tokens.count(t.first))
            {
                _tokens[t.first] = t.second;
            }
        }
    }

    int64_t SpecialTokensMixin::add_tokens(AddedToken new_tokens, bool special_tokens)
    {
        std::list< AddedToken > tokens = { new_tokens };

        return _add_tokens(tokens, special_tokens);
    }

    int64_t SpecialTokensMixin::add_tokens(std::list<AddedToken>& new_tokens, bool special_tokens)
    {
        return _add_tokens(new_tokens, special_tokens);
    }

    std::optional< AddedToken > SpecialTokensMixin::bos_token()
    {
        if (!_tokens["bos_token"])
        {
            if (_verbose)
                fprintf(stderr, "Using bos_token, but it is not set yet.\n");
        }

        return _tokens["bos_token"];
    }

    std::optional< AddedToken > SpecialTokensMixin::eos_token()
    {
        if (!_tokens["eos_token"])
        {
            if (_verbose)
                fprintf(stderr, "Using eos_token, but it is not set yet.\n");
        }

        return _tokens["eos_token"];
    }

    std::optional< AddedToken > SpecialTokensMixin::unk_token()
    {
        if (!_tokens["unk_token"])
        {
            if (_verbose)
                fprintf(stderr, "Using unk_token, but it is not set yet.\n");
        }

        return _tokens["unk_token"];
    }

    std::optional< AddedToken > SpecialTokensMixin::sep_token()
    {
        if (!_tokens["sep_token"])
        {
            if (_verbose)
                fprintf(stderr, "Using sep_token, but it is not set yet.\n");
        }

        return _tokens["sep_token"];
    }

    std::optional< AddedToken > SpecialTokensMixin::pad_token()
    {
        if (!_tokens["pad_token"])
        {
            if (_verbose)
                fprintf(stderr, "Using pad_token, but it is not set yet.\n");
        }

        return _tokens["pad_token"];
    }

    std::optional< AddedToken > SpecialTokensMixin::cls_token()
    {
        if (!_tokens["cls_token"])
        {
            if (_verbose)
                fprintf(stderr, "Using cls_token, but it is not set yet.\n");
        }

        return _tokens["cls_token"];
    }

    std::optional< AddedToken > SpecialTokensMixin::mask_token()
    {
        if (!_tokens["mask_token"])
        {
            if (_verbose)
                fprintf(stderr, "Using mask_token, but it is not set yet.\n");
        }

        return _tokens["mask_token"];
    }

    std::list< std::string > SpecialTokensMixin::additional_special_tokens()
    {
        std::list< std::string > ret;

        for (auto& t : _additional_special_tokens)
            ret.push_back(t.content);

        return ret;
    }

    void SpecialTokensMixin::bos_token(std::optional< AddedToken > value)
    {
        _tokens["bos_token"] = value;
    }

    void SpecialTokensMixin::eos_token(std::optional< AddedToken > value)
    {
        _tokens["eos_token"] = value;
    }

    void SpecialTokensMixin::unk_token(std::optional< AddedToken > value)
    {
        _tokens["unk_token"] = value;
    }

    void SpecialTokensMixin::sep_token(std::optional< AddedToken > value)
    {
        _tokens["sep_token"] = value;
    }

    void SpecialTokensMixin::pad_token(std::optional< AddedToken > value)
    {
        _tokens["pad_token"] = value;
    }

    void SpecialTokensMixin::cls_token(std::optional< AddedToken > value)
    {
        _tokens["cls_token"] = value;
    }

    void SpecialTokensMixin::mask_token(std::optional< AddedToken > value)
    {
        _tokens["mask_token"] = value;
    }

    void SpecialTokensMixin::additional_special_tokens(std::list<AddedToken> value)
    {
        _additional_special_tokens = value;
    }

    //id 'getters'

    //`Optional[int]`: Id of the beginning of sentence token in the vocabulary. Returns `None` if the token has not
    // been set.
    std::optional<int64_t> SpecialTokensMixin::bos_token_id()
    {
        auto t = bos_token();
        if (!t)
            return {};

        return convert_tokens_to_ids(t->content);
    }

    //`Optional[int]`: Id of the end of sentence token in the vocabulary. Returns `None` if the token has not been
    // set.
    std::optional<int64_t> SpecialTokensMixin::eos_token_id()
    {
        auto t = eos_token();
        if (!t)
            return {};

        return convert_tokens_to_ids(t->content);
    }

    //`Optional[int]`: Id of the unknown token in the vocabulary.Returns `None` if the token has not been set.
    std::optional<int64_t> SpecialTokensMixin::unk_token_id()
    {
        auto t = unk_token();
        if (!t)
            return {};

        return convert_tokens_to_ids(t->content);
    }

    //`Optional[int]`: Id of the separation token in the vocabulary, to separate context and query in an input
    // sequence.Returns `None` if the token has not been set.
    std::optional<int64_t> SpecialTokensMixin::sep_token_id()
    {
        auto t = sep_token();
        if (!t)
            return {};

        return convert_tokens_to_ids(t->content);
    }

    //`Optional[int]`: Id of the padding token in the vocabulary. Returns `None` if the token has not been set.
    std::optional<int64_t> SpecialTokensMixin::pad_token_id()
    {
        auto t = pad_token();
        if (!t)
            return {};

        return convert_tokens_to_ids(t->content);
    }

    // `int`: Id of the padding token type in the vocabulary.
    int64_t SpecialTokensMixin::pad_token_type_id()
    {
        return _pad_token_type_id;
    }

    //`Optional[int]`: Id of the classification token in the vocabulary, to extract a summary of an input sequence
    // leveraging self - attention along the full depth of the model.
    int64_t SpecialTokensMixin::cls_token_id()
    {
        auto t = cls_token();
        if (!t)
            return {};

        return convert_tokens_to_ids(t->content);
    }

    //`Optional[int]`: Id of the mask token in the vocabulary, used when training a model with masked-language
    // modeling.Returns `None` if the token has not been set.
    int64_t SpecialTokensMixin::mask_token_id()
    {
        auto t = mask_token();
        if (!t)
            return {};

        return convert_tokens_to_ids(t->content);
    }

    std::list<int64_t> SpecialTokensMixin::additional_special_tokens_ids()
    {
        return convert_tokens_to_ids(_additional_special_tokens);
    }

    //id setters
    void SpecialTokensMixin::bos_token_id(std::optional<int64_t> value)
    {
        if (value)
        {
            _tokens["bos_token"] = convert_ids_to_tokens(*value);
        }
        else
        {
            _tokens["bos_token"] = {};
        }
    }

    void SpecialTokensMixin::eos_token_id(std::optional<int64_t> value)
    {
        if (value)
        {
            _tokens["eos_token"] = convert_ids_to_tokens(*value);
        }
        else
        {
            _tokens["eos_token"] = {};
        }
    }

    void SpecialTokensMixin::unk_token_id(std::optional<int64_t> value)
    {
        if (value)
        {
            _tokens["unk_token"] = convert_ids_to_tokens(*value);
        }
        else
        {
            _tokens["unk_token"] = {};
        }
    }

    void SpecialTokensMixin::sep_token_id(std::optional<int64_t> value)
    {
        if (value)
        {
            _tokens["sep_token"] = convert_ids_to_tokens(*value);
        }
        else
        {
            _tokens["sep_token"] = {};
        }
    }

    void SpecialTokensMixin::pad_token_id(std::optional<int64_t> value)
    {
        if (value)
        {
            _tokens["pad_token"] = convert_ids_to_tokens(*value);
        }
        else
        {
            _tokens["pad_token"] = {};
        }
    }

    void SpecialTokensMixin::cls_token_id(std::optional<int64_t> value)
    {
        if (value)
        {
            _tokens["cls_token"] = convert_ids_to_tokens(*value);
        }
        else
        {
            _tokens["cls_token"] = {};
        }
    }

    void SpecialTokensMixin::mask_token_id(std::optional<int64_t> value)
    {
        if (value)
        {
            _tokens["mask_token"] = convert_ids_to_tokens(*value);
        }
        else
        {
            _tokens["mask_token"] = {};
        }
    }

    void SpecialTokensMixin::additional_special_tokens_ids(std::list<int64_t> values)
    {
        std::list<AddedToken> atl;
        for (auto& value : values)
            atl.push_back(convert_ids_to_tokens(value));

        _additional_special_tokens = atl;
    }

    //little bit different than python version, as we can't
    // have a single map with AddedToken and list<AddedToken>
    // values. So we just return a pair.
    std::pair<std::unordered_map<std::string, AddedToken>,
        std::list<AddedToken>> SpecialTokensMixin::special_tokens_map_extended()
    {
        std::pair<std::unordered_map<std::string, AddedToken>,
            std::list<AddedToken>> ret;

        for (auto& t : _tokens)
        {
            if (t.second)
                ret.first.insert(std::pair< std::string, AddedToken >(t.first, *t.second));
        }

        ret.second = _additional_special_tokens;

        return ret;
    }


    std::list< AddedToken > SpecialTokensMixin::all_special_tokens_extended()
    {
        std::list< AddedToken > all_toks;
        auto set_attr = special_tokens_map_extended();

        for (auto& t : set_attr.first)
            all_toks.push_back(t.second);

        for (auto& l : set_attr.second)
            all_toks.push_back(l);

        return all_toks;
    }

    std::list<int64_t> SpecialTokensMixin::all_special_ids()
    {
        auto all_toks = all_special_tokens_extended();
        auto all_ids = convert_tokens_to_ids(all_toks);
        return all_ids;
    }

    std::list<int64_t> SpecialTokensMixin::convert_tokens_to_ids(std::list< AddedToken> atl)
    {
        std::list<int64_t> l;
        for (auto& at : atl)
            l.push_back(convert_tokens_to_ids(at.content));
        return l;
    }

    std::list<int64_t> SpecialTokensMixin::convert_tokens_to_ids(std::list< std::string>& sl)
    {
        std::list<int64_t> l;
        for (auto& s : sl)
            l.push_back(convert_tokens_to_ids(s));
        return l;
    }
}