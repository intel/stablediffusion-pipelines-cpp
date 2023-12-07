// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#include "cpp_stable_diffusion_ov/pretrained_tokenizer.h"

namespace cpp_stable_diffusion_ov
{
    int64_t PreTrainedTokenizer::_add_tokens(std::list<AddedToken>& new_tokens_orig, bool special_tokens)
    {
        std::list< std::string> new_tokens;
        for (auto& tok : new_tokens_orig)
        {
            new_tokens.push_back(tok.content);
        }

        std::list< std::string> tokens_to_add;
        for (auto& token : new_tokens)
        {
            if (!special_tokens && _do_lower_case)
            {
                to_lower(token);
            }

            if ((unk_token() || (token != unk_token()->content))
                && (convert_tokens_to_ids(token) == convert_tokens_to_ids(unk_token()->content))
                && !(std::find(std::begin(tokens_to_add), std::end(tokens_to_add), token) != std::end(tokens_to_add)))
            {
                tokens_to_add.push_back(token);
                if (_verbose)
                    fprintf(stderr, "Adding %s to the vocabulary", token.c_str());
            }
        }

        std::unordered_map<std::string, uint64_t> added_tok_encoder;
        int64_t i = 0;
        for (auto& tok : tokens_to_add)
        {
            added_tok_encoder[tok] = len() + i;
            i++;
        }

        std::unordered_map<uint64_t, std::string> added_tok_decoder;
        for (auto& items : added_tok_encoder)
        {
            auto k = items.first;
            auto v = items.second;
            added_tok_decoder[v] = k;
        }

        _added_tokens_encoder.insert(added_tok_encoder.begin(), added_tok_encoder.end());
        _added_tokens_decoder.insert(added_tok_decoder.begin(), added_tok_decoder.end());

        // Make sure we don't split on any special tokens (even they were already in the vocab before e.g. for Albert)
        if (special_tokens)
        {
            if (new_tokens.size() == 1)
            {
                _insert_one_token_to_ordered_list(_unique_no_split_tokens, new_tokens.front());
            }
            else
            {
                std::set< std::string > sorted_strings_set;
                for (auto& s : new_tokens)
                {
                    sorted_strings_set.insert(s);
                }

                for (auto& s : _unique_no_split_tokens)
                {
                    sorted_strings_set.insert(s);
                }

                _unique_no_split_tokens = {};
                for (auto& s : sorted_strings_set)
                {
                    _unique_no_split_tokens.push_back(s);
                }
            }
        }
        else
        {
            if (tokens_to_add.size() == 1)
            {
                _insert_one_token_to_ordered_list(_unique_no_split_tokens, tokens_to_add.front());
            }
            else
            {
                std::set< std::string > sorted_strings_set;
                for (auto& s : tokens_to_add)
                {
                    sorted_strings_set.insert(s);
                }

                for (auto& s : _unique_no_split_tokens)
                {
                    sorted_strings_set.insert(s);
                }

                _unique_no_split_tokens = {};
                for (auto& s : sorted_strings_set)
                {
                    _unique_no_split_tokens.push_back(s);
                }
            }
        }

        _create_trie(_unique_no_split_tokens);

        return tokens_to_add.size();
    }

    void PreTrainedTokenizer::_create_trie(std::list<std::string>& unique_no_split_tokens)
    {
        Trie trie;
        for (auto token : unique_no_split_tokens)
        {
            auto special_tokens = all_special_tokens();
            bool token_in_all_special_tokens = std::find(std::begin(special_tokens), std::end(special_tokens), token) != std::end(special_tokens);
            if (_do_lower_case && !token_in_all_special_tokens)
            {
                to_lower(token);
                trie.add(token);
            }
            else
            {
                trie.add(token);
            }
        }

        _tokens_trie = trie;
    }

    int64_t PreTrainedTokenizer::num_special_tokens_to_add(bool pair)
    {
        std::vector<int64_t> token_ids_0;
        std::optional<std::vector<int64_t>> token_ids_1;
        if (pair)
            token_ids_1 = std::vector<int64_t>();

        return build_inputs_with_special_tokens(token_ids_0, token_ids_1).size();
    }

    std::list< std::string > PreTrainedTokenizer::tokenize(std::string text, std::optional<std::string> pair, bool add_special_tokens)
    {
        // Simple mapping string => AddedToken for special tokens with specific tokenization behaviors
        std::map<std::string, AddedToken> all_special_tokens_extended;
        for (auto at : this->all_special_tokens_extended())
        {
            all_special_tokens_extended.insert(std::pair< std::string, AddedToken>(at.content, at));
        }

#if 0
        for (auto at : all_special_tokens_extended)
        {
            std::cout << at.first << at.second.content << std::endl;
        }
#endif

        text = prepare_for_tokenization(text);



        /*
        * TODO:
        * if hasattr(self, "do_lower_case") and self.do_lower_case:
            # convert non-special tokens to lowercase
            escaped_special_toks = [
                re.escape(s_tok) for s_tok in (self.unique_no_split_tokens + self.all_special_tokens)
            ]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            text = re.sub(pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), text)
        */

        std::set<std::string> no_split_token(_unique_no_split_tokens.begin(), _unique_no_split_tokens.end());
        auto tokens = _tokens_trie.split(text);
        // ["This is something", "<special_token_1>", "  else"]
        std::vector<std::string> token_vec(tokens.begin(), tokens.end());


        auto tmp_token_vec = token_vec;
        for (size_t i = 0; i < tmp_token_vec.size(); i++)
        {
            auto token = token_vec[i];
            if (no_split_token.count(token))
            {
                std::optional<std::string> left = {};
                std::optional<std::string> right = {};
                if (i > 0)
                    left = token_vec[i - 1];

                if (i < (token_vec.size() - 1))
                    right = token_vec[i + 1];

                auto it = all_special_tokens_extended.find(token);
                if (it != all_special_tokens_extended.end())
                {
                    auto& at = it->second;
                    if (at.rstrip && right)
                    {
                        // A bit counter - intuitive but we strip the left of the string
                        // since tok_extended.rstrip means the special token is eating all white spaces on its right
                        token_vec[i + 1] = ltrim(*right);
                    }

                    if (at.lstrip && left)
                    {
                        // A bit counter - intuitive but we strip the left of the string
                        // since tok_extended.rstrip means the special token is eating all white spaces on its right
                        token_vec[i - 1] = ltrim(*left);
                    }
                }
                else
                {
                    // We strip left and right by default
                    if (right)
                    {
                        token_vec[i + 1] = ltrim(*right);
                    }

                    if (left)
                    {
                        token_vec[i - 1] = ltrim(*left);
                    }
                }
            }
        }



        std::list< std::string > tokenized_text;
        for (auto& token : token_vec)
        {
            // Need to skip eventual empty (fully stripped) tokens
            if (token.empty())
                continue;
            if (no_split_token.count(token))
            {
                tokenized_text.push_back(token);
            }
            else
            {
                auto _tokenize_output = _tokenize(token);
                tokenized_text.insert(tokenized_text.end(), _tokenize_output.begin(), _tokenize_output.end());
            }
        }

        //std::cout << "<-tokenize()" << std::endl;
        return tokenized_text;

    }

    BatchEncoding PreTrainedTokenizer::_encode_plus(std::optional<std::string> text,
        std::optional<std::string> text_pair,
        bool add_special_tokens,
        PaddingStrategy padding_strategy,
        TruncationStrategy truncation_strategy,
        std::optional<int64_t> max_length,
        int64_t stride,
        bool is_split_into_words,
        std::optional<int64_t> pad_to_multiple_of,
        std::optional<bool> return_token_type_ids,
        std::optional<bool> return_attention_mask,
        bool return_overflowing_tokens,
        bool return_special_tokens_mask,
        bool return_offsets_mapping,
        bool return_length,
        bool verbose)
    {
        if (!text)
            throw std::invalid_argument("text must be specified");

        std::vector<int64_t> first_ids;
        {
            auto tmp_tokens = tokenize(*text); // this dumb. thanks gcc. 

            auto tmp_list = convert_tokens_to_ids(tmp_tokens);

            first_ids = std::vector<int64_t>(tmp_list.begin(), tmp_list.end());

#if 0
            std::cout << "first ids = " << std::endl;
            for (auto& f : first_ids)
            {
                std::cout << "    " << f << std::endl;
            }
#endif

        }

        std::optional<std::vector<int64_t>> second_ids;
        if (text_pair)
        {
            auto tmp_tokens = tokenize(*text_pair);
            auto tmp_list = convert_tokens_to_ids(tmp_tokens);
            second_ids = std::vector<int64_t>(tmp_list.begin(), tmp_list.end());
        }


        return prepare_for_model(first_ids,
            second_ids,
            add_special_tokens,
            padding_strategy,
            truncation_strategy,
            max_length,
            stride,
            pad_to_multiple_of,
            return_token_type_ids,
            return_attention_mask,
            return_overflowing_tokens,
            return_special_tokens_mask,
            return_offsets_mapping,
            return_length,
            verbose,
            true //prepend_batch_axis
        );
    }

    BatchEncoding PreTrainedTokenizer::_batch_encode_plus(std::optional<std::list<std::string>> text,
        std::optional<std::list<std::string>> text_pair,
        bool add_special_tokens,
        PaddingStrategy padding_strategy,
        TruncationStrategy truncation_strategy,
        std::optional<int64_t> max_length,
        int64_t stride,
        bool is_split_into_words,
        std::optional<int64_t> pad_to_multiple_of,
        std::optional<bool> return_token_type_ids,
        std::optional<bool> return_attention_mask,
        bool return_overflowing_tokens,
        bool return_special_tokens_mask,
        bool return_offsets_mapping,
        bool return_length,
        bool verbose)
    {
        if (!text)
            throw std::invalid_argument("text must be specified");

        std::list< std::pair<std::vector<int64_t>, std::optional<std::vector<int64_t>>> > input_ids;
        if (text_pair)
        {
            if (text_pair->size() != text->size())
            {
                throw std::invalid_argument("if text pair list is specified, it must have same size as text list");
            }
        }

        std::list<std::string>::iterator text_it = text->begin();
        std::list<std::string>::iterator text_pair_it;
        if (text_pair)
            text_pair_it = text_pair->begin();

        for (size_t i = 0; i < text->size(); i++)
        {
            std::vector<int64_t> first_ids;
            {
                auto tmp_tokens = tokenize(*text_it);
                auto tmp_list = convert_tokens_to_ids(tmp_tokens);
                first_ids = std::vector<int64_t>(tmp_list.begin(), tmp_list.end());
            }

            std::optional<std::vector<int64_t>> second_ids;
            if (text_pair)
            {
                auto tmp_tokens = tokenize(*text_pair_it);
                auto tmp_list = convert_tokens_to_ids(tmp_tokens);
                second_ids = std::vector<int64_t>(tmp_list.begin(), tmp_list.end());
            }

            input_ids.push_back({ first_ids, second_ids });
            text_it++;
            if (text_pair)
                text_pair_it++;
        }

        return _batch_prepare_for_model(
            input_ids,
            add_special_tokens,
            padding_strategy,
            truncation_strategy,
            max_length,
            stride,
            pad_to_multiple_of,
            return_token_type_ids,
            return_attention_mask,
            return_overflowing_tokens,
            return_special_tokens_mask,
            return_length,
            verbose);
    }

    BatchEncoding PreTrainedTokenizer::_batch_prepare_for_model(
        std::list< std::pair<std::vector<int64_t>, std::optional<std::vector<int64_t>>> >& batch_ids_pairs,
        bool add_special_tokens,
        PaddingStrategy padding_strategy,
        TruncationStrategy truncation_strategy,
        std::optional<int64_t> max_length,
        int64_t stride,
        std::optional<int64_t> pad_to_multiple_of,
        std::optional<bool> return_token_type_ids,
        std::optional<bool> return_attention_mask,
        bool return_overflowing_tokens,
        bool return_special_tokens_mask,
        bool return_offsets_mapping,
        bool return_length,
        bool verbose)
    {
        BatchEncoding batch_outputs;
        for (auto& id_pairs : batch_ids_pairs)
        {
            auto& first_ids = id_pairs.first;
            auto& second_ids = id_pairs.second;

            auto outputs = prepare_for_model(
                first_ids,
                second_ids,
                add_special_tokens,
                PaddingStrategy::DO_NOT_PAD, // we pad in batch afterward
                truncation_strategy,
                max_length,
                stride,
                {}, // pad_to_multiple_of=None, we pad in batch afterward
                return_token_type_ids,
                false, //return_attention_mask, we pad in batch afterward
                return_overflowing_tokens,
                return_special_tokens_mask,
                false,
                return_length,
                verbose,
                false
            );

            for (auto& output_pairs : outputs)
            {
                auto& key = output_pairs.first;
                auto& value = output_pairs.second;

                if (batch_outputs.count(key) == 0)
                {
                    batch_outputs[key] = std::vector<std::vector<int64_t>>();
                }
                batch_outputs[key].push_back(value[0]);
            }
        }

        batch_outputs = pad(
            batch_outputs,
            padding_strategy,
            max_length,
            pad_to_multiple_of,
            return_attention_mask);

        return batch_outputs;
    }

    std::vector< int64_t > PreTrainedTokenizer::get_special_tokens_mask(
        std::vector< int64_t>& token_ids_0, std::optional<std::vector<int64_t>> token_ids_1, bool already_has_special_tokens)
    {
        if (already_has_special_tokens)
        {
            if (token_ids_1)
            {
                throw std::invalid_argument(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model.");
            }

            return PreTrainedTokenizerBase::get_special_tokens_mask(
                token_ids_0, token_ids_1, already_has_special_tokens);
        }


        //return [0] * ((len(token_ids_1) if token_ids_1 else 0) + len(token_ids_0))
        size_t len = token_ids_0.size();
        if (token_ids_1)
            len += token_ids_1->size();
        return std::vector< int64_t>(len, 0);
    }
}