// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#include "cpp_stable_diffusion_ov/pretrained_tokenizer_base.h"

namespace cpp_stable_diffusion_ov
{
    PreTrainedTokenizerBase::PreTrainedTokenizerBase(PreTrainedTokenizerBase_Params init)
        : SpecialTokensMixin(init.specialTokensInit)
    {
        _model_max_length = init.model_max_length;

        if ((init.padding_side != "right") && (init.padding_side != "left"))
        {
            throw std::invalid_argument("Padding side should be selected between 'right' and 'left', current value: " + init.padding_side);
        }
        _padding_side = init.padding_side;

        if ((init.truncation_side != "right") && (init.truncation_side != "left"))
        {
            throw std::invalid_argument("Truncation side should be selected between 'right' and 'left', current value: " + init.truncation_side);
        }

        _truncation_side = init.truncation_side;

        _clean_up_tokenization_spaces = init.clean_up_tokenization_spaces;
    }

    int64_t PreTrainedTokenizerBase::max_len_single_sentence()
    {
        return _model_max_length - num_special_tokens_to_add(false);
    }

    // The maximum combined length of a pair of sentences that can be fed to the model.
    int64_t PreTrainedTokenizerBase::max_len_sentences_pair()
    {
        return _model_max_length - num_special_tokens_to_add(true);
    }

    void PreTrainedTokenizerBase::_set_processor_class(std::optional<std::string> processor_class)
    {
        _processor_class = processor_class;
    }

    BatchEncoding PreTrainedTokenizerBase::call(std::optional<std::string> text,
        std::optional<std::string> text_pair,
        std::optional<std::string> text_target,
        std::optional<std::string> text_pair_target,
        bool add_special_tokens,
        std::optional<Padding> padding,
        std::optional<Truncation> truncation,
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
        if (!text && !text_target)
            throw std::invalid_argument("You need to specify either `text` or `text_target`.");

        BatchEncoding encodings;
        if (text)
        {
            // The context manager will send the inputs as normal texts and not text_target, but we shouldn't change the
            // input mode in this case.
            if (!_in_target_context_manager)
            {
                _switch_to_input_mode();
            }
            encodings = _call_one(text, text_pair, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of,
                return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping,
                return_length, verbose);
        }

        BatchEncoding target_encodings;
        if (text_target)
        {
            _switch_to_target_mode();
            target_encodings = _call_one(text_target, text_pair_target, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of,
                return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping,
                return_length, verbose);
        }

        // Leave back tokenizer in input mode
        _switch_to_input_mode();

        if (!text_target)
        {
            return encodings;
        }
        else if (!text)
        {
            return target_encodings;
        }
        else
        {
            encodings["labels"] = target_encodings["input_ids"];
            return encodings;
        }
    }

    BatchEncoding PreTrainedTokenizerBase::call(std::optional<std::list<std::string>> text,
        std::optional<std::list<std::string>> text_pair,
        std::optional<std::list<std::string>> text_target,
        std::optional<std::list<std::string>> text_pair_target,
        bool add_special_tokens,
        std::optional<Padding> padding,
        std::optional<Truncation> truncation,
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
        if (!text && !text_target)
            throw std::invalid_argument("You need to specify either `text` or `text_target`.");

        BatchEncoding encodings;
        if (text)
        {
            // The context manager will send the inputs as normal texts and not text_target, but we shouldn't change the
            // input mode in this case.
            if (!_in_target_context_manager)
            {
                _switch_to_input_mode();
            }
            encodings = _batch_call_one(text, text_pair, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of,
                return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping,
                return_length, verbose);
        }

        BatchEncoding target_encodings;
        if (text_target)
        {
            _switch_to_target_mode();
            target_encodings = _batch_call_one(text_target, text_pair_target, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of,
                return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping,
                return_length, verbose);
        }

        // Leave back tokenizer in input mode
        _switch_to_input_mode();

        if (!text_target)
        {
            return encodings;
        }
        else if (!text)
        {
            return target_encodings;
        }
        else
        {
            encodings["labels"] = target_encodings["input_ids"];
            return encodings;
        }
    }

    BatchEncoding PreTrainedTokenizerBase::_call_one(std::optional<std::string> text,
        std::optional<std::string> text_pair,
        bool add_special_tokens,
        std::optional<Padding> padding,
        std::optional<Truncation> truncation,
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
        //TODO: Don't make it optional param then? :)
        if (!text)
            throw std::invalid_argument("text must be specified");


        //if is_split_into_words:
        //    is_batched = isinstance(text, (list, tuple)) and text and isinstance(text[0], (list, tuple))
        //else:
        //    is_batched = isinstance(text, (list, tuple))
        // 
        // is_batched is always false, at least for this function signature as we know we only 
        // have a single string.
        return encode_plus(text, text_pair, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of,
            return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping,
            return_length, verbose);
    }

    BatchEncoding PreTrainedTokenizerBase::_batch_call_one(std::optional<std::list<std::string>> text,
        std::optional<std::list<std::string>> text_pair,
        bool add_special_tokens,
        std::optional<Padding> padding,
        std::optional<Truncation> truncation,
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
        //TODO: Don't make it optional param then? :)
        if (!text)
            throw std::invalid_argument("text must be specified");

        if (text_pair && (text->size() != text_pair->size()))
        {
            throw std::invalid_argument("batch length of `text`: " + std::to_string(text->size()) +
                "does not match batch length of `text_pair`:" + std::to_string(text_pair->size()));
        }

        return batch_encode_plus(text, text_pair, add_special_tokens, padding, truncation, max_length, stride, is_split_into_words, pad_to_multiple_of,
            return_token_type_ids, return_attention_mask, return_overflowing_tokens, return_special_tokens_mask, return_offsets_mapping,
            return_length, verbose);
    }

    BatchEncoding PreTrainedTokenizerBase::encode_plus(std::optional<std::string> text,
        std::optional<std::string> text_pair,
        bool add_special_tokens,
        std::optional<Padding> padding,
        std::optional<Truncation> truncation,
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
        BatchEncoding enc;

        PaddingStrategy padding_strategy;
        TruncationStrategy truncation_strategy;

        std::tie(padding_strategy, truncation_strategy, max_length) = _get_padding_truncation_strategies(padding,
            truncation,
            max_length,
            pad_to_multiple_of,
            verbose //, TODO: kwargs?
        );

        return _encode_plus(
            text,
            text_pair,
            add_special_tokens,
            padding_strategy,
            truncation_strategy,
            max_length,
            stride,
            is_split_into_words,
            pad_to_multiple_of,
            return_token_type_ids,
            return_attention_mask,
            return_overflowing_tokens,
            return_special_tokens_mask,
            return_offsets_mapping,
            return_length,
            verbose);
    }

    BatchEncoding PreTrainedTokenizerBase::batch_encode_plus(std::optional<std::list<std::string>> text,
        std::optional<std::list<std::string>> text_pair,
        bool add_special_tokens,
        std::optional<Padding> padding,
        std::optional<Truncation> truncation,
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
        BatchEncoding enc;

        PaddingStrategy padding_strategy;
        TruncationStrategy truncation_strategy;

        std::tie(padding_strategy, truncation_strategy, max_length) = _get_padding_truncation_strategies(
            padding,
            truncation,
            max_length,
            pad_to_multiple_of,
            verbose //, TODO: kwargs?
        );

        return _batch_encode_plus(
            text,
            text_pair,
            add_special_tokens,
            padding_strategy,
            truncation_strategy,
            max_length,
            stride,
            is_split_into_words,
            pad_to_multiple_of,
            return_token_type_ids,
            return_attention_mask,
            return_overflowing_tokens,
            return_special_tokens_mask,
            return_offsets_mapping,
            return_length,
            verbose);
    }

    BatchEncoding PreTrainedTokenizerBase::pad(BatchEncoding& encoded_inputs,
        std::optional<Padding> padding,
        std::optional<int64_t> max_length,
        std::optional<int64_t> pad_to_multiple_of,
        std::optional<bool> return_attention_mask,
        bool verbose)
    {

        if (!encoded_inputs.count(_model_input_names.front()))
        {
            throw std::invalid_argument("You should supply an encoding or a list of encodings to this method "
                "that includes" + _model_input_names.front());
        }

        auto required_input = encoded_inputs[_model_input_names.front()];

        //todo: need this kind of logic? If so, should BatchEncoding value be an std::optional?
        //if required_input is None or (isinstance(required_input, Sized) and len(required_input) == 0) :
        //    if return_attention_mask :
        //        encoded_inputs["attention_mask"] = []
        //        return encoded_inputs

        // in tokenization_utils_base.py, there's a bunch of logic that checks instance types, first element, etc.
        // we don't need any of that stuff..

        // Convert padding_strategy in PaddingStrategy
        PaddingStrategy padding_strategy;
        TruncationStrategy _;
        std::tie(padding_strategy, _, max_length) = _get_padding_truncation_strategies(
            padding,
            {}, //truncation
            max_length,
            {}, // pad_to_multiple_of
            verbose //, TODO: kwargs?
        );

        required_input = encoded_inputs[_model_input_names.front()];


        //todo: python version handles the case where required_input *is* a Batch1Encoding.
        // maybe we need to expose this as another overload?
        //if required_input and not isinstance(required_input[0], (list, tuple)) :
        //    encoded_inputs = self._pad(
        //        encoded_inputs,
        //        max_length = max_length,
        //        padding_strategy = padding_strategy,
        //        pad_to_multiple_of = pad_to_multiple_of,
        //        return_attention_mask = return_attention_mask,
        //        )
        //    return BatchEncoding(encoded_inputs, tensor_type = return_tensors)

        auto batch_size = required_input.size();
        for (auto& ei : encoded_inputs)
        {
            if (ei.second.size() != batch_size)
            {
                throw std::invalid_argument("Some items in the output dictionary have a different batch size than others.");
            }
        }

        if (padding_strategy == PaddingStrategy::LONGEST)
        {
            //std::cout << "pad: padding_strategy = PaddingStrategy::LONGEST" << std::endl;
            size_t max_length_tmp = 0;
            for (auto& inputs : required_input)
            {
                max_length_tmp = std::max(max_length_tmp, inputs.size());
            }

            max_length = (int64_t)max_length_tmp;
            padding_strategy = PaddingStrategy::MAX_LENGTH;
        }

        BatchEncoding batch_outputs;
        for (size_t bi = 0; bi < batch_size; bi++)
        {
            //auto inputs = encoded_inputs;
            Batch1Encoding inputs;
            for (auto& ei : encoded_inputs)
            {
                inputs[ei.first] = ei.second[bi];
            }

            Batch1Encoding outputs = _pad(inputs,
                max_length,
                padding_strategy,
                pad_to_multiple_of,
                return_attention_mask);

            for (auto& o : outputs)
            {
                if (!batch_outputs.count(o.first))
                {
                    batch_outputs[o.first] = std::vector<std::vector<int64_t>>();
                }
                batch_outputs[o.first].push_back(o.second);
            }
        }

        return batch_outputs;
    }

    Batch1Encoding PreTrainedTokenizerBase::_pad(Batch1Encoding& encoded_inputs,
            std::optional<int64_t> max_length,
            PaddingStrategy padding_strategy,
            std::optional<int64_t> pad_to_multiple_of,
            std::optional<bool> return_attention_mask)
    {
        // Load from model defaults
        if (!return_attention_mask)
        {
            return_attention_mask = (std::find(_model_input_names.begin(), _model_input_names.end(), "attention_mask") != _model_input_names.end());
        }

        auto required_input = encoded_inputs[_model_input_names.front()];

        if (padding_strategy == PaddingStrategy::LONGEST)
        {
            max_length = required_input.size();
        }

        if (max_length && pad_to_multiple_of && (*max_length % *pad_to_multiple_of != 0))
        {
            max_length = ((*max_length / *pad_to_multiple_of) + 1) * *pad_to_multiple_of;
        }


        bool needs_to_be_padded = true;
        if (max_length)
        {
            //std::cout << "padding_strategy = " << (int)padding_strategy << std::endl;
            //std::cout << "max_length = " << *max_length << std::endl;
            //std::cout << "required_input.size() = " << required_input.size() << std::endl;
            needs_to_be_padded = (padding_strategy != PaddingStrategy::DO_NOT_PAD) && (required_input.size() != *max_length);
        }

        // Initialize attention mask if not present.
        if (*return_attention_mask && !encoded_inputs.count("attention_mask"))
        {
            //encoded_inputs["attention_mask"] = [1] * len(required_input)
            encoded_inputs["attention_mask"] = std::vector<int64_t>(required_input.size(), 1);
        }

        if (needs_to_be_padded)
        {
            int64_t difference = *max_length - required_input.size();
            auto zeros_difference = std::vector<int64_t>(difference, 0);
            auto ones_difference = std::vector<int64_t>(difference, 1);
            auto pad_token_difference = std::vector<int64_t>(difference, *pad_token_id());
        
            if (_padding_side == "right")
            {
                if (*return_attention_mask)
                {

                    //encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"] + [0] * difference
                    encoded_inputs["attention_mask"].insert(encoded_inputs["attention_mask"].end(), zeros_difference.begin(), zeros_difference.end());
                }

                if (encoded_inputs.count("token_type_ids"))
                {
                    //encoded_inputs["token_type_ids"] = (
                    //    encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference
                    //    )
                    auto pad_token_type_difference = std::vector<int64_t>(difference, pad_token_type_id());
                    encoded_inputs["token_type_ids"].insert(encoded_inputs["token_type_ids"].end(), pad_token_type_difference.begin(), pad_token_type_difference.end());
                }

                if (encoded_inputs.count("special_tokens_mask"))
                {
                    //encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
                    encoded_inputs["special_tokens_mask"].insert(encoded_inputs["special_tokens_mask"].end(), ones_difference.begin(), ones_difference.end());
                }

                //encoded_inputs[self.model_input_names[0]] = required_input + [self.pad_token_id] * difference
                encoded_inputs[_model_input_names.front()].insert(encoded_inputs[_model_input_names.front()].end(), pad_token_difference.begin(), pad_token_difference.end());
            }
            else if (_padding_side == "left")
            {
                if (*return_attention_mask)
                {

                    //encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"] + [0] * difference
                    encoded_inputs["attention_mask"].insert(encoded_inputs["attention_mask"].begin(), zeros_difference.begin(), zeros_difference.end());
                }

                if (encoded_inputs.count("token_type_ids"))
                {
                    //encoded_inputs["token_type_ids"] = (
                    //    encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference
                    //    )
                    auto pad_token_type_difference = std::vector<int64_t>(difference, pad_token_type_id());
                    encoded_inputs["token_type_ids"].insert(encoded_inputs["token_type_ids"].begin(), pad_token_type_difference.begin(), pad_token_type_difference.end());
                }

                if (encoded_inputs.count("special_tokens_mask"))
                {
                    //encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
                    encoded_inputs["special_tokens_mask"].insert(encoded_inputs["special_tokens_mask"].begin(), ones_difference.begin(), ones_difference.end());
                }

                //encoded_inputs[self.model_input_names[0]] = required_input + [self.pad_token_id] * difference
                encoded_inputs[_model_input_names.front()].insert(encoded_inputs[_model_input_names.front()].begin(), pad_token_difference.begin(), pad_token_difference.end());
            }
            else
            {
                throw std::invalid_argument("Invalid padding strategy:" + _padding_side);
            }
        }

        return encoded_inputs;
    }

    std::vector< int64_t > PreTrainedTokenizerBase::create_token_type_ids_from_sequences(std::vector<int64_t>& token_ids_0, std::optional<std::vector<int64_t>> token_ids_1)
    {
        if (!token_ids_1)
        {
            return std::vector<int64_t>(token_ids_0.size(), 0);
        }

        std::vector< int64_t > ret = std::vector<int64_t>(token_ids_0.size(), 0);
        auto ones = std::vector<int64_t>(token_ids_1->size(), 1);
        ret.insert(ret.end(), ones.begin(), ones.end());
        return ret;
    }

    std::vector<int64_t> PreTrainedTokenizerBase::build_inputs_with_special_tokens(std::vector<int64_t>& token_ids_0, std::optional<std::vector<int64_t>> token_ids_1)
    {
        if (!token_ids_1)
            return token_ids_0;

        std::vector< int64_t > ret = token_ids_0;
        ret.insert(ret.end(), token_ids_1->begin(), token_ids_1->end());
        return ret;
    }

    BatchEncoding PreTrainedTokenizerBase::prepare_for_model(std::vector<int64_t> ids,
        std::optional<std::vector<int64_t>> pair_ids,
        bool add_special_tokens,
        Padding padding,
        std::optional<Truncation> truncation,
        std::optional<int64_t> max_length,
        int64_t stride,
        std::optional<int64_t> pad_to_multiple_of,
        std::optional<bool> return_token_type_ids,
        std::optional<bool> return_attention_mask,
        bool return_overflowing_tokens,
        bool return_special_tokens_mask,
        bool return_offsets_mapping,
        bool return_length,
        bool verbose,
        bool prepend_batch_axis)
    {
        PaddingStrategy padding_strategy;
        TruncationStrategy truncation_strategy;

        // Backward compatibility for 'truncation_strategy', 'pad_to_max_length'
        std::tie(padding_strategy, truncation_strategy, max_length) = _get_padding_truncation_strategies(padding,
            truncation,
            max_length,
            pad_to_multiple_of,
            verbose //, TODO: kwargs?
        );

        //std::cout << "prepare_for_model: padding_strategy = " << (int)padding_strategy << std::endl;

        bool pair = (pair_ids) ? true : false;
        size_t len_ids = ids.size();
        size_t len_pair_ids = 0;
        if (pair_ids)
            len_pair_ids = pair_ids->size();

        if (return_token_type_ids && !add_special_tokens)
        {
            throw std::invalid_argument("Asking to return token_type_ids while setting add_special_tokens to false "
                "results in an undefined behavior. Please set add_special_tokens to true or "
                "set return_token_type_ids to {}.");
        }

        if (return_overflowing_tokens
            && truncation_strategy == TruncationStrategy::LONGEST_FIRST
            && pair_ids)
        {
            throw std::invalid_argument(
                "Not possible to return overflowing tokens for pair of sequences with the "
                "`longest_first`. Please select another truncation strategy than `longest_first`, "
                "for instance `only_second` or `only_first`."
            );
        }

        // Load from model defaults
        if (!return_token_type_ids)
        {
            return_token_type_ids = std::find(_model_input_names.begin(), _model_input_names.end(), "token_type_ids") != _model_input_names.end();
            //std::cout << "return_token_type_ids is none, so setting to " << *return_token_type_ids << std::endl;
        }
    
        if (!return_attention_mask)
            return_attention_mask = std::find(_model_input_names.begin(), _model_input_names.end(), "attention_mask") != _model_input_names.end();

        BatchEncoding encoded_inputs;

        // Compute the total size of the returned encodings
        size_t total_len = len_ids + len_pair_ids + (add_special_tokens ? num_special_tokens_to_add(pair) : 0);

        // Truncation: Handle max sequence length
        std::vector<int64_t> overflowing_tokens;
        if ((truncation_strategy != TruncationStrategy::DO_NOT_TRUNCATE) && max_length && total_len > *max_length)
        {
            std::tie(ids, pair_ids, overflowing_tokens) = truncate_sequences(
                ids,
                pair_ids,
                (total_len - *max_length),
                truncation_strategy,
                stride
            );
        }

        if (return_overflowing_tokens)
        {
            //idk, python just shoves a list<int> and (int) into a (primarily) <string, list<list<int>> > dict. So
            // we sort of hackily get by here. 
            encoded_inputs["overflowing_tokens"] = { overflowing_tokens };
            encoded_inputs["num_truncated_tokens"] = { {(int64_t)total_len - *max_length} };
        }

        // Add special tokens
        std::vector<int64_t> sequence;
        std::vector<int64_t> token_type_ids;


        if (add_special_tokens)
        {
            //std::cout << "prepare_for_model: add_special_tokens = true" << std::endl;
            sequence = build_inputs_with_special_tokens(ids, pair_ids);
            token_type_ids = create_token_type_ids_from_sequences(ids, pair_ids);
        }
        else
        {
            //std::cout << "prepare_for_model: add_special_tokens = false" << std::endl;
            sequence = ids;
            token_type_ids = std::vector<int64_t>(ids.size(), 0);
            if (pair)
            {
                sequence.insert(sequence.begin(), pair_ids->begin(), pair_ids->end());
                auto tmp = std::vector<int64_t>(pair_ids->size(), 0);
                token_type_ids.insert(token_type_ids.begin(), tmp.begin(), tmp.end());
            }
        }

        // Build output dictionary
        encoded_inputs["input_ids"] = { sequence };
        if (return_token_type_ids && *return_token_type_ids)
        {
            encoded_inputs["token_type_ids"] = { token_type_ids };
        }
        if (return_special_tokens_mask)
        {
            if (add_special_tokens)
            {
                //encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(ids, pair_ids)
                encoded_inputs["special_tokens_mask"] = { get_special_tokens_mask(ids, pair_ids) };
            }
            else
            {
                //encoded_inputs["special_tokens_mask"] = [0] * len(sequence)
                encoded_inputs["special_tokens_mask"] = { std::vector<int64_t>(sequence.size(), 0) };
            }
        }

        // Check lengths
        //todo:
        // self._eventual_warn_about_too_long_sequence(encoded_inputs["input_ids"], max_length, verbose)

        // Padding
        if ((padding_strategy != PaddingStrategy::DO_NOT_PAD) || return_attention_mask)
        {
            encoded_inputs = pad(
                encoded_inputs,
                padding_strategy,
                max_length,
                pad_to_multiple_of,
                return_attention_mask);
        }

        if (return_length)
        {
            encoded_inputs["length"] = { {(int64_t)encoded_inputs["input_ids"].size()} };
        }

        //todo:
        //batch_outputs = BatchEncoding(
        //    encoded_inputs, tensor_type = return_tensors, prepend_batch_axis = prepend_batch_axis
        //)

        auto batch_outputs = encoded_inputs;

        return batch_outputs;
    }

    std::vector< int64_t > PreTrainedTokenizerBase::get_special_tokens_mask(
        std::vector< int64_t>& token_ids_0, std::optional<std::vector<int64_t>> token_ids_1, bool already_has_special_tokens)
    {
        if (!(already_has_special_tokens && !token_ids_1))
        {
            throw std::invalid_argument(
                "You cannot use ``already_has_special_tokens=False`` with this tokenizer. "
                "Please use a slow (full python) tokenizer to activate this argument. "
                "Or set `return_special_tokens_mask=True` when calling the encoding method "
                "to get the special tokens mask in any tokenizer. "
            );

        }
        auto spec_ids = all_special_ids();

        std::vector<int64_t> special_tokens_mask;
        for (auto& token : token_ids_0)
        {
            if (std::find(spec_ids.begin(), spec_ids.end(), token) != spec_ids.end())
            {
                special_tokens_mask.push_back(1);
            }
            else
            {
                special_tokens_mask.push_back(0);
            }
        }

        return special_tokens_mask;
    }

    // Truncates a sequence pair in-place following the strategy.
    std::tuple<  std::vector<int64_t>, std::optional<std::vector<int64_t>>, std::vector<int64_t>> PreTrainedTokenizerBase::truncate_sequences(
        std::vector<int64_t> ids,
        std::optional<std::vector<int64_t>> pair_ids,
        int64_t num_tokens_to_remove,
        Truncation tstrategy,
        int64_t stride)
    {
        if (num_tokens_to_remove <= 0)
        {
            std::tuple<  std::vector<int64_t>, std::optional<std::vector<int64_t>>, std::vector<int64_t>> ret = { ids, pair_ids, std::vector<int64_t>() };
            return ret;
        }

        TruncationStrategy truncation_strategy = *(tstrategy.as_strategy());

        std::vector<int64_t> overflowing_tokens;

        if (truncation_strategy == TruncationStrategy::ONLY_FIRST || (
            (truncation_strategy == TruncationStrategy::LONGEST_FIRST) && (!pair_ids)))
        {
            if (ids.size() > num_tokens_to_remove)
            {
                int64_t window_len = std::min((int64_t)ids.size(), stride + num_tokens_to_remove);
                if (_truncation_side == "left")
                {
                    //overflowing_tokens = ids[:window_len]
                    overflowing_tokens = vslice(ids, {}, window_len);
                    //ids = ids[num_tokens_to_remove:]
                    ids = vslice(ids, num_tokens_to_remove, {});
                }
                else if (_truncation_side == "right")
                {
                    //overflowing_tokens = ids[-window_len:]
                    overflowing_tokens = vslice(ids, -window_len, {});
                    //ids = ids[:-num_tokens_to_remove]
                    ids = vslice(ids, {}, -num_tokens_to_remove);
                }
                else
                {
                    throw std::invalid_argument("invalid truncation strategy :" + _truncation_side + ", use 'left' or 'right'.");
                }
            }
            else
            {
                std::string error_msg = "We need to remove " + std::to_string(num_tokens_to_remove) + "to truncate the input "
                    "but the first sequence has a length " + std::to_string(ids.size()) + ". ";

                if (truncation_strategy == TruncationStrategy::ONLY_FIRST)
                {
                    error_msg = error_msg + "Please select another truncation strategy than "
                        "TruncationStrategy::ONLY_FIRST, for instance 'longest_first' or 'only_second'.";
                }
                fprintf(stderr, "Error: %s\n", error_msg.c_str());
            }
        }
        else if (truncation_strategy == TruncationStrategy::LONGEST_FIRST)
        {
            fprintf(stderr, "Warning! Be aware, overflowing tokens are not returned for the setting you have chosen,"
                " i.e. sequence pairs with the 'TruncationStrategy::LONGEST_FIRST' "
                "truncation strategy. So the returned list will always be empty even if some "
                "tokens have been removed.\n");
            for (int64_t _ = 0; _ < num_tokens_to_remove; _++)
            {
                if (!pair_ids || ids.size() > pair_ids->size())
                {
                    if (_truncation_side == "right")
                    {
                        //ids = ids[:-1]
                        ids = vslice(ids, {}, -1);
                    }
                    else if (_truncation_side == "left")
                    {
                        //ids = ids[1:]
                        ids = vslice(ids, 1, {});
                    }
                    else
                    {
                        throw std::invalid_argument("invalid truncation strategy:" + _truncation_side);
                    }
                }
                else
                {
                    if (_truncation_side == "right")
                    {
                        //pair_ids = pair_ids[:-1]
                        pair_ids = vslice(*pair_ids, {}, -1);
                    }
                    else if (_truncation_side == "left")
                    {
                        //pair_ids = pair_ids[1:]
                        pair_ids = vslice(*pair_ids, 1, {});
                    }
                    else
                    {
                        throw std::invalid_argument("invalid truncation strategy:" + _truncation_side);
                    }
                }
            }
        }
        else if ((truncation_strategy == TruncationStrategy::ONLY_SECOND) && pair_ids)
        {
            if ((int64_t)pair_ids->size() > num_tokens_to_remove)
            {
                int64_t window_len = std::min((int64_t)pair_ids->size(), stride + num_tokens_to_remove);
                if (_truncation_side == "right ")
                {
                    //overflowing_tokens = pair_ids[-window_len:]
                    //pair_ids = pair_ids[:-num_tokens_to_remove]
                    overflowing_tokens = vslice(*pair_ids, -window_len, {});
                    pair_ids = vslice(*pair_ids, {}, -num_tokens_to_remove);
                }
                else if (_truncation_side == "left ")
                {
                    //overflowing_tokens = pair_ids[:window_len]
                    //pair_ids = pair_ids[num_tokens_to_remove:]
                    overflowing_tokens = vslice(*pair_ids, {}, window_len);
                    pair_ids = vslice(*pair_ids, num_tokens_to_remove, {});
                }
                else
                {
                    throw std::invalid_argument("invalid truncation strategy:" + _truncation_side);
                }
            }
            else
            {
                fprintf(stderr, "Error! We need to remove %lld to truncate the input "
                    "but the second sequence has a length %llu. "
                    "Please select another truncation strategy than ONLY_SECOND, "
                    "for instance 'longest_first' or 'only_first'.", num_tokens_to_remove, pair_ids->size());
            }
        }

        std::tuple<  std::vector<int64_t>, std::optional<std::vector<int64_t>>, std::vector<int64_t>> ret = { ids, pair_ids, overflowing_tokens };
        return ret;
    }

    std::tuple< PaddingStrategy, TruncationStrategy, std::optional<int64_t>> PreTrainedTokenizerBase::_get_padding_truncation_strategies(
        std::optional< Padding > padding,
        std::optional< Truncation > truncation,
        std::optional<int64_t> max_length,
        std::optional<int64_t> pad_to_multiple_of,
        bool verbose,
        get_padding_truncation_strategies_params kwargs)
    {
        TruncationStrategy old_truncation_strategy = kwargs.truncation_strategy;
        bool old_pad_to_max_length = kwargs.pad_to_max_length;

        // Backward compatibility for previous behavior, maybe we should deprecate it:
        // If you only set max_length, it activates truncation for max_length
        if (max_length && !padding && !truncation)
        {
            if (verbose)
            {
                fprintf(stderr, "Truncation was not explicitly activated but `max_length` is provided a specific value, please"
                    " use `truncation=True` to explicitly truncate examples to max length. Defaulting to"
                    " 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the"
                    " tokenizer you can select this strategy more precisely by providing a specific strategy to"
                    " `truncation`.\n");
            }

            truncation = TruncationStrategy::LONGEST_FIRST;
        }

        PaddingStrategy padding_strategy = PaddingStrategy::DO_NOT_PAD;

        // Get padding strategy
        if (padding && padding->is_false() && old_pad_to_max_length)
        {
            if (verbose)
            {
                fprintf(stderr, "The `pad_to_max_length` argument is deprecated and will be removed in a future version, "
                    "use `padding=True` or `padding='longest'` to pad to the longest sequence in the batch, or "
                    "use `padding='max_length'` to pad to a max length. In this case, you can give a specific "
                    "length with `max_length` (e.g. `max_length=45`) or leave max_length to None to pad to the "
                    "maximal input size of the model (e.g. 512 for Bert).\n");
            }
            if (!max_length)
                padding_strategy = PaddingStrategy::LONGEST;
            else
                padding_strategy = PaddingStrategy::MAX_LENGTH;
        }
        else if (!padding->is_false())
        {
            if (padding->is_true())
            {
                if (verbose)
                {
                    if (max_length && (!truncation || (truncation && *truncation == TruncationStrategy::DO_NOT_TRUNCATE)))
                    {
                        fprintf(stderr, "`max_length` is ignored when `padding`=`True` and there is no truncation strategy. "
                            "To pad to max length, use `padding='max_length'`.\n");
                    }
                    if (old_pad_to_max_length)
                    {
                        fprintf(stderr, "Though `pad_to_max_length` = `True`, it is ignored because `padding`=`True`.\n");
                    }
                    padding_strategy = PaddingStrategy::LONGEST;  // Default to pad to the longest sequence in the batch
                }
            }
            else if (padding->as_strategy())
            {
                padding_strategy = *padding->as_strategy();

            }
        }

        // Get truncation strategy
        TruncationStrategy truncation_strategy = TruncationStrategy::DO_NOT_TRUNCATE;
        if (!truncation && old_truncation_strategy != TruncationStrategy::DO_NOT_TRUNCATE)
        {
            if (verbose)
            {
                fprintf(stderr, "The `truncation_strategy` argument is deprecated and will be removed in a future version, use"
                    " `truncation=True` to truncate examples to a max length. You can give a specific length with"
                    " `max_length` (e.g. `max_length=45`) or leave max_length to None to truncate to the maximal input"
                    " size of the model (e.g. 512 for Bert).  If you have pairs of inputs, you can give a specific"
                    " truncation strategy selected among `truncation='only_first'` (will only truncate the first"
                    " sentence in the pairs) `truncation='only_second'` (will only truncate the second sentence in the"
                    " pairs) or `truncation='longest_first'` (will iteratively remove tokens from the longest sentence"
                    " in the pairs).\n");
            }
            truncation_strategy = old_truncation_strategy;
        }
        else if (truncation && !truncation->is_false())
        {
            if (truncation->is_true())
            {
                truncation_strategy = TruncationStrategy::LONGEST_FIRST;
            }
            else if (truncation->as_strategy())
            {
                truncation_strategy = *truncation->as_strategy();
            }

        }

        // Set max length if needed
        if (!max_length)
        {
            if (padding_strategy == PaddingStrategy::MAX_LENGTH)
            {
                //todo: *sort of* ported from python version, but in that case the constant was too big to even fit into a 64-bit int...
                // anyway, address this madness.
                std::cout << "_model_max_length = " << _model_max_length << std::endl;
                if (_model_max_length > (int64_t)10000000)
                {
                    if (verbose)
                    {
                        fprintf(stderr, "Asking to pad to max_length but no maximum length is provided and the model has no"
                            " predefined maximum length. Default to no padding.");
                    }
                    padding_strategy = PaddingStrategy::DO_NOT_PAD;
                }
                else
                {
                    max_length = _model_max_length;
                }
            }

            if (truncation_strategy != TruncationStrategy::DO_NOT_TRUNCATE)
            {
                // .. and here...
                if (_model_max_length > (int64_t)10000000)
                {
                    if (verbose)
                    {
                        fprintf(stderr, "Asking to truncate to max_length but no maximum length is provided and the model has"
                            " no predefined maximum length. Default to no truncation.");
                    }
                    truncation_strategy = TruncationStrategy::DO_NOT_TRUNCATE;
                }
                else
                {
                    max_length = _model_max_length;
                }
            }
        }

        // Test if we have a padding token
        if (padding_strategy != PaddingStrategy::DO_NOT_PAD && (!pad_token() || (pad_token_id() && *pad_token_id() < 0)))
        {
            throw std::invalid_argument("Asking to pad but the tokenizer does not have a padding token. "
                "Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)` "
                "or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`.");
        }

        // Check that we will truncate to a multiple of pad_to_multiple_of if both are provided
        if (truncation_strategy != TruncationStrategy::DO_NOT_TRUNCATE
            && padding_strategy != PaddingStrategy::DO_NOT_PAD
            && pad_to_multiple_of
            && max_length
            && (*max_length % *pad_to_multiple_of != 0))
        {
            throw std::invalid_argument("Truncation and padding are both activated but "
                "truncation length (" + std::to_string(*max_length) +
                ") is not a multiple of pad_to_multiple_of (" + std::to_string(*pad_to_multiple_of) + ").");
        }

        std::tuple< PaddingStrategy, TruncationStrategy, std::optional<int64_t>> ret = { padding_strategy, truncation_strategy, max_length };

        return ret;
    }
}