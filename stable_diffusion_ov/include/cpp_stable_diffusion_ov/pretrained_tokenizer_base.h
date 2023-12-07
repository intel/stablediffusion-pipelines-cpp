// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#pragma once

#include "cpp_stable_diffusion_ov/special_tokens_mixin.h"
#include "cpp_stable_diffusion_ov/tokenization_utils.h"
#include "cpp_stable_diffusion_ov/visibility.h"

namespace cpp_stable_diffusion_ov
{
    class CPP_SD_OV_API PreTrainedTokenizerBase : public SpecialTokensMixin
    {
    public:

        struct PreTrainedTokenizerBase_Params
        {
            SpecialTokensMixin_Params specialTokensInit;

            int64_t model_max_length = std::numeric_limits<int64_t>::max();
            std::string padding_side = "right"; // must be "right" or "left"
            std::string truncation_side = "right"; // must be "right" or "left"
            bool clean_up_tokenization_spaces = true;
            std::optional<std::string> processor_class;

        };


        PreTrainedTokenizerBase(PreTrainedTokenizerBase_Params init);

        // The maximum length of a sentence that can be fed to the model.
        int64_t max_len_single_sentence();

        // The maximum combined length of a pair of sentences that can be fed to the model.
        int64_t max_len_sentences_pair();

        void _set_processor_class(std::optional<std::string> processor_class);



        // Returns the vocabulary as a dictionary of token to index.
        //  `tokenizer.get_vocab()[token]` is equivalent to `tokenizer.convert_tokens_to_ids(token)` when `token` is in the
        // vocab.
        virtual std::unordered_map<std::string, int64_t> get_vocab() = 0;

        // Converts a string in a sequence of tokens, replacing unknown tokens with the `unk_token`.
        /*
            Args:
                text: The sequence to be encoded.
                pair: A second sequence to be encoded with the first.
                add_special_tokens: Whether or not to add the special tokens associated with the corresponding model.
                kwargs: Will be passed to the underlying model specific encode method. See details in


            Returns:
                `List[str]`: The list of tokens.
            """
        */
        //TODO: Figure out the kwargs thing
        virtual std::list< std::string > tokenize(std::string txt, std::optional<std::string> pair = {}, bool add_special_tokens = false) = 0;

        //call with single string
        BatchEncoding call(std::optional<std::string> text = {},
            std::optional<std::string> text_pair = {},
            std::optional<std::string> text_target = {},
            std::optional<std::string> text_pair_target = {},
            bool add_special_tokens = true,
            std::optional<Padding> padding = {},
            std::optional<Truncation> truncation = {},
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
            bool verbose = true);

        //call with list of strings
        BatchEncoding call(std::optional<std::list<std::string>> text = {},
            std::optional<std::list<std::string>> text_pair = {},
            std::optional<std::list<std::string>> text_target = {},
            std::optional<std::list<std::string>> text_pair_target = {},
            bool add_special_tokens = true,
            std::optional<Padding> padding = {},
            std::optional<Truncation> truncation = {},
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
            bool verbose = true);

    protected:

        BatchEncoding _call_one(std::optional<std::string> text = {},
            std::optional<std::string> text_pair = {},
            bool add_special_tokens = true,
            std::optional<Padding> padding = {},
            std::optional<Truncation> truncation = {},
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
            bool verbose = true);

        BatchEncoding _batch_call_one(std::optional<std::list<std::string>> text = {},
            std::optional<std::list<std::string>> text_pair = {},
            bool add_special_tokens = true,
            std::optional<Padding> padding = {},
            std::optional<Truncation> truncation = {},
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
            bool verbose = true);

        BatchEncoding encode_plus(std::optional<std::string> text = {},
            std::optional<std::string> text_pair = {},
            bool add_special_tokens = true,
            std::optional<Padding> padding = {},
            std::optional<Truncation> truncation = {},
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
            bool verbose = true);

        BatchEncoding batch_encode_plus(std::optional<std::list<std::string>> text = {},
            std::optional<std::list<std::string>> text_pair = {},
            bool add_special_tokens = true,
            std::optional<Padding> padding = {},
            std::optional<Truncation> truncation = {},
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
            bool verbose = true);


        //must be implemented in child class
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
            bool verbose = true) = 0;

        //must be implemented in child class
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
            bool verbose = true) = 0;

        BatchEncoding pad(BatchEncoding& encoded_inputs,
            std::optional<Padding> padding = {},
            std::optional<int64_t> max_length = {},
            std::optional<int64_t> pad_to_multiple_of = {},
            std::optional<bool> return_attention_mask = {},
            bool verbose = true);

        // Pad encoded inputs (on left/right and up to predefined length or max length in the batch)
        Batch1Encoding _pad(Batch1Encoding& encoded_inputs,
            std::optional<int64_t> max_length = {},
            PaddingStrategy padding_strategy = PaddingStrategy::DO_NOT_PAD,
            std::optional<int64_t> pad_to_multiple_of = {},
            std::optional<bool> return_attention_mask = {});

        //Create the token type IDs corresponding to the sequences passed.  
        // Should be overridden in a subclass if the model has a special way of building those.
        virtual std::vector< int64_t >  create_token_type_ids_from_sequences(std::vector<int64_t>& token_ids_0, std::optional<std::vector<int64_t>> token_ids_1 = {});

        //Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenatingand
        // adding special tokens.
        //
        // This implementation does not add special tokens and this method should be overridden in a subclass.
        std::vector<int64_t> virtual build_inputs_with_special_tokens(std::vector<int64_t>& token_ids_0, std::optional<std::vector<int64_t>> token_ids_1 = {});

        BatchEncoding prepare_for_model(std::vector<int64_t> ids,
            std::optional<std::vector<int64_t>> pair_ids = {},
            bool add_special_tokens = true,
            Padding padding = false,
            std::optional<Truncation> truncation = {},
            std::optional<int64_t> max_length = {},
            int64_t stride = 0,
            std::optional<int64_t> pad_to_multiple_of = {},
            std::optional<bool> return_token_type_ids = {},
            std::optional<bool> return_attention_mask = {},
            bool return_overflowing_tokens = false,
            bool return_special_tokens_mask = false,
            bool return_offsets_mapping = false,
            bool return_length = false,
            bool verbose = true,
            bool prepend_batch_axis = false);

        virtual std::vector< int64_t > get_special_tokens_mask(
            std::vector< int64_t>& token_ids_0, std::optional<std::vector<int64_t>> token_ids_1 = {}, bool already_has_special_tokens = false);


        // Truncates a sequence pair in-place following the strategy.
        std::tuple<  std::vector<int64_t>, std::optional<std::vector<int64_t>>, std::vector<int64_t>> truncate_sequences(
            std::vector<int64_t> ids,
            std::optional<std::vector<int64_t>> pair_ids = {},
            int64_t num_tokens_to_remove = 0,
            Truncation tstrategy = "longest_first",
            int64_t stride = 0);

        //pure virtual methods

        //Returns the number of added tokens when encoding a sequence with special tokens.
        virtual int64_t num_special_tokens_to_add(bool pair) = 0;

        //Find the correct padding/truncation strategy with backward compatibility for old arguments (truncation_strategy
        // and pad_to_max_length) and behaviors.

        struct get_padding_truncation_strategies_params
        {
            TruncationStrategy truncation_strategy = TruncationStrategy::DO_NOT_TRUNCATE;
            bool pad_to_max_length = false;
        };

        // returns { padding_strategy, truncation_strategy, max_length }
        std::tuple< PaddingStrategy, TruncationStrategy, std::optional<int64_t>> _get_padding_truncation_strategies(
            std::optional< Padding > padding,
            std::optional< Truncation > truncation = {},
            std::optional<int64_t> max_length = {},
            std::optional<int64_t> pad_to_multiple_of = {},
            bool verbose = true,
            get_padding_truncation_strategies_params kwargs = { TruncationStrategy::DO_NOT_TRUNCATE, false });


        // private method to put the tokenizer in input mode (when it has different modes for input/outputs)
        virtual void _switch_to_input_mode() {};

        // Private method to put the tokenizer in target mode (when it has different modes for input/outputs)
        virtual void _switch_to_target_mode() {};

        // Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.
        static void clean_up_tokenization(std::string& out_string)
        {
            out_string = std::regex_replace(out_string, std::regex(" ."), ".");
            out_string = std::regex_replace(out_string, std::regex(" ?"), "?");
            out_string = std::regex_replace(out_string, std::regex(" !"), "!");
            out_string = std::regex_replace(out_string, std::regex(" ,"), ",");
            out_string = std::regex_replace(out_string, std::regex(" ' "), "'");
            out_string = std::regex_replace(out_string, std::regex(" n't"), "n't");
            out_string = std::regex_replace(out_string, std::regex(" 'm"), "'m");
            out_string = std::regex_replace(out_string, std::regex(" 's"), "'s");
            out_string = std::regex_replace(out_string, std::regex(" 've"), "'ve");
            out_string = std::regex_replace(out_string, std::regex(" 're"), "'re");
        }


        //todo:
        /*
        def batch_decode(
            self,
            sequences: Union[List[int], List[List[int]], "np.ndarray", "torch.Tensor", "tf.Tensor"],
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: bool = None,
            **kwargs,
        ) -> List[str]:

        def decode(
            self,
            token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor", "tf.Tensor"],
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: bool = None,
            **kwargs,
        ) -> str

        def _decode(
            self,
            token_ids: Union[int, List[int]],
            skip_special_tokens: bool = False,
            clean_up_tokenization_spaces: bool = None,
            **kwargs,
        ) -> str:
            raise NotImplementedError


        */

        //int64_t 

        //class members from python class PreTrainedTokenizerBase
        bool _clean_up_tokenization_spaces = true;
        int64_t _model_max_length;

        bool _in_target_context_manager = false;

        // attributes from python class PreTrainedTokenizerBase
        std::unordered_map<std::string, std::string> _vocab_files_names = {};
        std::unordered_map<std::string, std::unordered_map<std::string, std::string>> _pretrained_vocab_files_map = {};

        //pretrained_init_configuration : Dict[str, Dict[str, Any]] = {} <-- figure this out
        std::unordered_map < std::string, std::unordered_map<std::string, std::optional<int64_t>>> _max_model_input_sizes = {};
        std::optional<std::string> _auto_class = {};

        // first name has to correspond to main model input name
        // to make sure `tokenizer.pad(...)` works correctly
        std::list< std::string > _model_input_names = { "input_ids", "token_type_ids", "attention_mask" };
        std::string _padding_side = "right";
        std::string _truncation_side = "right";
        std::optional<std::string> _slow_tokenizer_class = {};

        std::optional<std::string> _processor_class;

    };
}

