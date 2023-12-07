// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#include "cpp_stable_diffusion_ov/clip_tokenizer.h"
#include "cpp_stable_diffusion_ov/tokenization_utils.h"

int main(int argc, char* argv[])
{
    try
    {
        int64_t tok_max_length = 77;
        cpp_stable_diffusion_ov::CLIPTokenizer::CLIPTokenizer_Params init;
        init.baseInit.baseInit.model_max_length = tok_max_length;
        auto tokenizer = std::make_shared< cpp_stable_diffusion_ov::CLIPTokenizer >(init);

        std::string prompt = "The quick brown fox jumped over the lazy, lazy dog!";

        cpp_stable_diffusion_ov::BatchEncoding text_inputs = tokenizer->call(
            prompt,  //text 
            {},      //text_pair
            {},      //text_target
            {},      //text_pair_target
            true,      //add_special_tokens 
            "max_length",  //padding
            true,         //truncation
            tok_max_length);  //max_length

        if (text_inputs.count("input_ids") != 1)
        {
            std::cout << "Error! Expected resulting text_inputs map to have an \"input_ids\" string entry" << std::endl;
            return 1;
        }

        auto text_input_ids = text_inputs["input_ids"];

        if (text_input_ids.empty())
        {
            std::cout << "Error! text_inputs[\"input_ids\"] is an empty vector!" << std::endl;
            return 1;
        }

        if (text_input_ids[0].size() != tok_max_length)
        {
            std::cout << "Error! text_inputs[\"input_ids\"][0].size() to be 77!" << std::endl;
            return 1;
        }

        const int64_t correct_tokens[] = { 49406, 518, 3712, 2866, 3240, 16901, 962, 518, 10753, 267, 10753, 1929, 256 };

        bool bOkay = true;
        for (size_t i = 0; i < text_input_ids[0].size(); i++)
        {
            if (i < sizeof(correct_tokens) / sizeof(correct_tokens[0]))
            {
                if (text_input_ids[0][i] != correct_tokens[i])
                {
                    std::cout << "Incorrect token at index " << i << std::endl;
                    std::cout << "Expected " << correct_tokens[i] << ", but got " << text_input_ids[0][i] << std::endl;
                    bOkay = false;
                    break;
                }
            }
            else
            {
                if (text_input_ids[0][i] != 49407)
                {
                    std::cout << "Incorrect pad token at index " << i << std::endl;
                    std::cout << "Expected 49407, but got " << text_input_ids[0][i] << std::endl;
                    bOkay = false;
                    break;
                }
            }
            
        }

        if (bOkay)
        {
            std::cout << "Correct results. Simple Tokenizer Test passes." << std::endl;
        }


    }
    catch (const std::exception& error) {
        std::cout << "in Simple Tokenization routine: exception: " << error.what() << std::endl;
    }


	return 0;
}