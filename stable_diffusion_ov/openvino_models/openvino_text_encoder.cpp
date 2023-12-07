// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#include "cpp_stable_diffusion_ov/openvino_text_encoder.h"
#include "cpp_stable_diffusion_ov/openvino_model_utils.h"

namespace cpp_stable_diffusion_ov
{
    OpenVINOTextEncoder::OpenVINOTextEncoder(std::string model_path, std::string device, size_t max_len, ov::Core& core)
    {
        std::cout << "Compiling text encoder model for " << device << "..." << std::endl;
        _compiled_model = core.compile_model(model_path, device);
        std::cout << "Compiling text encoder model for " << device << " done" << std::endl;

        auto outputs = _compiled_model.outputs();

        for (auto& out : outputs) {
            auto outShape = out.get_shape();

            if (outShape == std::vector<size_t>{1, max_len, 768})
            {
                _out_name = out.get_any_name();
            }
        }

        if (_out_name.empty())
        {
            throw std::invalid_argument("Did not find output tensor of size {1, max_len, 768}");
        }

        _infer_request = _compiled_model.create_infer_request();
    }

    ov::Tensor OpenVINOTextEncoder::operator()(std::vector<std::vector<int64_t>>& input_ids)
    {
        const ov::Tensor& in_tensor = _infer_request.get_input_tensor();

        auto num_ids = in_tensor.get_shape()[1];
        switch (in_tensor.get_element_type())
        {
        case ov::element::i32:
        {
            int32_t* pTensor = in_tensor.data<int32_t>();
            for (int i = 0; i < num_ids; i++)
            {
                pTensor[i] = (int32_t)(input_ids[0][i]);
            }
            break;
        }

        case ov::element::i64:
        {
            int64_t* pTensor = in_tensor.data<int64_t>();
            for (int i = 0; i < num_ids; i++)
            {
                pTensor[i] = (int64_t)(input_ids[0][i]);
            }
            break;
        }

        }


        std::cout << "in_tensor shape 1 = " << num_ids << std::endl;


        _infer_request.infer();

        ov::Tensor out_tensor = _infer_request.get_tensor(_out_name);

        return out_tensor;
    }
}