// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#include "cpp_stable_diffusion_ov/openvino_vae_encoder.h"
#include "cpp_stable_diffusion_ov/openvino_model_utils.h"

namespace cpp_stable_diffusion_ov
{
    OpenVINOVAEEncoder::OpenVINOVAEEncoder(std::string model_path, std::string device, size_t max_len, ov::Core& core)
    {
        std::cout << "Compiling vae encoder model for " << device << "..." << std::endl;
        _compiled_model = core.compile_model(model_path, device);
        std::cout << "Compiling vae encoder model for " << device << " done" << std::endl;

        _infer_request = _compiled_model.create_infer_request();
    }

    ov::Tensor OpenVINOVAEEncoder::operator()(ov::Tensor& sample)
    {
        _infer_request.set_input_tensor(sample);

        _infer_request.infer();

        ov::Tensor out_tensor = _infer_request.get_output_tensor();

        return out_tensor;
    }
}