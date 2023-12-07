// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#pragma once

#include <openvino/openvino.hpp>
#include <string>
#include <optional>
#include "cpp_stable_diffusion_ov/visibility.h"

namespace cpp_stable_diffusion_ov
{
    class CPP_SD_OV_API OpenVINOVAEEncoder
    {
    public:

        OpenVINOVAEEncoder(std::string model_path, std::string device, size_t max_len, ov::Core &core);

        ov::Tensor operator()(ov::Tensor& sample);

    private:

        ov::CompiledModel _compiled_model;
        ov::InferRequest _infer_request;


    };
}