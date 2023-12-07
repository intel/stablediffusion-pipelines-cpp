// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#pragma once

#include <openvino/openvino.hpp>
#include <string>
#include <optional>
#include "cpp_stable_diffusion_ov/visibility.h"

namespace cpp_stable_diffusion_ov
{

    class CPP_SD_OV_API OpenVINOTextEncoder
    {
    public:

        OpenVINOTextEncoder(std::string model_path, std::string device, size_t max_len, ov::Core &core);

        ov::Tensor operator()(std::vector<std::vector<int64_t>>& input_ids);

    private:

        ov::CompiledModel _compiled_model;
        ov::InferRequest _infer_request;

        std::string _out_name;
    };
}