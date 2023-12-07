// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#pragma once

#include <openvino/openvino.hpp>
#include <string>
#include <optional>
#include "cpp_stable_diffusion_ov/visibility.h"

namespace cpp_stable_diffusion_ov
{
    class CPP_SD_OV_API OpenVINOUNet
    {
    public:

        OpenVINOUNet(std::string model_path, std::string device, size_t max_len, std::optional<std::string> cache_dir = {});

        ov::Tensor operator()(ov::Tensor& sample, int64_t timestep, ov::Tensor& encoder_hidden_states);

    private:

        std::shared_ptr<ov::Model> _model;
        ov::CompiledModel _compiled_model;
        ov::InferRequest _infer_request;

        ov::Shape _sample_shape;
        ov::Shape _encoder_hidden_states_shape;

    };
}