// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#include "cpp_stable_diffusion_ov/openvino_unet.h"
#include "cpp_stable_diffusion_ov/openvino_model_utils.h"

namespace cpp_stable_diffusion_ov
{
    OpenVINOUNet::OpenVINOUNet(std::string model_path, std::string device, size_t max_len, std::optional<std::string> cache_dir)
    {
        ov::Core core;

        if (cache_dir) {
            // enables caching of device-specific 'blobs' during core.compile_model
            // routine. This speeds up calls to compile_model for successive runs.
            core.set_property(ov::cache_dir(*cache_dir));
        }

        //Read the OpenVINO encoder IR (.xml/.bin) from disk, producing an ov::Model object.
        _model = core.read_model(model_path);

        _sample_shape = ov::Shape{ 1, 4, 64, 64 };
        _encoder_hidden_states_shape = ov::Shape{ 1, max_len, 768 };
        std::map<std::string, ov::PartialShape> reshape_map;
        reshape_map["sample"] = _sample_shape;
        reshape_map["encoder_hidden_states"] = _encoder_hidden_states_shape;
        _model->reshape(reshape_map);

        logBasicModelInfo(_model);

        // Produce a compiled-model object, given the device ("CPU", "GPU", etc.)
        std::cout << "Compiling unet model for " << device << "..." << std::endl;
        _compiled_model = core.compile_model(_model, device);
        std::cout << "Compiling unet model for " << device << " done" << std::endl;

        _infer_request = _compiled_model.create_infer_request();
    }

    ov::Tensor OpenVINOUNet::operator()(ov::Tensor& sample, int64_t timestep, ov::Tensor& encoder_hidden_states)
    {
        if (sample.get_shape() != _sample_shape)
        {
            throw std::invalid_argument("invalid sample shape");
        }

        if (encoder_hidden_states.get_shape() != _encoder_hidden_states_shape)
        {
            throw std::invalid_argument("invalid encoder_hidden_states shape");
        }

        auto timestep_tensor = _infer_request.get_tensor("timestep");
        auto* pTimestep = timestep_tensor.data<int64_t>();
        *pTimestep = timestep;

        _infer_request.set_tensor("sample", sample);
        _infer_request.set_tensor("encoder_hidden_states", encoder_hidden_states);

        //save_tensor_to_disk(sample, "sample.raw");
        //save_tensor_to_disk(encoder_hidden_states, "encoder_hidden_states.raw");

        _infer_request.infer();

        ov::Tensor out_tensor = _infer_request.get_tensor("out_sample");

        //std::cout << "out_tensor ptr = " << (void *)out_tensor.data<float>() << std::endl;
        //save_tensor_to_disk(out_tensor, "out_tensor.raw");

        return out_tensor;
    }
}