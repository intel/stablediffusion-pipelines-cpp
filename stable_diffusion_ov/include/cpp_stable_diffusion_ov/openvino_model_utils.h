// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#pragma once

#include <openvino/openvino.hpp>
#include <fstream>
#include <filesystem>

namespace cpp_stable_diffusion_ov
{

    static inline void logBasicModelInfo(const std::shared_ptr<ov::Model>& model) {
        std::cout << "Model name: " << model->get_friendly_name() << std::endl;

        // Dump information about model inputs/outputs
        ov::OutputVector inputs = model->inputs();
        ov::OutputVector outputs = model->outputs();

        std::cout << "\tInputs: " << std::endl;
        for (const ov::Output<ov::Node>& input : inputs) {
            const std::string name = input.get_any_name();
            const ov::element::Type type = input.get_element_type();
            const ov::PartialShape shape = input.get_partial_shape();
            const ov::Layout layout = ov::layout::get_layout(input);

            std::cout << "\t\t" << name << ", " << type << ", " << shape << ", " << layout.to_string() << std::endl;
        }

        std::cout << "\tOutputs: " << std::endl;
        for (const ov::Output<ov::Node>& output : outputs) {
            const std::string name = output.get_any_name();
            const ov::element::Type type = output.get_element_type();
            const ov::PartialShape shape = output.get_partial_shape();
            const ov::Layout layout = ov::layout::get_layout(output);

            std::cout << "\t\t" << name << ", " << type << ", " << shape << ", " << layout.to_string() << std::endl;
        }

        return;
    }

    static inline void save_tensor_to_disk(ov::Tensor& t, std::string filename)
    {
        std::ofstream wf(filename.c_str(), std::ios::out | std::ios::binary);
        if (!wf)
        {
            std::cout << "could not open file for writing" << std::endl;
            return;
        }

        size_t total_bytes = t.get_byte_size();
        float* pTData = t.data<float>();
        wf.write((char*)pTData, total_bytes);
        wf.close();
    }

    static inline void load_tensor_from_disk(ov::Tensor& t, std::string filename)
    {
        std::ifstream rf(filename.c_str(), std::ios::in | std::ios::binary);
        if (!rf)
        {
            throw std::invalid_argument("load_tensor_from_disk: Could not open " + filename + " for reading.");
            return;
        }

        auto fsize = std::filesystem::file_size(filename);
        if (fsize != t.get_byte_size())
        {
            throw std::invalid_argument("load_tensor_from_disk: Error! Was expecting " + filename + " to be "
                + std::to_string(t.get_byte_size()) + " bytes, but it is " + std::to_string(fsize) + " bytes.");
        }

        size_t total_bytes = t.get_byte_size();
        float* pTData = t.data<float>();
        rf.read((char*)pTData, total_bytes);
        rf.close();
    }
}