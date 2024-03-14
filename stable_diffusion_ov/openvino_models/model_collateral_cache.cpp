// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#include "cpp_stable_diffusion_ov/model_collateral_cache.h"

#include "cpp_stable_diffusion_ov/openvino_text_encoder.h"
#include "pipelines/unet_loop.h"
#include "pipelines/unet_loop_split.h"
#include "pipelines/unet_loop_batch2.h"
#include "pipelines/unet_loop_sd15_internal_blobs.h"
#include "cpp_stable_diffusion_ov/openvino_vae_decoder.h"
#include "cpp_stable_diffusion_ov/openvino_vae_encoder.h"
#include "cpp_stable_diffusion_ov/openvino_model_utils.h"

#ifdef WIN32
#define OS_SEP "\\"
#else
#define OS_SEP "/"
#endif

namespace cpp_stable_diffusion_ov
{

    void ModelCollateralCache::Reset()
    {
        _text_encoder.reset();
        _unet_loop.reset();
        _vae_decoder.reset();
        _vae_encoder.reset();
        _model_folder = {};
        _txt_encoder_device = {};
        _unet_positive_device = {};
        _unet_negative_device = {};
        _vae_decoder_device = {};
        _vae_encoder_device = {};
    }

    ModelCollateralCache::ModelCollateral ModelCollateralCache::GetModelCollateral(std::string model_folder,
        std::optional<std::string> unet_sub_dir,
        std::optional<std::string> cache_dir,
        std::string text_encoder_device,
        std::string unet_positive_device,
        std::string unet_negative_device,
        std::string vae_decoder_device,
        std::string vae_encoder_device)
    {
        std::lock_guard<std::mutex> lck(_mutex);

        ov::Core core;

        if (cache_dir) {
            // enables caching of device-specific 'blobs' during core.compile_model
            // routine. This speeds up calls to compile_model for successive runs.
            core.set_property(ov::cache_dir(*cache_dir));
        }

        std::cout << "GetModelCollateral start." << std::endl;

        size_t tok_max_length = 77;

        if (_model_folder && (model_folder != *_model_folder))
        {
            std::cout << "Model folder changed. Resetting..." << std::endl;
            Reset();
            _model_folder = model_folder;
        }

        std::string text_encoder_model_name = model_folder + OS_SEP + "text_encoder.xml";
        std::string vae_decoder_model_name = model_folder + OS_SEP + "vae_decoder.xml";
        std::string vae_encoder_model_name = model_folder + OS_SEP + "vae_encoder.xml";

        //if the unet loop isn't set, or it's set but either unet+ / unet- devices have changed.
        if (!_unet_loop || !_unet_positive_device || !_unet_negative_device || (*_unet_positive_device != unet_positive_device) ||
            (*_unet_negative_device != unet_negative_device))
        {
            std::string unet_model_folder = model_folder;
            if (unet_sub_dir)
            {
                unet_model_folder = unet_model_folder + OS_SEP + *unet_sub_dir;
            }

            if ((unet_model_folder.find("int8") != std::string::npos) || (unet_model_folder.find("INT8") != std::string::npos))
            {
                std::cout << "Creating new int8 unet loop with +/- devices as: " << unet_positive_device << ", " << unet_negative_device << std::endl;
                _unet_loop = std::make_shared< UNetLoopSD15InternalBlobs >(unet_model_folder,
                    unet_positive_device,
                    unet_negative_device,
                    tok_max_length,
                    512,
                    512,
                    core);
            }
            else
            {
                auto model = core.read_model(unet_model_folder + OS_SEP + "unet.xml");

                logBasicModelInfo(model);

                auto latent_model_input_batch_size = model->input("latent_model_input").get_shape()[0];
                if (latent_model_input_batch_size == 2)
                {
                    if (unet_positive_device != unet_negative_device)
                    {
                        std::cout << "Warning! Batch2 model is being used, so we can't split across two separate devices. Will use single device = " << unet_positive_device << std::endl;
                    }

                    std::cout << "Creating new fp16 batch2 unet loop with device as:" << unet_positive_device <<  std::endl;

                    _unet_loop = std::make_shared< UNetLoopBatch2 >(model,
                        unet_positive_device,
                        tok_max_length,
                        512,
                        512,
                        core);
                }
                else if(latent_model_input_batch_size == 1)
                {
                    std::cout << "Creating new fp16 split unet loop with +/- devices as:" << unet_positive_device << ", " << unet_negative_device << std::endl;
                    _unet_loop = std::make_shared< UNetLoopSplit >(model,
                        unet_positive_device,
                        unet_negative_device,
                        tok_max_length,
                        512,
                        512,
                        core);
                }
                else
                {
                    throw std::runtime_error("Expected unet model to have batch size 1 or 2, but it has batch size of " + std::to_string(latent_model_input_batch_size));
                }
            }

            _unet_positive_device = unet_positive_device;
            _unet_negative_device = unet_negative_device;
        }


        if (!_text_encoder || !_txt_encoder_device || (*_txt_encoder_device != text_encoder_device))
        {
            std::cout << "Creating new text encoder with device as :" << text_encoder_device << std::endl;
            _text_encoder = std::make_shared< OpenVINOTextEncoder >(text_encoder_model_name, text_encoder_device, tok_max_length, core);
            _txt_encoder_device = text_encoder_device;
        }
        else
        {
            std::cout << "Using cached text encoder with device as :" << text_encoder_device << std::endl;
        }

        if (!_vae_decoder || !_vae_decoder_device || (*_vae_decoder_device != vae_decoder_device))
        {
            std::cout << "Creating new vae decoder with device as :" << vae_decoder_device << std::endl;
            _vae_decoder = std::make_shared< OpenVINOVAEDecoder >(vae_decoder_model_name, vae_decoder_device, tok_max_length, core);
            _vae_decoder_device = vae_decoder_device;
        }
        else
        {
            std::cout << "Using cached vae decoder with device as :" << vae_decoder_device << std::endl;
        }

        if (!_vae_encoder || !_vae_encoder_device || (*_vae_encoder_device != vae_encoder_device))
        {
            std::cout << "Creating new vae encoder with device as :" << vae_encoder_device << std::endl;
            _vae_encoder = std::make_shared< OpenVINOVAEEncoder >(vae_encoder_model_name, vae_encoder_device, tok_max_length, core);
            _vae_encoder_device = vae_encoder_device;
        }
        else
        {
            std::cout << "Using cached vae encoder with device as :" << vae_encoder_device << std::endl;
        }

        ModelCollateral m;
        m.text_encoder = _text_encoder;
        m.unet_loop = _unet_loop;
        m.vae_decoder = _vae_decoder;
        m.vae_encoder = _vae_encoder;

        std::cout << "GetModelCollateral end." << std::endl;

        return m;
    }
}