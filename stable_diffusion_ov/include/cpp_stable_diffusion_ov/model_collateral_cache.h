// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#pragma once
#include <mutex>
#include <optional>
#include <memory>
#include <string>
#include "cpp_stable_diffusion_ov/visibility.h"

namespace cpp_stable_diffusion_ov
{
    class OpenVINOTextEncoder;
    class UNetLoop;
    class OpenVINOVAEDecoder;
    class OpenVINOVAEEncoder;

    class CPP_SD_OV_API ModelCollateralCache
    {
    public:
        ModelCollateralCache(ModelCollateralCache const&) = delete;
        ModelCollateralCache& operator=(ModelCollateralCache const&) = delete;
        ~ModelCollateralCache() {}

        struct ModelCollateral
        {
            std::shared_ptr< OpenVINOTextEncoder > text_encoder;
            std::shared_ptr< UNetLoop > unet_loop;
            std::shared_ptr< OpenVINOVAEDecoder > vae_decoder;
            std::shared_ptr< OpenVINOVAEEncoder > vae_encoder;
        };

        static ModelCollateralCache* instance()
        {
            static ModelCollateralCache instance{};
            return &instance;
        }

        ModelCollateral GetModelCollateral(std::string model_folder,
            std::optional<std::string> unet_sub_dir,
            std::optional<std::string> cache_dir,
            std::string text_encoder_device,
            std::string unet_positive_device,
            std::string unet_negative_device,
            std::string vae_decoder_device,
            std::string vae_encoder_device);

        std::optional<std::string> CurrentTextEncoderDevice() { return _txt_encoder_device; };
        std::optional<std::string> CurrentUNetPositiveDevice() { return _unet_positive_device; };
        std::optional<std::string> CurrentUNetNegativeDevice() { return _unet_negative_device; };
        std::optional<std::string> CurrentVAEDecoderDevice() { return _vae_decoder_device; };
        std::optional<std::string> CurrentVAEEncoderDevice() { return _vae_encoder_device; };

        void Reset();

    private:

        explicit ModelCollateralCache() {};
        std::mutex _mutex;

        std::optional<std::string> _txt_encoder_device;
        std::optional<std::string> _unet_positive_device;
        std::optional<std::string> _unet_negative_device;
        std::optional<std::string> _vae_decoder_device;
        std::optional<std::string> _vae_encoder_device;
        std::optional<std::string> _model_folder;


        std::shared_ptr< OpenVINOTextEncoder > _text_encoder;
        std::shared_ptr< UNetLoop > _unet_loop;
        std::shared_ptr< OpenVINOVAEDecoder > _vae_decoder;
        std::shared_ptr< OpenVINOVAEEncoder > _vae_encoder;

        //std::shared_ptr< ov::Core > _core;
    };
}
