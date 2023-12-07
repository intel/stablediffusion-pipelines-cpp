// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#pragma once
#include "cpp_stable_diffusion_ov/visibility.h"

#ifdef CPP_SD_OV_AUDIO_STATIC_LIBRARY  
#    define CPP_SD_OV_API
#else
#    ifdef IMPLEMENT_CPP_SD_AUDIO_OV_API  // defined if we are building the C++ Stable Diffusion DLL's (instead of using them)
#        define CPP_SD_OV_AUDIO_API        CPP_SD_OV_CORE_EXPORTS
#    else
#        define CPP_SD_OV_AUDIO_API        CPP_SD_OV_CORE_IMPORTS
#    endif  // IMPLEMENT_CPP_SD_AUDIO_OV_API
#endif      // CPP_SD_OV_STATIC_LIBRARY