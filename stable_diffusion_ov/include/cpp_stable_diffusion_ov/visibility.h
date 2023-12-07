// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#pragma once

#if defined(_WIN32) || defined(__CYGWIN__)
#    define CPP_SD_OV_CORE_IMPORTS __declspec(dllimport)
#    define CPP_SD_OV_CORE_EXPORTS __declspec(dllexport)
#    define _CPP_SD_OV_HIDDEN_METHOD
#elif defined(__GNUC__) && (__GNUC__ >= 4) || defined(__clang__)
#    define CPP_SD_OV_CORE_IMPORTS   __attribute__((visibility("default")))
#    define CPP_SD_OV_CORE_EXPORTS   __attribute__((visibility("default")))
#    define _CPP_SD_OV_HIDDEN_METHOD __attribute__((visibility("hidden")))
#else
#    define CPP_SD_OV_CORE_IMPORTS
#    define CPP_SD_OV_CORE_EXPORTS
#    define _CPP_SD_OV_HIDDEN_METHOD
#endif

#ifdef _MSC_VER
#    pragma warning(disable : 4251)
#    pragma warning(disable : 4275)
#endif

#ifdef CPP_SD_OV_STATIC_LIBRARY  // defined if we are building or calling OpenVINO as a static library
#    define CPP_SD_OV_API
#else
#    ifdef IMPLEMENT_CPP_SD_OV_API  // defined if we are building the C++ Stable Diffusion DLL's (instead of using them)
#        define CPP_SD_OV_API        CPP_SD_OV_CORE_EXPORTS
#    else
#        define CPP_SD_OV_API        CPP_SD_OV_CORE_IMPORTS
#    endif  // IMPLEMENT_CPP_SD_OV_API
#endif      // CPP_SD_OV_STATIC_LIBRARY