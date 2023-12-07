// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#pragma once
#include <cstddef>

namespace cpp_stable_diffusion_ov
{
    // Callback to inform on progress for a single unet loop.
    // Return true to continue processing.
    // Return false to cancel.
    typedef bool (*CallbackFuncUnetIteration)(size_t unet_step_i_complete,
        size_t num_unet_steps,
        void* user);
}