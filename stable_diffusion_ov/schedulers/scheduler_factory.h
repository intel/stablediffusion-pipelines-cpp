// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#pragma once
#include <memory>
#include <string>
#include "cpp_stable_diffusion_ov/visibility.h"

namespace cpp_stable_diffusion_ov
{
    class Scheduler;
    class CPP_SD_OV_API SchedulerFactory
    {

    public:
        // valid sched_names = ["PNDMScheduler", "USTMScheduler"]
        static std::shared_ptr<Scheduler> Generate(const std::string& sched_name);
    };
}