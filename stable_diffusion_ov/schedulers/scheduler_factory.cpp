// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#include <stdexcept>
#include "schedulers/scheduler_factory.h"
#include "schedulers/pndm_scheduler.h"
#include "schedulers/ustm_scheduler.h"
#include "schedulers/euler_discrete_scheduler.h"

namespace cpp_stable_diffusion_ov
{
    std::shared_ptr<Scheduler> SchedulerFactory::Generate(const std::string& sched_name)
    {
        if (sched_name == "PNDMScheduler")
        {
            std::optional< std::vector<float> > trained_betas;
            auto scheduler = std::make_shared< PNDMScheduler>(
                1000, //num_train_timesteps
                0.00085f, //beta_start
                0.012f, //beta_end
                "scaled_linear", //beta_schedule
                trained_betas,
                true, //skip_prk_steps
                false, //set_alpha_to_one
                "epsilon",
                1); //steps_offset

            return scheduler;
        }
        else if (sched_name == "USTMScheduler")
        {
            std::optional< std::vector<float> > trained_betas;

            auto scheduler = std::make_shared< USTMScheduler>(
                1000, //num_train_timesteps
                0.00085f, //beta_start
                0.012f, //beta_end
                "scaled_linear", //beta_schedule
                trained_betas,
                false, //set_alpha_to_one
                "epsilon",
                1); //steps_offset

            return scheduler;
        }
        else if (sched_name == "EulerDiscreteScheduler")
        {
            std::optional< std::vector<float> > trained_betas;

            auto scheduler = std::make_shared< EulerDiscreteScheduler >(
                1000, //num_train_timesteps
                0.00085f, //beta_start
                0.012f, //beta_end
                "scaled_linear", //beta_schedule
                trained_betas,
                "epsilon",
                "linear",
                false,
                "linspace",
                1); //steps_offset

            return scheduler;
        }
        else
        {
            throw std::invalid_argument("Invalid scheduler name. It must be \"EulerDiscreteScheduler\", \"PNDMScheduler\", or \"USTMScheduler\"");
        }
    }
}