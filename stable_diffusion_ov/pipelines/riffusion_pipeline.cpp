#include <future>
#if USE_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#endif
#include "pipelines/riffusion_pipeline.h"
#include "tokenizers/clip_tokenizer.h"
#include "openvino_models/openvino_text_encoder.h"
#include "pipelines/unet_loop.h"
#include "pipelines/unet_loop_split.h"
#include "pipelines/unet_loop_sd15_internal_blobs.h"
#include "openvino_models/openvino_vae_decoder.h"
#include "openvino_models/openvino_vae_encoder.h"
#include "schedulers/pndm_scheduler.h"
#include "openvino_models/openvino_model_utils.h"
#include "audio_utils/spectrogram_image_converter.h"
#include "utils/rng.h"
#include "schedulers/scheduler_factory.h"
#include "openvino_models/model_collateral_cache.h"

//boo!
#include <torch/torch.h>

#ifdef WIN32
#define OS_SEP "\\"
#else
#define OS_SEP "/"
#endif

//todo remove this when we strip libtorch from this project
static void dump_tensor(torch::Tensor z, const char* fname)
{
    z = z.contiguous();
    std::ofstream wf(fname, std::ios::binary);
    wf.write((char*)z.data_ptr(), z.numel() * z.element_size());
    wf.close();
}

//todo: this is also defined in pndm_scheduler. Move it to common place
static std::vector<float> linspace(float start, float end, size_t steps)
{
    std::vector<float> res(steps);

    for (size_t i = 0; i < steps; i++)
    {
        res[i] = (float)(start + (steps - (steps - i)) * ((end - start) / (steps - 1)));
    }

    return res;
}


RiffusionPipeline::RiffusionPipeline(std::string model_folder,
    std::optional< std::string > cache,
    std::string text_encoder_device,
    std::string unet_positive_device,
    std::string unet_negative_device,
    std::string vae_decoder_device,
    std::string vae_encoder_device)
    : _model_folder(model_folder)
{
    //todo: In python, '77' comes from tokenizer config json. Figure out how to make
    // this more robust. Ideally want to avoid needing configuration
    // files to be carried around, but it just depends on how flexible
    // we need it to be. Is is static enough to be part of some compiled-in
    // factory? That would be ideal..
    _tok_max_length = 77;
    CLIPTokenizer::CLIPTokenizer_Params init;
    init.baseInit.baseInit.model_max_length = _tok_max_length;
    _tokenizer = std::make_shared< CLIPTokenizer >(init);

    std::string cache_dir;
    if (cache)
    {
        cache_dir = *cache;
    }
    else
    {
        //todo: set default cache, or just don't set cache dir at all?
        std::string cache_dir = "my_cache";
    }
    

#if 0
    std::string text_encoder_model_name = model_folder + OS_SEP + "text_encoder.xml";
    std::string vae_decoder_model_name = model_folder + OS_SEP + "vae_decoder.xml";
    std::string vae_encoder_model_name = model_folder + OS_SEP + "vae_encoder.xml";

    if (model_folder.find("riffusion-unet-quantized-int8") != std::string::npos)
    {
        auto unet_loop_split = std::make_shared< UNetLoopSD15InternalBlobs >(model_folder,
            unet_positive_device,
            unet_negative_device,
            _tok_max_length,
            _width,
            _height,
            cache_dir);
        _unet_loop = unet_loop_split;

        text_encoder_model_name = model_folder + OS_SEP + "text_encoder_fp16.xml";
        vae_decoder_model_name = model_folder + OS_SEP + "vae_decoder_fp16.xml";
        vae_encoder_model_name = model_folder + OS_SEP + "vae_encoder_org.xml";
    }
    else
    {
        auto unet_loop_split = std::make_shared< UNetLoopSplit >(model_folder + OS_SEP + "unet.xml",
            unet_positive_device,
            unet_negative_device,
            _tok_max_length,
            _width,
            _height,
            cache_dir);
        _unet_loop = unet_loop_split;
    }

    _vae_encoder = std::make_shared< OpenVINOVAEEncoder >(vae_encoder_model_name, vae_encoder_device, _tok_max_length, cache_dir);
    _vae_decoder = std::make_shared< OpenVINOVAEDecoder >(vae_decoder_model_name, vae_decoder_device, _tok_max_length, cache_dir);
    _text_encoder = std::make_shared< OpenVINOTextEncoder >(text_encoder_model_name, text_encoder_device, _tok_max_length, cache_dir);
#else

    auto m = ModelCollateralCache::instance()->GetModelCollateral(model_folder, cache_dir, text_encoder_device, unet_positive_device,
        unet_negative_device, vae_decoder_device, vae_encoder_device);

    _unet_loop = m.unet_loop;
    _vae_decoder = m.vae_decoder;
    _text_encoder = m.text_encoder;
    _vae_encoder = m.vae_encoder;

#endif
 
}

static bool run_img_to_wav_routine(std::shared_ptr<std::vector<float>> output, std::shared_ptr<std::vector<uint8_t>> image, SpectrogramImageConverter &converter, size_t chan)
{
    std::cout << "run_img_to_wav_routine->" << std::endl;
    auto wav = converter.audio_from_spectrogram_image(image, 512, 512, chan);
    output->insert(output->end(), wav->begin(), wav->end());
    std::cout << "<-run_img_to_wav_routine" << std::endl;

    return true;
}

std::pair< std::shared_ptr<std::vector<float>>, std::shared_ptr<std::vector<float>> > RiffusionPipeline::operator()(
    bool bStereo,
    const std::string prompt_start,
    std::optional< std::string > negative_prompt_start,
    std::optional< std::string > prompt_end,
    std::optional< std::string > negative_prompt_end,
    std::optional< unsigned int > seed_start,
    std::optional< unsigned int > seed_end,
    float denoising_start,
    float denoising_end,
    float guidance_scale_start,
    float guidance_scale_end,
    int num_inference_steps_per_sample,
    int num_interpolation_steps,
    std::string seed_image,
    float alpha_power,
    const std::string& scheduler_str,
    std::optional<std::pair< CallbackFuncUnetIteration, void*>> unet_iteration_callback,
    std::optional<std::pair< CallbackFuncInterpolationIteration, void*>> interp_iteration_callback
    )
{
    std::pair< std::shared_ptr<std::vector<float>>, std::shared_ptr<std::vector<float>> > output_pair;
    std::vector<float> alphas;
    if (num_interpolation_steps > 1)
    {
        alphas = linspace(0.f, 1.f, num_interpolation_steps);
    }
    else
    {
        alphas = { 0.f };
    }

    for (auto& a : alphas)
    {
        a = a * 2 - 1;
        //s = np.sign(a)
        float s;
        if (a > 0.)
            s = 1.f;
        else if (a < 0.f)
            s = -1.f;
        else
            s = 0.f;
        a = (std::powf(std::fabsf(a), alpha_power) * s + 1) / 2.f;
    }


    for (size_t i = 0; i < alphas.size(); i++)
    {
        std::cout << "alphas[" << i << "] = " << alphas[i] << std::endl;
    }

    ov::Tensor embed_start;
    _embed_text(prompt_start, embed_start);
    save_tensor_to_disk(embed_start, "embed_start.raw");

    ov::Tensor embed_end;
    _embed_text(*prompt_end, embed_end);
    //save_tensor_to_disk(embed_end, "embed_end.raw");

    ov::Tensor text_embedding = ov::Tensor(embed_start.get_element_type(),
        embed_start.get_shape());

    //image latents
    auto init_image_tensor = ov::Tensor(embed_start.get_element_type(),
        ov::Shape({ 1, 3, _height, _width }));

#if USE_OPENCV
    cv::Mat init_image = cv::imread(_model_folder + OS_SEP + seed_image + ".png", cv::IMREAD_COLOR);

    if ((init_image.cols != 512) || (init_image.rows != 512))
    {
        throw std::invalid_argument("init_image must be 512x512");
    }

    //preprocess (fill tensor with BGR image)
    {
        float* pTensorR = init_image_tensor.data<float>();
        float* pTensorG = pTensorR + _width * _height;
        float* pTensorB = pTensorG + _width * _height;
        uint8_t* pBGR = init_image.ptr();

        // convert BGR to NCHW tensor, applying mean / scale at the same time
        for (size_t p = 0; p < _width * _height; p++)
        {
            float B = (float)pBGR[p * 3 + 0] / 255.f;
            float G = (float)pBGR[p * 3 + 1] / 255.f;
            float R = (float)pBGR[p * 3 + 2] / 255.f;
            B = 2.f * B - 1.f;
            G = 2.f * G - 1.f;
            R = 2.f * R - 1.f;
            pTensorB[p] = B;
            pTensorG[p] = G;
            pTensorR[p] = R;
        }
    }

    //std::string fname = _model_folder + OS_SEP + seed_image + ".raw";
    //save_tensor_to_disk(init_image_tensor, fname);
#else
    std::string fname = _model_folder + OS_SEP + seed_image + ".raw";
    load_tensor_from_disk(init_image_tensor, fname);
#endif

    std::cout << "calling vae encoder" << std::endl;
    auto moments = (*_vae_encoder)(init_image_tensor);
    //save_tensor_to_disk(moments, "moments_ov.raw");
    std::cout << "vae encoder done" << std::endl;

    ov::Tensor mean, logvar;
    // Split moments into mean and var tensors. 
    {
        float* pMoments = moments.data<float>();
        ov::Shape tmp_shape = moments.get_shape();
        tmp_shape[1] /= 2;


       // Given that moments is NCHW [1, 4, 512, 512], we don't need to allocate a new buffer.
       // Just wrap 2 new tensors as existing ptr
        mean = ov::Tensor(moments.get_element_type(),
            tmp_shape, pMoments);
        logvar = ov::Tensor(moments.get_element_type(),
            tmp_shape, pMoments + mean.get_size());
    }

    //save_tensor_to_disk(mean, "mean_ov.raw");
    //save_tensor_to_disk(logvar, "logvar_ov.raw");

    
    //self.std = torch.exp(0.5 * self.logvar)
    {
        float* pStd = logvar.data<float>();
        for (size_t i = 0; i < logvar.get_size(); i++)
        {
            float l = pStd[i];

            //self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
            if (l < -30.f)
                l = -30.f;
            if (l > 20.f)
                l = 20.f;

            //self.std = torch.exp(0.5 * self.logvar)
            pStd[i] = std::expf(0.5f * l);
        }

        //save_tensor_to_disk(logvar, "std_ov.raw");
    }

    ov::Tensor init_latents;
    {
        torch::Generator generator= at::detail::createCPUGenerator();
        {
            std::lock_guard<std::mutex> lock(generator.mutex());
            generator.set_current_seed(*seed_start);
        }

        auto mean_shape = mean.get_shape();

        //DiagonalGaussianDistribution::sample()
        auto sample = torch::randn({ (int64_t)mean_shape[0],  (int64_t)mean_shape[1], (int64_t)mean_shape[2], (int64_t)mean_shape[3] }, generator);

        //dump_tensor(sample, "sample_ov.raw");
        sample = sample.contiguous();

        float* pSample = (float*)sample.data_ptr();
        float* pMean = mean.data<float>();
        float* pStd = logvar.data<float>();
        for (size_t i = 0; i < mean.get_size(); i++)
        {
            
            //x = self.mean + self.std * sample
            pMean[i] = pMean[i] + pStd[i] * pSample[i];

            //might as well do this in here too
            //init_latents = 0.18215 * init_latents
            pMean[i] *= 0.18215f;
        }

        //shallow copy
        init_latents = mean;
    }

    //save_tensor_to_disk(init_latents, "init_latents_ov.raw");

    //todo. Looks like riffusion supports inpainting, but doesn't enable it by default.
   // maybe someday we'll enable it? It'd be intertesting to use this feature to 'fix' 
   // artifacts within a 5 second audio snippet -- i.e. 'magic eraser' for audio.
   /*
   # Prepare mask latent
   mask: T.Optional[torch.Tensor] = None
   if mask_image:
       vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
       mask = preprocess_mask(mask_image, scale_factor=vae_scale_factor).to(
           device=self.device, dtype=embed_start.dtype
       )
   */

    ov::Tensor uncond_embeddings;
    {
        //todo: this can be exposed as a settable parameter. 
        // The way it will work, if we want to do what riffusion python
        // pipeline does, is use only 1 negative prompt. I think 
        // we *could* support a negative prompt for start & end prompt,
        // and interpolate between them in the same way as positive prompt?
        // but I'm not sure..
        std::optional<std::string> negative_prompt;
        if (!negative_prompt)
            negative_prompt = "";

        //invoke the text encoder, producing the negative prompt embeddings
        auto uncond_input = _tokenizer->call(
            *negative_prompt,  //text 
            {},      //text_pair
            {},      //text_target
            {},      //text_pair_target
            true,      //add_special_tokens 
            "max_length",  //padding
            true,         //truncation
            _tok_max_length);  //max_length

        auto uncond_input_ids = uncond_input["input_ids"];

        auto negative_prompt_embeds = (*_text_encoder)(uncond_input_ids);
        uncond_embeddings = ov::Tensor(negative_prompt_embeds.get_element_type(),
            negative_prompt_embeds.get_shape());
        negative_prompt_embeds.copy_to(uncond_embeddings);

        //save_tensor_to_disk(uncond_embeddings, "uncond_embeddings_ov.raw");
    }
    

    SpectrogramImageConverter converter;
    
    auto output_L = std::make_shared<std::vector<float>>();
    std::shared_ptr<std::vector<float>> output_R;
    if (bStereo)
    {
        output_R = std::make_shared<std::vector<float>>();
    }

    std::future<bool> last_iteration_img_to_wav_L;
    std::future<bool> last_iteration_img_to_wav_R;

    size_t interp_i = 0;
    for (auto alpha : alphas)
    {
        std::cout << "alpha = " << alpha << std::endl;
        float guidance_scale = guidance_scale_start * (1.f - alpha) + guidance_scale_end * alpha;

        std::cout << "guidance scale = " << guidance_scale << std::endl;

        // text_embedding = embed_start + alpha * (embed_end - embed_start)
        {
            float* pTextEmbedding = text_embedding.data<float>();
            float* pStart = embed_start.data<float>();
            float* pEnd = embed_end.data<float>();
            for (size_t i = 0; i < text_embedding.get_size(); i++)
            {
                pTextEmbedding[i] = pStart[i] + alpha * (pEnd[i] - pStart[i]);
            }
        }

        //save_tensor_to_disk(text_embedding, "text_embedding" + std::to_string(alpha) + "_ov_.raw");
        std::cout << "generating spectrogram image for alpha " << alpha << "..." << std::endl;
        auto image = _interpolate_img2img(text_embedding,
            init_latents,
            uncond_embeddings,
            seed_start,
            seed_end,
            alpha,
            denoising_start,
            denoising_end,
            num_inference_steps_per_sample,
            guidance_scale,
            scheduler_str,
            unet_iteration_callback);

        //cv::Mat outu8 = cv::Mat(512, 512, CV_8UC3, image->data());
        //cv::imwrite("img" + std::to_string(alpha) + ".png", outu8);

        if (interp_iteration_callback)
        {
            interp_iteration_callback->first(interp_i, alphas.size(), {}, image, _width, _height, interp_iteration_callback->second);
        }

        if (last_iteration_img_to_wav_L.valid())
            last_iteration_img_to_wav_L.wait();

        if (last_iteration_img_to_wav_R.valid())
            last_iteration_img_to_wav_R.wait();

        size_t chan_L = bStereo ? 1 : 0;
        last_iteration_img_to_wav_L = std::async(run_img_to_wav_routine, output_L, image, converter, chan_L);

        if (bStereo)
        {
            last_iteration_img_to_wav_R = std::async(run_img_to_wav_routine, output_R, image, converter, 2);
        }

        interp_i++;
    }

    if (last_iteration_img_to_wav_L.valid())
        last_iteration_img_to_wav_L.wait();

    if (last_iteration_img_to_wav_R.valid())
        last_iteration_img_to_wav_R.wait();

    output_pair.first = output_L;
    output_pair.second = output_R;
    return output_pair;
}

// will overwrite v0 with result.
static void slerp(float t, ov::Tensor& v0, ov::Tensor& v1, float dot_threshold = 0.9995f)
{
    std::cout << "slerp: t = " << t << std::endl;
    float norm_v0 = 0.f;
    float norm_v1 = 0.f;

    float* pV0 = v0.data<float>();
    float* pV1 = v1.data<float>();
    for (size_t i = 0; i < v0.get_size(); i++)
    {
        norm_v0 += pV0[i] * pV0[i];
        norm_v1 += pV1[i] * pV1[i];
    }

    norm_v0 = std::sqrtf(norm_v0);
    norm_v1 = std::sqrtf(norm_v1);

    std::cout << "norm_v0 = " << norm_v0 << std::endl;
    std::cout << "norm_v1 = " << norm_v1 << std::endl;

    float dot = 0.f;
    for (size_t i = 0; i < v0.get_size(); i++)
    {
        dot += pV0[i] * pV1[i] / (norm_v0 * norm_v1);
    }

    std::cout << "dot = " << dot << std::endl;
    if (std::fabsf(dot) > dot_threshold)
    {
        for (size_t i = 0; i < v0.get_size(); i++)
        {
            if (i == 0)
            {
                std::cout << "i = 0; v0 = " << pV0[i] << " v1 = " << pV1[i] << std::endl;
            }
            pV0[i] = (1.f - t) * pV0[i] + t * pV1[i];
        }
    }
    else
    {
        auto theta_0 = std::acosf(dot);
        std::cout << "theta_0 = " << theta_0 << std::endl;
        auto sin_theta_0 = std::sinf(theta_0);
        auto theta_t = theta_0 * t;
        auto sin_theta_t = std::sinf(theta_t);
        auto s0 = std::sinf(theta_0 - theta_t) / sin_theta_0;
        auto s1 = sin_theta_t / sin_theta_0;
        for (size_t i = 0; i < v0.get_size(); i++)
        {
            pV0[i] = s0 * pV0[i] + s1 * pV1[i];
        }
    }
}

std::shared_ptr<std::vector<uint8_t>> RiffusionPipeline::_interpolate_img2img(ov::Tensor& text_embedding,
    ov::Tensor init_latents,
    ov::Tensor& uncond_embeddings,
    std::optional< unsigned int > seed_start,
    std::optional< unsigned int > seed_end,
    float interpolate_alpha,
    float strength_a,
    float strength_b,
    int num_inference_steps,
    float guidance_scale,
    const std::string& scheduler_str,
    std::optional<std::pair< CallbackFuncUnetIteration, void*>> unet_iteration_callback
)
{
    auto scheduler = SchedulerFactory::Generate(scheduler_str);

    // set timesteps
    scheduler->set_timesteps(num_inference_steps);

    bool do_classifier_free_guidance = (guidance_scale > 1.f);

    float strength = (1.f - interpolate_alpha) * strength_a + interpolate_alpha * strength_b;
    auto offset = scheduler->get_steps_offset();
    size_t init_timestep = (size_t)(num_inference_steps * strength) + offset;
    init_timestep = std::min(init_timestep, (size_t)num_inference_steps);

    std::cout << "offset = " << offset << std::endl;
    std::cout << "num_inference_steps = " << num_inference_steps << std::endl;
    std::cout << "strength = " << strength << std::endl;
    std::cout << "init_timestep = " << init_timestep << std::endl;

    auto s_timesteps = scheduler->timesteps();
    double timesteps_i = s_timesteps[s_timesteps.size() - init_timestep];

    std::cout << "timesteps_i = " << timesteps_i << std::endl;

    ov::Tensor noise;
    //todo: there's no reason that these noise_a/noise_b tensors needs to be generated 'for each alpha'.
    // Generate them once, and re-use.
    // beware though,slerp function overwrites t0 with result. So that will also
    // need to change when this optimization is made.
    torch::Generator generator_start = at::detail::createCPUGenerator();
    {
        std::lock_guard<std::mutex> lock(generator_start.mutex());
        generator_start.set_current_seed(*seed_start);
    }

    torch::Generator generator_end = at::detail::createCPUGenerator();
    {
        std::lock_guard<std::mutex> lock(generator_end.mutex());
        generator_end.set_current_seed(*seed_end);
    }

    auto init_latents_shape = init_latents.get_shape();
    auto noise_a = torch::randn({ (int64_t)init_latents_shape[0],
        (int64_t)init_latents_shape[1],
        (int64_t)init_latents_shape[2],
        (int64_t)init_latents_shape[3] }, generator_start);
    noise_a = noise_a.contiguous();

    ov::Tensor noise_a_ov = ov::Tensor(init_latents.get_element_type(),
        init_latents_shape, (float*)noise_a.data_ptr());

    auto noise_b = torch::randn({ (int64_t)init_latents_shape[0],
        (int64_t)init_latents_shape[1],
        (int64_t)init_latents_shape[2],
        (int64_t)init_latents_shape[3] }, generator_end);
    noise_b = noise_b.contiguous();

    ov::Tensor noise_b_ov = ov::Tensor(init_latents.get_element_type(),
        init_latents_shape, (float*)noise_b.data_ptr());

    //save_tensor_to_disk(noise_a_ov, "noise_a_ov_.raw");
    //save_tensor_to_disk(noise_b_ov, "noise_b_ov_.raw");

    slerp(interpolate_alpha, noise_a_ov, noise_b_ov);
    noise = noise_a_ov;

    //save_tensor_to_disk(noise, "noise" + std::to_string(interpolate_alpha) + ".raw");
    //save_tensor_to_disk(init_latents, "init_latents_into_add_noise" + std::to_string(interpolate_alpha) + ".raw");

    //init_latents = self.scheduler.add_noise(init_latents, noise, timesteps)

    ov::Tensor latents;
    latents = scheduler->add_noise(init_latents, noise, timesteps_i);
    
    //ov::Tensor latents = init_latents;

    //save_tensor_to_disk(latents, "noisy_init_latents_ov_" + std::to_string(interpolate_alpha) + ".raw");

    // We don't do this part, as it seems like it's only used when mask != None.
    //init_latents_orig = init_latents

    auto t_start = std::max((size_t)num_inference_steps - init_timestep + offset, (size_t)0);
    auto timesteps_tmp = scheduler->timesteps();

    auto timesteps = std::vector<double>(timesteps_tmp.begin() + t_start, timesteps_tmp.end());

    latents = (*_unet_loop)(timesteps, latents, text_embedding, uncond_embeddings, guidance_scale, scheduler, unet_iteration_callback);

    //save_tensor_to_disk(latents, "unet_loop_output_ov_" + std::to_string(interpolate_alpha) + ".raw");
    //latents = 1 / 0.18215 * latents
    {
        auto* pLatents = latents.data<float>();
        for (size_t i = 0; i < latents.get_size(); i++)
        {
            pLatents[i] = 1.f / 0.18215f * pLatents[i];
        }
    }

    std::cout << "running vae decoder" << std::endl;
    auto image = (*_vae_decoder)(latents);
    std::cout << "running vae decoder done" << std::endl;
    //save_tensor_to_disk(image, "vae_decoder_out_ov.raw");

    //image = (image / 2 + 0.5).clamp(0, 1)
    {
        auto* pImage = image.data<float>();
        for (size_t i = 0; i < image.get_size(); i++)
        {
            pImage[i] = pImage[i] / 2.f + 0.5f;
            pImage[i] = pImage[i] > 1.f ? 1.f : pImage[i];
            pImage[i] = pImage[i] < 0 ? 0 : pImage[i];
        }
    }

    std::shared_ptr<std::vector<uint8_t>> image_buf_8u = std::make_shared<std::vector<uint8_t>>(_width * _height * 3);
    //convert to nchw, 8 bit
    float* pRF = image.data<float>();
    float* pGF = pRF + _width * _height;
    float* pBF = pGF + _width * _height;
    uint8_t* pImagebuf = image_buf_8u->data();
    bool bGiveBGR = false;

#if USE_OPENCV
    for (int i = 0; i < _width * _height; i++)
    {
        pImagebuf[i * 3 + 0] = (uint8_t)(pBF[i] * 255.f + 0.5f);
        pImagebuf[i * 3 + 1] = (uint8_t)(pGF[i] * 255.f + 0.5f);
        pImagebuf[i * 3 + 2] = (uint8_t)(pRF[i] * 255.f + 0.5f);
    }
    cv::Mat outu8 = cv::Mat(512, 512, CV_8UC3, image_buf_8u->data());
    cv::imwrite("spec_ov_" + std::to_string(interpolate_alpha) + ".png", outu8);
#endif
    

    for (int i = 0; i < _width * _height; i++)
    {
        pImagebuf[i * 3 + 0] = (uint8_t)(pRF[i] * 255.f + 0.5f);
        pImagebuf[i * 3 + 1] = (uint8_t)(pGF[i] * 255.f + 0.5f);
        pImagebuf[i * 3 + 2] = (uint8_t)(pBF[i] * 255.f + 0.5f);
    }
    
    //std::exit(0);
    

    return image_buf_8u;

}

void RiffusionPipeline::_embed_text(const std::string prompt,
    ov::Tensor& text_embeds)
{
    BatchEncoding text_inputs = _tokenizer->call(
        prompt,  //text 
        {},      //text_pair
        {},      //text_target
        {},      //text_pair_target
        true,      //add_special_tokens 
        "max_length",  //padding
        true,         //truncation
        _tok_max_length);  //max_length

    auto text_input_ids = text_inputs["input_ids"];

    //invoke the text encoder, producing the positive prompt embeddings
    auto positive_prompt_embeds = (*_text_encoder)(text_input_ids);
    text_embeds = ov::Tensor(positive_prompt_embeds.get_element_type(),
        positive_prompt_embeds.get_shape());
    positive_prompt_embeds.copy_to(text_embeds);
}


