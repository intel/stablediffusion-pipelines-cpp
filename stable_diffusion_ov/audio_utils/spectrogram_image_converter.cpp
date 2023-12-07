// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#include <cmath>
#include <fstream>
#include <iostream>
#include "cpp_stable_diffusion_audio_ov/spectrogram_image_converter.h"
#include <torch/torch.h>

namespace cpp_stable_diffusion_ov
{

    static void dump_tensor(torch::Tensor z, const char* fname)
    {
        z = z.contiguous();
        std::ofstream wf(fname, std::ios::binary);
        wf.write((char*)z.data_ptr(), z.numel() * z.element_size());
        wf.close();
    }

    static inline void save_vector_to_disk(std::vector<uint8_t>& v, std::string filename)
    {
        std::ofstream wf(filename.c_str(), std::ios::out | std::ios::binary);
        if (!wf)
        {
            std::cout << "could not open file for writing" << std::endl;
            return;
        }

        size_t total_bytes = v.size() * sizeof(uint8_t);
        uint8_t* pTData = v.data();
        wf.write((char*)pTData, total_bytes);
        wf.close();
    }

    static inline void save_vector_to_disk(std::vector<float>& v, std::string filename)
    {
        std::ofstream wf(filename.c_str(), std::ios::out | std::ios::binary);
        if (!wf)
        {
            std::cout << "could not open file for writing" << std::endl;
            return;
        }

        size_t total_bytes = v.size() * sizeof(float);
        float* pTData = v.data();
        wf.write((char*)pTData, total_bytes);
        wf.close();
    }

    static float _hz_to_mel(float freq, std::string mel_scale = "htk")
    {
        if ((mel_scale != "slaney") && (mel_scale != "htk"))
        {
            throw std::invalid_argument("mel_scale should be one of \"htk\" or \"slaney\".");
        }

        if (mel_scale == "htk")
            return 2595.f * std::log10(1.f + (freq / 700.f));

        // Fill in the linear part
        auto f_min = 0.f;
        auto f_sp = 200.f / 3.f;

        auto mels = (freq - f_min) / f_sp;

        // fill in the log-scale part
        auto min_log_hz = 1000.f;
        auto min_log_mel = (min_log_hz - f_min) / f_sp;
        auto logstep = std::log(6.4f) / 27.f;

        if (freq >= min_log_hz)
        {
            mels = min_log_mel + std::log(freq / min_log_hz) / logstep;
        }

        return mels;
    }

    static torch::Tensor _mel_to_hz(torch::Tensor& mels, std::string mel_scale = "htk")
    {
        if ((mel_scale != "slaney") && (mel_scale != "htk"))
        {
            throw std::invalid_argument("mel_scale should be one of \"htk\" or \"slaney\".");
        }

        if (mel_scale == "htk")
        {
            return 700.f * (torch::pow(10.f, mels / 2595.0f) - 1.f);
        }

        //fill in the linear scale
        float f_min = 0.f;
        float f_sp = 200.f / 3.f;
        auto freqs = f_min + f_sp * mels;

        //and now the nonlinear scale
        float min_log_hz = 1000.f;
        auto  min_log_mel = (min_log_hz - f_min) / f_sp;
        auto logstep = std::log(6.4f) / 27.f;

        auto log_t = mels >= min_log_mel;
        freqs[log_t] = min_log_hz * torch::exp(logstep * (mels[log_t] - min_log_mel));

        return freqs;
    }

    static torch::Tensor _create_triangular_filterbank(
        torch::Tensor& all_freqs,
        torch::Tensor& f_pts
    )
    {
        // Adopted from torchaudio, Librosa
        // calculate the difference between each filter mid point and each stft freq point in hertz
        //f_diff = f_pts[1:] - f_pts[:-1]  # (n_filter + 1)
        auto f_diff = f_pts.index({ torch::indexing::Slice(1, torch::indexing::None) })
            - f_pts.index({ torch::indexing::Slice(torch::indexing::None, -1) });

        //slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_filter + 2)
        auto slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1);

        //create overlapping triangles
        auto zero = torch::zeros(1);

        //down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_filter)
        auto down_slopes = (-1.f * slopes.index({ torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(torch::indexing::None, -2) })) /
            f_diff.index({ torch::indexing::Slice(torch::indexing::None, -1) });

        //up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_filter)
        auto up_slopes = slopes.index({ torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(2, torch::indexing::None) }) /
            f_diff.index({ torch::indexing::Slice(1, torch::indexing::None) });

        //fb = torch.max(zero, torch.min(down_slopes, up_slopes))
        auto fb = torch::max(zero, torch::min(down_slopes, up_slopes));

        return fb;
    }



    class GriffinLim
    {
    public:
        GriffinLim(int n_fft = 400,
            int n_iter = 32,
            std::optional<int> win_length = {},
            std::optional<int> hop_length = {},
            float power = 2.f,
            float momentum = 0.99,
            c10::optional<int64_t> length = {},
            bool rand_init = true)
        {
            _n_fft = n_fft;
            _n_iter = n_iter;
            if (win_length)
                _win_length = *win_length;
            else
                _win_length = _n_fft;

            if (hop_length)
                _hop_length = *hop_length;
            else
                _hop_length = _win_length / 2;

            _window = torch::hann_window(_win_length);
            _length = length;
            _power = power;
            _momentum = momentum;
            _rand_init = rand_init;

            if (_rand_init)
            {
                _generator = at::detail::createCPUGenerator();
                _generator.set_current_seed(0);
            }
        }

        torch::Tensor operator()(torch::Tensor specgram)
        {
            if ((_momentum < 0) || (_momentum >= 1))
                throw std::invalid_argument("momentum must be in range [0, 1)");

            float momentum = _momentum / (1 + _momentum);

            auto shape = specgram.sizes();
            specgram = specgram.reshape({ -1, shape[1], shape[2] });

            specgram = specgram.pow(1 / _power);

            torch::Tensor angles;
            if (_rand_init)
            {
                auto rand_options = torch::TensorOptions().dtype(torch::kComplexFloat);
                angles = torch::rand(specgram.sizes(), _generator, rand_options);
                //dump_tensor(angles, "angles.raw");
            }
            else
            {
                auto full_options = torch::TensorOptions().dtype(torch::kComplexFloat);
                angles = torch::full(specgram.sizes(), 1, full_options);
            }

            // And initialize the previous iterate to 0
            auto tprev = torch::zeros(1, torch::TensorOptions().dtype(specgram.dtype()));

            int n_iter = _n_iter;
            for (int i = 0; i < n_iter; i++)
            {
                // Invert with our current estimate of the phases
                auto inverse = torch::istft(specgram * angles, _n_fft, _hop_length, _win_length, _window, true, false, {}, _length);

                // at::Tensor stft(const at::Tensor & self,
           //                  int64_t n_fft,
           //                  c10::optional<int64_t> hop_length = c10::nullopt,
           //                  c10::optional<int64_t> win_length = c10::nullopt,
           //                  const c10::optional<at::Tensor> &window = {},
           //                  bool center = true,
           //                  c10::string_view pad_mode = "reflect",
           //                  bool normalized = false,
           //                  c10::optional<bool> onesided = c10::nullopt,
           //                  c10::optional<bool> return_complex = c10::nullopt);
           //
                // Rebuild the spectrogram
                auto rebuilt = torch::stft(inverse,
                    _n_fft, _hop_length, _win_length, _window,
                    true, //center
                    "reflect",
                    false,
                    true,
                    true);

                // update our phase estimates
                angles = rebuilt;
                if (momentum)
                    angles = angles - tprev.mul_(momentum);
                angles = angles.div(angles.abs().add(1e-16f));

                // Store the previous iterate
                tprev = rebuilt;
            }

            // Return the final phase estimates
            auto waveform = torch::istft(specgram * angles, _n_fft, _hop_length, _win_length, _window, true, false, {}, _length);

            //unpack batch
            waveform = waveform.reshape({ shape[0], waveform.sizes().back() });

            return waveform;
        }

    private:

        int _n_fft;
        int _n_iter;
        int _win_length;
        int _hop_length;
        c10::optional<int64_t> _length;
        float _power;
        float _momentum;
        bool _rand_init;

        torch::Tensor _window;
        torch::Generator _generator;

    };

    static torch::Tensor melscale_fbanks(
        int n_freqs,
        float f_min,
        float f_max,
        int n_mels,
        int sample_rate,
        std::optional<std::string> norm = {},
        std::string mel_scale = "htk"
    )
    {
        if (norm && *norm != "slaney")
            throw std::invalid_argument("norm must be one of None or 'slaney'");

        // freq bins
        auto all_freqs = torch::linspace(0, sample_rate / 2, n_freqs);

        // calculate mel freq bins
        auto m_min = _hz_to_mel(f_min, mel_scale);
        auto m_max = _hz_to_mel(f_max, mel_scale);

        auto m_pts = torch::linspace(m_min, m_max, n_mels + 2);
        auto f_pts = _mel_to_hz(m_pts, mel_scale);

        // create filterbank
        //fb = _create_triangular_filterbank(all_freqs, f_pts)
        auto fb = _create_triangular_filterbank(all_freqs, f_pts);

        if (norm && *norm == "slaney")
        {
            //Slaney-style mel is scaled to be approx constant energy per channel
            // enorm = 2.0 / (f_pts[2:n_mels + 2] - f_pts[:n_mels])
            auto enorm = 2.f / (f_pts.index({ torch::indexing::Slice(2, torch::indexing::None, n_mels + 2) })
                - f_pts.index({ torch::indexing::Slice(torch::indexing::None, n_mels) }));
            fb *= enorm.unsqueeze(0);
        }

        return fb;
    }

    class MelScale
    {
    public:

        MelScale(int n_mels = 128,
            int sample_rate = 16000,
            float f_min = 0.f,
            std::optional<float> f_max = {},
            int n_stft = 201,
            std::optional<std::string> norm = {},
            std::string mel_scale = "htk")
        {
            _n_mels = n_mels;
            _sample_rate = sample_rate;
            _f_min = f_min;
            _f_max = (float)(_sample_rate / 2);
            if (f_max)
                _f_max = *f_max;

            _norm = norm;
            _mel_scale = mel_scale;

            if (_f_min > _f_max)
                throw std::invalid_argument("required _f_min =< _f_max");

            _fb = melscale_fbanks(n_stft, _f_min, _f_max, _n_mels, _sample_rate, _norm, _mel_scale);
        }

        torch::Tensor operator()(torch::Tensor& specgram)
        {
            auto mel_specgram = torch::matmul(specgram.transpose(-1, -2), _fb).transpose(-1, -2);

            return mel_specgram;
        }

    private:

        int _n_mels;
        int _sample_rate;
        float _f_min;
        float _f_max;
        std::optional<std::string> _norm;
        std::string _mel_scale;
        torch::Tensor _fb;
    };

    class InverseMelScale
    {

    public:
        InverseMelScale(int n_stft,
            int n_mels = 128,
            int sample_rate = 16000,
            float f_min = 0.f,
            std::optional<float> f_max = {},
            int max_iter = 100000,
            float tolerance_loss = 1e-5f,
            float tolerance_change = 1e-8,
            std::optional<std::string> norm = {},
            std::string mel_scale = "htk")
        {
            std::cout << "InverseMelScale()" << std::endl;
            _n_mels = n_mels;
            _sample_rate = sample_rate;
            if (f_max)
                _f_max = *f_max;
            else
                _f_max = (float)(sample_rate / 2);

            _f_min = f_min;
            _max_iter = max_iter;
            _tolerance_loss = tolerance_loss;
            _tolerance_change = tolerance_change;
            //self.sgdargs = sgdargs or {"lr": 0.1, "momentum": 0.9}
            if (_f_min > _f_max)
                throw std::invalid_argument("required _f_min =< _f_max");

            _fb = melscale_fbanks(n_stft, _f_min, _f_max, _n_mels, _sample_rate, norm, mel_scale);

            //dump_tensor(_fb, "fb.raw");
        }

        //new version (as of May 2023) in torchaudio. Version that uses lstsq
        torch::Tensor operator()(torch::Tensor& melspec_orig)
        {
            //dump_tensor(melspec_orig, "melspec.raw");

            // pack back
            auto shape = melspec_orig.sizes();

            auto n_mels = shape[shape.size() - 2];
            auto time = shape[shape.size() - 1];

            // melspec = melspec.view(-1, shape[-2], shape[-1])
            auto melspec = melspec_orig.view({ -1, n_mels, time });

            auto fb_shape = _fb.sizes();

            //freq, _ = self.fb.size()  # (freq, n_mels)
            auto freq = _fb.sizes()[0];

            if (_n_mels != n_mels)
                throw std::invalid_argument("Expected an input with " + std::to_string(_n_mels) +
                    " mel bins.Found: " + std::to_string(n_mels));

            auto fb_trans = _fb.transpose(-1, -2);
            fb_trans = fb_trans.unsqueeze(0);

            //dump_tensor(fb_trans, "fb_trans_into_lstsq.raw");
            //dump_tensor(melspec, "melspec_into_lstsq.raw");
            auto fb_lstsq_ret = torch::linalg::lstsq(fb_trans, melspec, {}, "gels");

            //fb_lstsq_ret = (solution, residuals, rank, singular_values)
            auto solution = std::get<0>(fb_lstsq_ret);
            //dump_tensor(solution, "solution.raw");

            auto specgram = torch::relu(solution);


            //unpack batch
            specgram = specgram.view({ shape[0], freq, time });

            //dump_tensor(specgram, "specgram.raw");

            return specgram;

        }


    private:

        float _n_mels;
        int _sample_rate;
        float _f_max;
        float _f_min;
        int _max_iter;
        float _tolerance_loss;
        float _tolerance_change;

        torch::Tensor _fb;

    };

    class Spectrogram
    {
    public:

        Spectrogram(int n_fft = 400,
            std::optional<int> win_length = {},
            std::optional<int> hop_length = {},
            int pad = 0,
            std::optional<float> power = 2.f,
            bool normalized = false,
            bool center = true,
            std::string pad_mode = "reflect",
            bool onesided = true)
        {
            _nfft = n_fft;
            if (win_length)
            {
                _win_length = *win_length;
            }
            else
            {
                _win_length = n_fft;
            }

            if (hop_length)
            {
                _hop_length = *hop_length;
            }
            else
            {
                _hop_length = _win_length / 2;
            }

            _window = torch::hann_window(_win_length);

            _pad = pad;
            _power = power;
            _normalized = normalized;
            _center = center;
            _pad_mode = pad_mode;
            _onesided = onesided;
        }

        torch::Tensor operator()(torch::Tensor waveform)
        {
            if (_pad > 0)
            {
                //waveform = torch.nn.functional.pad(waveform, (pad, pad), "constant")
                namespace F = torch::nn::functional;
                auto pad_options = F::PadFuncOptions({ _pad, _pad }).mode(torch::kConstant);
                waveform = F::pad(waveform, pad_options);
            }

            bool frame_length_norm = false;
            bool window_norm = _normalized;

            // pack batch
            auto shape = waveform.sizes();
            //waveform = waveform.reshape(-1, shape.back());

            //waveform = waveform.reshape(-1, shape[-1])
            waveform = waveform.reshape({ -1, shape.back() });

            std::cout << "Waveform shape after reshape =" << waveform.sizes() << std::endl;

            auto spec_f = torch::stft(waveform,
                _nfft, _hop_length, _win_length, _window, _center, _pad_mode, _normalized, _onesided, true);

            std::cout << "spec_f shape =" << spec_f.sizes() << std::endl;

            //unpack batch
            {
                auto shape = waveform.sizes().vec();
                auto spec_f_shape = spec_f.sizes().vec();

                std::vector<int64_t> new_shape;

                // Copy all elements except the last from shape
                new_shape.insert(new_shape.end(), shape.begin(), shape.end() - 1);

                // Copy the last two elements from spec_f_shape
                new_shape.insert(new_shape.end(), spec_f_shape.end() - 2, spec_f_shape.end());

                spec_f = spec_f.reshape(new_shape);
            }

            if (window_norm)
            {
                spec_f /= _window.pow(2.0).sum().sqrt();
            }

            if (_power)
            {
                if (*_power == 1.0)
                    return spec_f.abs();

                return spec_f.abs().pow(*_power);
            }

            return spec_f;
        }

    private:

        int _nfft;
        int _win_length;
        int _hop_length;
        torch::Tensor _window;
        int _pad;
        std::optional<float> _power;
        bool _normalized;
        bool _center;
        std::string _pad_mode;
        bool _onesided;


    };

    SpectrogramImageConverter::SpectrogramImageConverter(SpectrogramParams params)
        : _params(params)
    {
        _inverse_mel_scale = std::make_shared< InverseMelScale>(_params.n_fft() / 2 + 1,
            _params.num_frequencies,
            _params.sample_rate,
            _params.min_frequency,
            _params.max_frequency,
            _params.max_mel_iters,
            1e-5f,
            1e-8f,
            _params.mel_scale_norm,
            _params.mel_scale_type);

        c10::optional<int64_t> length = {};
        _griffin_lim = std::make_shared< GriffinLim >(_params.n_fft(),
            _params.num_griffin_lim_iters,
            _params.win_length(),
            _params.hop_length(),
            1.0,
            0.99f,
            length,
            true);

        std::optional<float> power = {};
        _spectrogram = std::make_shared< Spectrogram >(params.n_fft(),
            params.win_length(),
            params.hop_length(),
            0,
            power,
            false,
            true,
            "reflect",
            true);

        _mel_scale = std::make_shared< MelScale >(params.num_frequencies,
            params.sample_rate,
            params.min_frequency,
            params.max_frequency,
            params.n_fft() / 2 + 1,
            params.mel_scale_norm,
            params.mel_scale_type);

    }

    std::shared_ptr<std::vector<float>> SpectrogramImageConverter::_spec_from_image(std::shared_ptr<std::vector<uint8_t>> image_buf_8u,
        int image_width,
        int image_height,
        float max_value, size_t chan)
    {
        static int spec_save = 0;
        //save_vector_to_disk(*image_buf_8u, "image_before_spec_conv_ov" + std::to_string(spec_save) + ".raw");
        auto spectrogram = std::make_shared<std::vector<float>>(image_width * image_height);

        float power = _params.power_for_image;
        float stereo = _params.stereo;

        //so we're about to convert 8u, HWC images back into
        // f32 CHW... which is where we were at originally.
        // todo: If user doesn't want to view spectrogram images,
        // (or even if they do), we might want to pass f32 output
        // of unet loop (after clipping) directly here... would
        // save us a tiny bit of time but more importantly, would
        // preserve original precision.


        // so we want to combine the following operations into 1 loop
        // # Flip Y
        // image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        // # Munge channels into a numpy array of (channels, frequency, time)
        // data = np.array(image).transpose(2, 0, 1)
        // if stereo:
        //   # Take the Gand B channels as done in image_from_spectrogram
        //   data = data [[1, 2], :, : ]
        // else:
        //   data = data[0:1, : , : ]
        // # Convert to floats
        // data = data.astype(np.float32)
        // # Invert
        // data = 255 - data
        // # Rescale to 0 - 1
        // data = data / 255
        // # Reverse the power curve
        // data = np.power(data, 1 / power)
        // # Rescale to max value
        // data = data * max_value

        uint8_t* pImage8uBase = image_buf_8u->data();
        float* pSpecBase = spectrogram->data();
        for (int y = 0; y < image_height; y++)
        {
            int flipped_y = (image_height - 1) - y;
            uint8_t* pImage8u = pImage8uBase + flipped_y * image_width * 3;
            float* pSpec = pSpecBase + y * image_width;
            for (int x = 0; x < image_width; x++)
            {
                // Convert to floats
                float data = (float)pImage8u[x * 3 + chan];

                // Invert
                data = 255.f - data;

                // Rescale to 0-1
                data = data / 255.f;

                // Reverse the power curve
                data = std::pow(data, 1.f / power);

                // Rescale to max value
                data = data * max_value;

                pSpec[x] = data;
            }
        }

        //save_vector_to_disk(*spectrogram, "spec_ov" + std::to_string(spec_save) + ".raw");
        spec_save++;

        return spectrogram;
    }

    std::shared_ptr<std::vector<float>> SpectrogramImageConverter::audio_from_spectrogram_image(
        std::shared_ptr<std::vector<uint8_t>> image_buf_8u,
        int image_width,
        int image_height,
        size_t chan,
        bool apply_filters,
        float max_value)
    {

        if (_params.stereo)
            throw std::invalid_argument("stereo not supported yet");

        if (chan > 2)
        {
            throw std::invalid_argument("chan argument must be 0 (mono), 1(left), or 2(right)");
        }

        static int spec_save = 0;
        //save_vector_to_disk(*image_buf_8u, "image_before_spec_conv_ov" + std::to_string(spec_save) + ".raw");

        auto spectrogram = _spec_from_image(image_buf_8u, image_width, image_width, max_value, chan);

        //save_vector_to_disk(*spectrogram, "spectrogram_vec" + std::to_string(spec_save) + ".raw");

        return _audio_from_spectrogram(spectrogram);
    }



    std::shared_ptr<std::vector<float>> SpectrogramImageConverter::_audio_from_spectrogram(std::shared_ptr<std::vector<float>> spectrogram,
        bool apply_filters)
    {
        torch::Tensor amplitudes_mel = torch::from_blob(spectrogram->data(), { 1, 512, 512 });

        // Reconstruct the waveform
        // waveform = self.waveform_from_mel_amplitudes(amplitudes_mel)
        {
            // Convert from mel scale to linear
            // amplitudes_linear = self.inverse_mel_scaler(amplitudes_mel)
#if 0
            InverseMelScale inverse_mel_scaler(_params.n_fft() / 2 + 1,
                _params.num_frequencies,
                _params.sample_rate,
                _params.min_frequency,
                _params.max_frequency,
                _params.max_mel_iters,
                1e-5f,
                1e-8f,
                _params.mel_scale_norm,
                _params.mel_scale_type);

            auto amplitudes_linear = inverse_mel_scaler(amplitudes_mel);
#else
            auto amplitudes_linear = (*_inverse_mel_scale)(amplitudes_mel);
#endif

#if 0
            GriffinLim griffin_lim(_params.n_fft(),
                _params.num_griffin_lim_iters,
                _params.win_length(),
                _params.hop_length(),
                1.0,
                0.99f,
                {},
                true
            );

            auto waveform = griffin_lim(amplitudes_linear);
#else
            auto waveform = (*_griffin_lim)(amplitudes_linear);
#endif

            size_t num_elems = 1;
            auto shape = waveform.sizes();
            for (auto s : shape)
                num_elems *= s;

            std::shared_ptr<std::vector<float>> out_samples = std::make_shared<std::vector<float>>(num_elems);

            // convert to audio segment
            {
                waveform = waveform.contiguous();

                //normalize to -1, 1


                float max = -1.f;
                float* pWaveForm = (float*)waveform.data_ptr();
                for (size_t i = 0; i < num_elems; i++)
                {
                    if (std::fabs(pWaveForm[i]) > max)
                        max = std::fabs(pWaveForm[i]);
                }
                std::cout << "max = " << max << std::endl;

                if (max > 0.f)
                {
                    for (size_t i = 0; i < num_elems; i++)
                    {
                        pWaveForm[i] *= (1.f / max);
                    }
                }


                waveform = waveform.transpose(1, 0);
                waveform = waveform.contiguous();

                pWaveForm = (float*)waveform.data_ptr();
                float* pSamples = out_samples->data();
                for (size_t i = 0; i < num_elems; i++)
                {
                    pSamples[i] = pWaveForm[i];
                }
            }

            return out_samples;
        }
    }

    static torch::Tensor _wav_to_spectrogram(std::shared_ptr<std::vector<float>> audio_wav,
        SpectrogramParams& params,
        std::shared_ptr< Spectrogram > _spectrogram,
        std::shared_ptr< MelScale > _mel_scale)
    {
        torch::Tensor waveform_tensor = torch::from_blob(audio_wav->data(), { 1, (int64_t)audio_wav->size() });

        torch::Tensor spectrogram;
        auto spectrogram_complex = (*_spectrogram)(waveform_tensor);

        // Take the magnitude
        auto amplitudes = torch::abs(spectrogram_complex);

        spectrogram = (*_mel_scale)(amplitudes);

        float power = params.power_for_image;

        float max = torch::max(spectrogram).item<float>();

        // Rescale to 0 - 1
        spectrogram /= max;

        // apply the power curve
        spectrogram = torch::pow(spectrogram, power);

        // Rescale to 0-255
        spectrogram *= 255.f;

        // Invert
        spectrogram = 255.f - spectrogram;

        spectrogram = spectrogram.contiguous();

        return spectrogram;
    }

    SpectrogramImageConverter::Image SpectrogramImageConverter::spectrogram_image_from_audio(
        std::shared_ptr<std::vector<float>> audio_wav,
        std::shared_ptr<std::vector<float>> audio_wavR)
    {

        if (audio_wavR)
        {
            if (audio_wavR->size() != audio_wav->size())
            {
                throw std::invalid_argument("spectrogram_image_from_audio: audio_wavR length != audio_wav length");
            }
        }

#if 0

        //todo: handle number of channels, etc.
        torch::Tensor waveform_tensor = torch::from_blob(audio_wav->data(), { 1, (int64_t)audio_wav->size() });

        //dump_tensor(waveform_tensor, "waveform_tensor.raw");

        torch::Tensor spectrogram;
        //spectrogram_from_audio
        {
            auto spectrogram_complex = (*_spectrogram)(waveform_tensor);

            // Take the magnitude
            auto amplitudes = torch::abs(spectrogram_complex);

            spectrogram = (*_mel_scale)(amplitudes);
        }


        std::cout << "spectrogram shape = " << spectrogram.sizes() << std::endl;
        //image from spectrogram
        {
            float power = params.power_for_image;

            float max = torch::max(spectrogram).item<float>();
            std::cout << "max = " << max << std::endl;

            // Rescale to 0 - 1
            spectrogram /= max;

            // apply the power curve
            spectrogram = torch::pow(spectrogram, power);

            // Rescale to 0-255
            spectrogram *= 255.f;

            // Invert
            spectrogram = 255.f - spectrogram;

            spectrogram = spectrogram.contiguous();
        }
#else
        auto spectrogram = _wav_to_spectrogram(audio_wav, _params, _spectrogram, _mel_scale);
        torch::Tensor spectrogram_R;
        if (audio_wavR)
        {
            spectrogram_R = _wav_to_spectrogram(audio_wavR, _params, _spectrogram, _mel_scale);
        }
#endif
        size_t spec_height = spectrogram.sizes()[1];
        size_t spec_width = spectrogram.sizes()[2];

        Image img;
        std::shared_ptr<std::vector<uint8_t>> image = std::make_shared< std::vector<uint8_t> >(spec_height * spec_width * 3);
        img.image_buf = image;
        img.height = spec_height;
        img.width = spec_width;

        {
            //we're performing y-flip here, so start img ptr on last line.
            uint8_t* pImg = image->data() + ((spec_height - 1) * spec_width * 3);
            float* pSpecL = (float*)spectrogram.data_ptr();
            float* pSpecR;
            if (audio_wavR)
                pSpecR = (float*)spectrogram_R.data_ptr();
            else
                pSpecR = (float*)spectrogram.data_ptr();

            for (size_t y = 0; y < spec_height; y++)
            {
                for (size_t x = 0; x < spec_width; x++)
                {
                    pImg[x * 3 + 0] = (pSpecL[x] + pSpecR[x]) / 2.f + 0.5f;
                    pImg[x * 3 + 1] = pSpecL[x] + 0.5f;
                    pImg[x * 3 + 2] = pSpecR[x] + 0.5f;
                }

                //and each line it moves 'up' one line.
                pImg -= spec_width * 3;
                pSpecL += spec_width;
                pSpecR += spec_width;
            }
        }


        return img;
    }
}