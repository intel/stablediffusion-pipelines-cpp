// Copyright(C) 2023 Intel Corporation
// SPDX - License - Identifier: Apache - 2.0
#pragma once
#include <random>
#include <time.h>
#include <optional>

namespace cpp_stable_diffusion_ov
{
    // numpy.random.rand (uniform distribution) equivelant
    class RNG
    {
    public:
        RNG(std::optional<unsigned int> seed = {})
        {
            if (seed)
            {
                _seed = *seed;
            }
            else
            {
                time_t t;
                _seed = (unsigned)time(&t);
            }

            std::cout << "RNG: seed = " << _seed << std::endl;
            _g = std::mt19937(_seed);
        }

        double gen()
        {
            int a = _g() >> 5;
            int b = _g() >> 6;
            double value = (a * 67108864.0 + b) / 9007199254740992.0;
            return value;
        }

        unsigned int seed() { return _seed; };

    private:
        std::mt19937 _g;
        unsigned int _seed;
    };

    // numpy.random.randn (gaussian distribution) equivelant
    class RNG_G
    {
    public:
        RNG_G(std::optional<unsigned int> seed = {})
        {
            if (seed)
            {
                _seed = *seed;
            }
            else
            {
                time_t t;
                _seed = (unsigned)time(&t);
            }

            std::cout << "RNG_G: seed = " << _seed << std::endl;
            _g = std::mt19937(_seed);
        }

        double gen()
        {
            if (_haveNextVal)
            {
                _haveNextVal = false;
                return _nextVal;
            }

            double f, x1, x2, r2;
            do {
                int a1 = _g() >> 5;
                int b1 = _g() >> 6;
                int a2 = _g() >> 5;
                int b2 = _g() >> 6;
                x1 = 2.0 * ((a1 * 67108864.0 + b1) / 9007199254740992.0) - 1.0;
                x2 = 2.0 * ((a2 * 67108864.0 + b2) / 9007199254740992.0) - 1.0;
                r2 = x1 * x1 + x2 * x2;
            } while (r2 >= 1.0 || r2 == 0.0);

            /* Box-Muller transform */
            f = sqrt(-2.0 * log(r2) / r2);
            _haveNextVal = true;
            _nextVal = f * x1;
            return f * x2;
        }

        unsigned int seed() { return _seed; };

    private:
        std::mt19937 _g;
        unsigned int _seed;
        bool _haveNextVal = false;
        double _nextVal = 0.;
    };
}