#pragma GCC diagnostic ignored "-Wunused-result"

#ifndef _COMMON_H_
#define _COMMON_H_

#define flag { printf("\nLINE: %d\n", __LINE__); fflush(stdout); }

#include <vector>
#include <cmath>
#include <pmmintrin.h>

struct Problem
{
    Problem(uint32_t const nr_instance, uint32_t const nr_field) 
        : nr_feature(0), nr_instance(nr_instance), nr_field(nr_field), 
          v(2.0f/static_cast<float>(nr_field)), 
          J(static_cast<uint64_t>(nr_instance)*nr_field), 
          Y(nr_instance) {}
    uint32_t nr_feature, nr_instance, nr_field;
    float v;
    std::vector<uint32_t> J;
    std::vector<float> Y;
};

Problem read_problem(std::string const path);

uint32_t const kW_NODE_SIZE = 2;

struct Model
{
    Model(uint32_t const nr_feature, uint32_t const nr_factor, uint32_t const nr_field) 
        : W(static_cast<uint64_t>(nr_feature)*nr_field*nr_factor*kW_NODE_SIZE, 0), 
          nr_feature(nr_feature), nr_factor(nr_factor), nr_field(nr_field) {}
    std::vector<float> W;
    const uint32_t nr_feature, nr_factor, nr_field;
};

FILE *open_c_file(std::string const &path, std::string const &mode);

std::vector<std::string> 
argv_to_args(int const argc, char const * const * const argv);

inline float wTx(Problem const &prob, Model &model, uint32_t const i, 
    float const kappa=0, float const eta=0, float const lambda=0, 
    bool const do_update=false)
{
    uint32_t const nr_factor = model.nr_factor;
    uint32_t const nr_field = model.nr_field;
    uint32_t const nr_feature = model.nr_feature;
    uint64_t const align0 = nr_factor*kW_NODE_SIZE;
    uint64_t const align1 = nr_field*align0;

    uint32_t const * const J = &prob.J[i*nr_field];
    float * const W = model.W.data();

    __m128 const XMMv = _mm_set1_ps(prob.v);
    __m128 const XMMkappav = _mm_set1_ps(kappa*prob.v);
    __m128 const XMMeta = _mm_set1_ps(eta);
    __m128 const XMMlambda = _mm_set1_ps(lambda);

    __m128 XMMt = _mm_setzero_ps();
    for(uint32_t f1 = 0; f1 < nr_field; ++f1)
    {
        uint32_t const j1 = J[f1];
        if(j1 >= nr_feature)
            continue;

        for(uint32_t f2 = f1+1; f2 < nr_field; ++f2)
        {
            uint32_t const j2 = J[f2];
            if(j2 >= nr_feature)
                continue;

            float * const w1 = W + j1*align1 + f2*align0;
            float * const w2 = W + j2*align1 + f1*align0;

            if(do_update)
            {
                float * const wg1 = w1 + nr_factor;
                float * const wg2 = w2 + nr_factor;
                for(uint32_t d = 0; d < nr_factor; d += 4)
                {
                    __m128 XMMw1 = _mm_load_ps(w1+d);
                    __m128 XMMw2 = _mm_load_ps(w2+d);

                    __m128 XMMwg1 = _mm_load_ps(wg1+d);
                    __m128 XMMwg2 = _mm_load_ps(wg2+d);

                    __m128 XMMg1 = _mm_add_ps(
                                   _mm_mul_ps(XMMlambda, XMMw1),
                                   _mm_mul_ps(XMMkappav, XMMw2));
                    __m128 XMMg2 = _mm_add_ps(
                                   _mm_mul_ps(XMMlambda, XMMw2),
                                   _mm_mul_ps(XMMkappav, XMMw1));

                    XMMwg1 = _mm_add_ps(XMMwg1, _mm_mul_ps(XMMg1, XMMg1));
                    XMMwg2 = _mm_add_ps(XMMwg2, _mm_mul_ps(XMMg2, XMMg2));

                    XMMw1 = _mm_sub_ps(XMMw1, _mm_mul_ps(XMMeta, 
                            _mm_mul_ps(_mm_rsqrt_ps(XMMwg1), XMMg1)));
                    XMMw2 = _mm_sub_ps(XMMw2, _mm_mul_ps(XMMeta, 
                            _mm_mul_ps(_mm_rsqrt_ps(XMMwg2), XMMg2)));

                    _mm_store_ps(w1+d, XMMw1);
                    _mm_store_ps(w2+d, XMMw2);

                    _mm_store_ps(wg1+d, XMMwg1);
                    _mm_store_ps(wg2+d, XMMwg2);
                }
            }
            else
            {
                for(uint32_t d = 0; d < nr_factor; d += 4)
                {
                    __m128 const XMMw1 = _mm_load_ps(w1+d);
                    __m128 const XMMw2 = _mm_load_ps(w2+d);

                    XMMt = _mm_add_ps(XMMt, 
                           _mm_mul_ps(_mm_mul_ps(XMMw1, XMMw2), XMMv));
                }
            }
        }
    }

    if(do_update)
        return 0;

    XMMt = _mm_hadd_ps(XMMt, XMMt);
    XMMt = _mm_hadd_ps(XMMt, XMMt);
    float t;
    _mm_store_ss(&t, XMMt);

    return t;
}

float predict(Problem const &prob, Model &model, 
    std::string const &output_path = std::string(""));
#endif // _COMMON_H_
