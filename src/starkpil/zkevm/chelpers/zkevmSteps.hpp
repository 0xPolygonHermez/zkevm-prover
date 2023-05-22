#ifndef STARKS_STEPS_HPP
#define STARKS_STEPS_HPP

#include "goldilocks_cubic_extension.hpp"
#include "zhInv.hpp"

#include "starks.hpp"
#include "constant_pols_starks.hpp"

class ZkevmSteps : public Steps
{
public:
    void step2prev_first(StepsParams &params, uint64_t i);
    void step2prev_i(StepsParams &params, uint64_t i);
    void step2prev_last(StepsParams &params, uint64_t i);
    void step2prev_parser_first_avx(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch);
#ifdef __AVX512__
    void step2prev_parser_first_avx512(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch);
#endif

    void step3prev_first(StepsParams &params, uint64_t i);
    void step3prev_i(StepsParams &params, uint64_t i);
    void step3prev_last(StepsParams &params, uint64_t i);
    void step3prev_parser_first_avx(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch);
#ifdef __AVX512__
    void step3prev_parser_first_avx512(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch);
#endif

    void step3_first(StepsParams &params, uint64_t i);
    void step3_i(StepsParams &params, uint64_t i);
    void step3_last(StepsParams &params, uint64_t i);
    void step3_parser_first(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch);
    void step3_parser_first_avx(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch);
    void step3_parser_first_avx_jump(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch);
#ifdef __AVX512__
    void step3_parser_first_avx512(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch);
#endif

    void step42ns_first(StepsParams &params, uint64_t i);
    void step42ns_i(StepsParams &params, uint64_t i);
    void step42ns_last(StepsParams &params, uint64_t i);
    void step42ns_parser_first(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch);
    void step42ns_parser_first_avx(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch);
    void step42ns_parser_first_avx_jump(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch);
#ifdef __AVX512__
    void step42ns_parser_first_avx512(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch);
#endif

    void step52ns_first(StepsParams &params, uint64_t i);
    void step52ns_i(StepsParams &params, uint64_t i);
    void step52ns_last(StepsParams &params, uint64_t i);
    void step52ns_parser_first(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch);
    void step52ns_parser_first_avx(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch);
#ifdef __AVX512__
    void step52ns_parser_first_avx512(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch);
#endif
};

#endif // STARKS_STEPS_HPP
