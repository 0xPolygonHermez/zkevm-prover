#ifndef STARKS_RECURSIVE_1_STEPS_HPP
#define STARKS_RECURSIVE_1_STEPS_HPP

#include "goldilocks_cubic_extension.hpp"
#include "zhInv.hpp"

#include "starks.hpp"
#include "constant_pols_starks.hpp"

class Recursive1Steps : public Steps
{
public:
    void step2prev_first(StepsParams &params, uint64_t i);
    void step2prev_i(StepsParams &params, uint64_t i);
    void step2prev_last(StepsParams &params, uint64_t i);

    void step3prev_first(StepsParams &params, uint64_t i);
    void step3prev_i(StepsParams &params, uint64_t i);
    void step3prev_last(StepsParams &params, uint64_t i);

    void step3_first(StepsParams &params, uint64_t i);
    void step3_i(StepsParams &params, uint64_t i);
    void step3_last(StepsParams &params, uint64_t i);

    void step4_first(StepsParams &params, uint64_t i);
    void step4_i(StepsParams &params, uint64_t i);
    void step4_last(StepsParams &params, uint64_t i);

    void step42ns_first(StepsParams &params, uint64_t i);
    void step42ns_i(StepsParams &params, uint64_t i);
    void step42ns_last(StepsParams &params, uint64_t i);

    void step52ns_first(StepsParams &params, uint64_t i);
    void step52ns_i(StepsParams &params, uint64_t i);
    void step52ns_last(StepsParams &params, uint64_t i);
};

#endif // STARKS_RECURSIVE_1_STEPS_HPP
