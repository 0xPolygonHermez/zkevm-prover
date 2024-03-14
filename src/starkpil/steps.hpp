#ifndef STEPS_HPP
#define STEPS_HPP

struct StepsParams
{
    Goldilocks::Element *pols;
    ConstantPolsStarks *pConstPols;
    ConstantPolsStarks *pConstPols2ns;
    Polinomial &challenges;
    Polinomial &x_n;
    Polinomial &x_2ns;
    ZhInv &zi;
    Polinomial &evals;
    Polinomial &xDivXSubXi;
    Polinomial &xDivXSubWXi;
    Goldilocks::Element *publicInputs;
    Goldilocks::Element *q_2ns;
    Goldilocks::Element *f_2ns;
};

class Steps
{
public:
    virtual void step2prev_first(StepsParams &params, uint64_t i) = 0;
    virtual void step2prev_i(StepsParams &params, uint64_t i) = 0;
    virtual void step2prev_last(StepsParams &params, uint64_t i) = 0;
    virtual void step2prev_parser_first_avx(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch){};
    virtual void step2prev_parser_first_avx512(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch){};

    virtual void step3prev_first(StepsParams &params, uint64_t i) = 0;
    virtual void step3prev_i(StepsParams &params, uint64_t i) = 0;
    virtual void step3prev_last(StepsParams &params, uint64_t i) = 0;
    virtual void step3prev_parser_first_avx(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch){};
    virtual void step3prev_parser_first_avx512(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch){};

    virtual void step3_first(StepsParams &params, uint64_t i) = 0;
    virtual void step3_i(StepsParams &params, uint64_t i) = 0;
    virtual void step3_last(StepsParams &params, uint64_t i) = 0;
    virtual void step3_parser_first(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch){};
    virtual void step3_parser_first_avx(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch){};
    virtual void step3_parser_first_avx_jump(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch){};
    virtual void step3_parser_first_avx512(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch){};

    virtual void step42ns_first(StepsParams &params, uint64_t i) = 0;
    virtual void step42ns_i(StepsParams &params, uint64_t i) = 0;
    virtual void step42ns_last(StepsParams &params, uint64_t i) = 0;
    virtual void step42ns_parser_first(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch){};
    virtual void step42ns_parser_first_avx(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch){};
    virtual void step42ns_parser_first_avx_jump(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch){};
    virtual void step42ns_parser_first_avx512(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch){};

    virtual void step52ns_first(StepsParams &params, uint64_t i) = 0;
    virtual void step52ns_i(StepsParams &params, uint64_t i) = 0;
    virtual void step52ns_last(StepsParams &params, uint64_t i) = 0;

    virtual void step52ns_parser_first(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch){};
    virtual void step52ns_parser_first_avx(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch){};
    virtual void step52ns_parser_first_avx512(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch){};
};

#endif // STEPS
