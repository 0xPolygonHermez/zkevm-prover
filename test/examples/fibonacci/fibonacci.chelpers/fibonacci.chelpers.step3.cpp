#include "goldilocks_cubic_extension.hpp"
#include "zhInv.hpp"
#include "starks.hpp"
#include "constant_pols_starks.hpp"
#include "fibonacciSteps.hpp"

void FibonacciSteps::step3_first(StepsParams &params, uint64_t i) {
     Goldilocks::Element tmp_0;
     Goldilocks::mul(tmp_0, params.pols[0 + i*2], params.pols[0 + i*2]);
     Goldilocks::Element tmp_1;
     Goldilocks::mul(tmp_1, params.pols[1 + i*2], params.pols[1 + i*2]);
     Goldilocks::add(params.pols[2048 + i*1], tmp_0, tmp_1);
}

void FibonacciSteps::step3_i(StepsParams &params, uint64_t i) {
     Goldilocks::Element tmp_0;
     Goldilocks::mul(tmp_0, params.pols[0 + i*2], params.pols[0 + i*2]);
     Goldilocks::Element tmp_1;
     Goldilocks::mul(tmp_1, params.pols[1 + i*2], params.pols[1 + i*2]);
     Goldilocks::add(params.pols[2048 + i*1], tmp_0, tmp_1);
}

void FibonacciSteps::step3_last(StepsParams &params, uint64_t i) {
     Goldilocks::Element tmp_0;
     Goldilocks::mul(tmp_0, params.pols[0 + i*2], params.pols[0 + i*2]);
     Goldilocks::Element tmp_1;
     Goldilocks::mul(tmp_1, params.pols[1 + i*2], params.pols[1 + i*2]);
     Goldilocks::add(params.pols[2048 + i*1], tmp_0, tmp_1);
}
