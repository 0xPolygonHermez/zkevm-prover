#include "goldilocks_cubic_extension.hpp"
#include "zhInv.hpp"
#include "starks.hpp"
#include "constant_pols_starks.hpp"
#include "fibonacciSteps.hpp"
#include "fibonacci.chelpers.step2prev.parser.hpp" 
#include <immintrin.h>

void FibonacciSteps::step2prev_parser_first_avx(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch) {
#pragma omp parallel for
   for (uint64_t i = 0; i < nrows; i+= nrowsBatch) {
      int i_args = 0;
      __m256i tmp1[NTEMP1_];
      Goldilocks3::Element_avx tmp3[NTEMP3_];
      uint64_t offsetsDest[4], offsetsSrc0[4], offsetsSrc1[4];
      uint64_t numConstPols = params.pConstPols->numPols();
      

      for (int kk = 0; kk < NOPS_; ++kk) {
          switch (op2prev[kk]) {
              default: {
                  std::cout << " Wrong operation in step42ns_first!" << std::endl;
                  exit(1);
              }
          }
       }
       if (i_args != NARGS_) std::cout << " " << i_args << " - " << NARGS_ << std::endl;
       assert(i_args == NARGS_);
   }
}