#include "goldilocks_cubic_extension.hpp"
#include "zhInv.hpp"
#include "starks.hpp"
#include "constant_pols_starks.hpp"
#include "fibonacciSteps.hpp"
#include "fibonacci.chelpers.step3.parser.hpp" 
#include <immintrin.h>

void FibonacciSteps::step3_parser_first_avx(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch) {
#pragma omp parallel for
   for (uint64_t i = 0; i < nrows; i+= nrowsBatch) {
      int i_args = 0;
      __m256i tmp1[NTEMP1_];
      Goldilocks3::Element_avx tmp3[NTEMP3_];
      uint64_t offsetsDest[4], offsetsSrc0[4], offsetsSrc1[4];
      uint64_t numConstPols = params.pConstPols->numPols();
      

      for (int kk = 0; kk < NOPS_; ++kk) {
          switch (op3[kk]) {
           case 0: {
               for (uint64_t j = 0; j < AVX_SIZE_; ++j) {
                  offsetsSrc0[j] = args3[i_args + 1] + (i + j) * args3[i_args + 2];
                  offsetsSrc1[j] = args3[i_args + 3] + (i + j) * args3[i_args + 4];
               }
                Goldilocks::mul_avx(tmp1[args3[i_args]], &params.pols[0], &params.pols[0], offsetsSrc0, offsetsSrc1);
                i_args += 5;
                break;
            }
           case 1: {
                Goldilocks::add_avx(&params.pols[args3[i_args] + i * args3[i_args + 1]], args3[i_args + 1], tmp1[args3[i_args + 2]], tmp1[args3[i_args + 3]]);
                i_args += 4;
                break;
            }
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