#include "goldilocks_cubic_extension.hpp"
#include "zhInv.hpp"
#include "starks.hpp"
#include "constant_pols_starks.hpp"
#include "allSteps.hpp"
#include "all.chelpers.step3.parser.hpp" 
#include <immintrin.h>

void AllSteps::step3_parser_first_avx(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch) {
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
                Goldilocks::copy_avx(tmp1[args3[i_args]], &params.pols[args3[i_args + 1] + i * args3[i_args + 2]], args3[i_args + 2]);
                i_args += 3;
                break;
            }
           case 1: {
               for (uint64_t j = 0; j < AVX_SIZE_; ++j) {
                  offsetsSrc0[j] = args3[i_args + 1] + (((i + j) + args3[i_args + 2]) % args3[i_args + 3]) * args3[i_args + 4];
               }
                Goldilocks::copy_avx(tmp1[args3[i_args]], &params.pols[0], offsetsSrc0);
                i_args += 5;
                break;
            }
           case 2: {
               for (uint64_t j = 0; j < AVX_SIZE_; ++j) {
                  offsetsSrc0[j] = args3[i_args + 1] + (i + j) * args3[i_args + 2];
                  offsetsSrc1[j] = args3[i_args + 3] + (((i + j) + args3[i_args + 4]) % args3[i_args + 5]) * args3[i_args + 6];
               }
                Goldilocks::mul_avx(tmp1[args3[i_args]], &params.pols[0], &params.pols[0], offsetsSrc0, offsetsSrc1);
                i_args += 7;
                break;
            }
           case 3: {
                Goldilocks::copy_avx(tmp1[args3[i_args]], &params.pConstPols->getElement(args3[i_args + 1], i), numConstPols);
                i_args += 2;
                break;
            }
           case 4: {
                Goldilocks3::mul13c_avx(tmp3[args3[i_args]], tmp1[args3[i_args + 1]], (Goldilocks3::Element &)*params.challenges[args3[i_args + 2]]);
                i_args += 3;
                break;
            }
           case 5: {
                Goldilocks3::add13_avx(tmp3[args3[i_args]], tmp1[args3[i_args + 1]], tmp3[args3[i_args + 2]]);
                i_args += 3;
                break;
            }
           case 6: {
                Goldilocks3::mul33c_avx(tmp3[args3[i_args]], tmp3[args3[i_args + 1]], (Goldilocks3::Element &)*params.challenges[args3[i_args + 2]]);
                i_args += 3;
                break;
            }
           case 7: {
                Goldilocks3::sub33c_avx(tmp3[args3[i_args]], tmp3[args3[i_args + 1]], (Goldilocks3::Element &)*params.challenges[args3[i_args + 2]]);
                i_args += 3;
                break;
            }
           case 8: {
                Goldilocks3::mul13_avx(tmp3[args3[i_args]], tmp1[args3[i_args + 1]], tmp3[args3[i_args + 2]]);
                i_args += 3;
                break;
            }
           case 9: {
                Goldilocks3::add33c_avx(&params.pols[args3[i_args] + i * args3[i_args + 1]], args3[i_args + 1], tmp3[args3[i_args + 2]], (Goldilocks3::Element &)*params.challenges[args3[i_args + 3]]);
                i_args += 4;
                break;
            }
           case 10: {
                Goldilocks3::sub_avx(tmp3[args3[i_args]], tmp3[args3[i_args + 1]], &params.pols[args3[i_args + 2] + i * args3[i_args + 3]], args3[i_args + 3]);
                i_args += 4;
                break;
            }
           case 11: {
                Goldilocks3::add_avx(&params.pols[args3[i_args] + i * args3[i_args + 1]], args3[i_args + 1], &params.pols[args3[i_args + 2] + i * args3[i_args + 3]], tmp3[args3[i_args + 4]], args3[i_args + 3]);
                i_args += 5;
                break;
            }
           case 12: {
               for (uint64_t j = 0; j < AVX_SIZE_; ++j) {
                  offsetsSrc0[j] = args3[i_args + 1] + (((i + j) + args3[i_args + 2]) % args3[i_args + 3]) * numConstPols;
               }
                Goldilocks::copy_avx(tmp1[args3[i_args]], &params.pConstPols->getElement(0, 0), offsetsSrc0);
                i_args += 4;
                break;
            }
           case 13: {
               for (uint64_t j = 0; j < AVX_SIZE_; ++j) {
                  offsetsSrc0[j] = args3[i_args + 1] + (((i + j) + args3[i_args + 2]) % args3[i_args + 3]) * args3[i_args + 4];
               }
                Goldilocks::copy_avx(tmp1[args3[i_args]], &params.pols[0], offsetsSrc0);
                i_args += 5;
                break;
            }
           case 14: {
               for (uint64_t j = 0; j < AVX_SIZE_; ++j) {
                  offsetsDest[j] = args3[i_args] + (((i + j) + args3[i_args + 1]) % args3[i_args + 2]) * args3[i_args + 3];
               }
                Goldilocks3::add33c_avx(&params.pols[0], offsetsDest, tmp3[args3[i_args + 4]], (Goldilocks3::Element &)*params.challenges[args3[i_args + 5]]);
                i_args += 6;
                break;
            }
           case 15: {
                Goldilocks3::add33c_avx(tmp3[args3[i_args]], &params.pols[args3[i_args + 1] + i * args3[i_args + 2]], (Goldilocks3::Element &)*params.challenges[args3[i_args + 3]], args3[i_args + 2]);
                i_args += 4;
                break;
            }
           case 16: {
               for (uint64_t j = 0; j < AVX_SIZE_; ++j) {
                  offsetsSrc0[j] = args3[i_args + 1] + (((i + j) + args3[i_args + 2]) % args3[i_args + 3]) * args3[i_args + 4];
               }
                Goldilocks3::mul33c_avx(tmp3[args3[i_args]], &params.pols[0], (Goldilocks3::Element &)*params.challenges[args3[i_args + 5]], offsetsSrc0);
                i_args += 6;
                break;
            }
           case 17: {
                Goldilocks3::add_avx(tmp3[args3[i_args]], &params.pols[args3[i_args + 1] + i * args3[i_args + 2]], tmp3[args3[i_args + 3]], args3[i_args + 2]);
                i_args += 4;
                break;
            }
           case 18: {
                Goldilocks3::add1c3c_avx(tmp3[args3[i_args]], Goldilocks::fromU64(args3[i_args + 1]), (Goldilocks3::Element &)*params.challenges[args3[i_args + 2]]);
                i_args += 3;
                break;
            }
           case 19: {
                Goldilocks3::add_avx(tmp3[args3[i_args]], tmp3[args3[i_args + 1]], tmp3[args3[i_args + 2]]);
                i_args += 3;
                break;
            }
           case 20: {
                Goldilocks3::mul_avx(tmp3[args3[i_args]], tmp3[args3[i_args + 1]], tmp3[args3[i_args + 2]]);
                i_args += 3;
                break;
            }
           case 21: {
                Goldilocks3::mul_avx(&params.pols[args3[i_args] + i * args3[i_args + 1]], args3[i_args + 1], tmp3[args3[i_args + 2]], tmp3[args3[i_args + 3]]);
                i_args += 4;
                break;
            }
           case 22: {
                Goldilocks3::mul13c_avx(tmp3[args3[i_args]], params.x_n[i], (Goldilocks3::Element &)*params.challenges[args3[i_args + 1]], params.x_n.offset());
                i_args += 2;
                break;
            }
           case 23: {
                Goldilocks3::add33c_avx(tmp3[args3[i_args]], tmp3[args3[i_args + 1]], (Goldilocks3::Element &)*params.challenges[args3[i_args + 2]]);
                i_args += 3;
                break;
            }
           case 24: {
                Goldilocks3::mul1c3c_avx(tmp3[args3[i_args]], Goldilocks::fromU64(args3[i_args + 1]), (Goldilocks3::Element &)*params.challenges[args3[i_args + 2]]);
                i_args += 3;
                break;
            }
           case 25: {
                Goldilocks3::mul13_avx(tmp3[args3[i_args]], params.x_n[i], tmp3[args3[i_args + 1]], params.x_n.offset());
                i_args += 2;
                break;
            }
           case 26: {
                Goldilocks3::add33c_avx(tmp3[args3[i_args]], tmp3[args3[i_args + 1]], (Goldilocks3::Element &)*params.challenges[args3[i_args + 2]]);
                i_args += 3;
                break;
            }
           case 27: {
                Goldilocks3::mul_avx(tmp3[args3[i_args]], tmp3[args3[i_args + 1]], tmp3[args3[i_args + 2]]);
                i_args += 3;
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