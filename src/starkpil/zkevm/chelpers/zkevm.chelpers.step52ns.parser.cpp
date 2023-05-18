#include "goldilocks_cubic_extension.hpp"
#include "zhInv.hpp"
#include "starks.hpp"
#include "constant_pols_starks.hpp"
#include "zkevmSteps.hpp"
#include "zkevm.chelpers.step52ns.parser.hpp"
#include <immintrin.h>

void ZkevmSteps::step52ns_parser_first_avx(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch)
{

#pragma omp parallel for
     for (uint64_t i = 0; i < nrows; i += nrowsBatch)
     {
          __m256i tmp0_0, tmp0_1, tmp0_2;
          __m256i tmp1_0, tmp1_1, tmp1_2;
          __m256i tmp2_0, tmp2_1, tmp2_2;

          tmp2_0 = _mm256_setzero_si256();
          tmp2_1 = _mm256_setzero_si256();
          tmp2_2 = _mm256_setzero_si256();

          // prepare constant arguments (challenge5, challenge6 and evals):
          Goldilocks::Element *challenge5 = params.challenges[5];
          Goldilocks::Element *challenge6 = params.challenges[6];
          Goldilocks::Element challenge5_ops[3];
          Goldilocks::Element challenge6_ops[3];

          challenge5_ops[0] = challenge5[0] + challenge5[1];
          challenge5_ops[1] = challenge5[0] + challenge5[2];
          challenge5_ops[2] = challenge5[1] + challenge5[2];

          challenge6_ops[0] = challenge6[0] + challenge6[1];
          challenge6_ops[1] = challenge6[0] + challenge6[2];
          challenge6_ops[2] = challenge6[1] + challenge6[2];

          Goldilocks::Element aux0_ops[4], aux1_ops[4], aux2_ops[4];
          Goldilocks::Element aux0[4], aux1[4], aux2[4];
          __m256i chall50_, chall51_, chall52_;
          __m256i chall5o0_, chall5o1_, chall5o2_;
          __m256i chall60_, chall61_, chall62_;
          __m256i chall6o0_, chall6o1_, chall6o2_;

          for (int k = 0; k < AVX_SIZE_; ++k)
          {
               aux0_ops[k] = challenge5_ops[0];
               aux1_ops[k] = challenge5_ops[1];
               aux2_ops[k] = challenge5_ops[2];
               aux0[k] = challenge5[0];
               aux1[k] = challenge5[1];
               aux2[k] = challenge5[2];
          }
          Goldilocks::load_avx(chall5o0_, aux0_ops);
          Goldilocks::load_avx(chall5o1_, aux1_ops);
          Goldilocks::load_avx(chall5o2_, aux2_ops);
          Goldilocks::load_avx(chall50_, aux0);
          Goldilocks::load_avx(chall51_, aux1);
          Goldilocks::load_avx(chall52_, aux2);

          for (int k = 0; k < AVX_SIZE_; ++k)
          {
               aux0_ops[k] = challenge6_ops[0];
               aux1_ops[k] = challenge6_ops[1];
               aux2_ops[k] = challenge6_ops[2];
               aux0[k] = challenge6[0];
               aux1[k] = challenge6[1];
               aux2[k] = challenge6[2];
          }
          Goldilocks::load_avx(chall6o0_, aux0_ops);
          Goldilocks::load_avx(chall6o1_, aux1_ops);
          Goldilocks::load_avx(chall6o2_, aux2_ops);
          Goldilocks::load_avx(chall60_, aux0);
          Goldilocks::load_avx(chall61_, aux1);
          Goldilocks::load_avx(chall62_, aux2);
          Goldilocks::Element *evals_ = params.evals[0];

          // Parser
          int i_args = 0;
          for (int kk = 0; kk < NOPS_; ++kk)
          {
               switch (op52[kk])
               {
               case 0:
               {
                    Goldilocks3::mul13c_avx(tmp0_0, tmp0_1, tmp0_2, &params.pols[args52[i_args] + i * args52[i_args + 1]], (Goldilocks3::Element &)*params.challenges[5], args52[i_args + 1]);
                    i_args += 2;
                    break;
               }
               case 1:
               {
                    Goldilocks3::mul_avx(tmp0_0, tmp0_1, tmp0_2, tmp0_0, tmp0_1, tmp0_2, chall50_, chall51_, chall52_, chall5o0_, chall5o1_, chall5o2_);
                    break;
               }
               case 2:
               {
                    Goldilocks3::mul_avx(tmp0_0, tmp0_1, tmp0_2, tmp0_0, tmp0_1, tmp0_2, chall60_, chall61_, chall62_, chall6o0_, chall6o1_, chall6o2_);
                    break;
               }
               case 3:
               {
                    Goldilocks3::mul_avx(tmp1_0, tmp1_1, tmp1_2, tmp0_0, tmp0_1, tmp0_2, chall50_, chall51_, chall52_, chall5o0_, chall5o1_, chall5o2_);
                    break;
               }
               case 4:
               {
                    Goldilocks3::mul_avx(tmp0_0, tmp0_1, tmp0_2, tmp2_0, tmp2_1, tmp2_2, chall60_, chall61_, chall62_, chall6o0_, chall6o1_, chall6o2_);
                    break;
               }
               case 5:
               {
                    Goldilocks3::mul_avx(tmp0_0, tmp0_1, tmp0_2, tmp0_0, tmp0_1, tmp0_2, params.xDivXSubXi[i]);
                    break;
               }
               case 6:
               {
                    Goldilocks3::mul_avx(tmp0_0, tmp0_1, tmp0_2, tmp0_0, tmp0_1, tmp0_2, params.xDivXSubWXi[i]);
                    break;
               }
               case 7:
               {
                    Goldilocks3::add_avx(tmp0_0, tmp0_1, tmp0_2, tmp0_0, tmp0_1, tmp0_2, tmp2_0, tmp2_1, tmp2_2);
                    break;
               }
               case 8:
               {
                    Goldilocks3::add_avx(tmp0_0, tmp0_1, tmp0_2, tmp1_0, tmp1_1, tmp1_2, tmp0_0, tmp0_1, tmp0_2);
                    break;
               }
               case 9:
               {
                    Goldilocks3::add_avx(tmp0_0, tmp0_1, tmp0_2, tmp0_0, tmp0_1, tmp0_2, &params.pols[args52[i_args] + i * args52[i_args + 1]], args52[i_args + 1]);
                    i_args += 2;
                    break;
               }
               case 10:
               {
                    Goldilocks3::add31_avx(tmp0_0, tmp0_1, tmp0_2, tmp0_0, tmp0_1, tmp0_2, &params.pols[args52[i_args] + i * args52[i_args + 1]], args52[i_args + 1]);
                    i_args += 2;
                    break;
               }
               case 11:
               {
                    Goldilocks3::sub13c_avx(tmp2_0, tmp2_1, tmp2_2, &params.pols[args52[i_args] + i * args52[i_args + 1]], &evals_[args52[i_args + 2] * 3], args52[i_args + 1]);
                    i_args += 3;
                    break;
               }
               case 12:
               {
                    Goldilocks3::sub33c_avx(tmp2_0, tmp2_1, tmp2_2, &params.pols[args52[i_args] + i * args52[i_args + 1]], &evals_[args52[i_args + 2] * 3], args52[i_args + 1]);
                    i_args += 3;
                    break;
               }
               case 13:
               {
                    Goldilocks3::sub13c_avx(tmp2_0, tmp2_1, tmp2_2, &params.pConstPols2ns->getElement(args52[i_args], i), &evals_[args52[i_args + 1] * 3], params.pConstPols2ns->numPols());
                    i_args += 2;
                    break;
               }
               case 14:
               {
                    Goldilocks3::sub13c_avx(tmp0_0, tmp0_1, tmp0_2, &params.pConstPols2ns->getElement(5, i), evals_, params.pConstPols2ns->numPols());
                    break;
               }
               case 15:
               {
                    Goldilocks3::copy_avx(&(params.f_2ns[i * 3]), tmp0_0, tmp0_1, tmp0_2);
                    break;
               }
               case 16:
               {
                    // 1, 10, -> 16,: 768
                    Goldilocks3::mul_avx(tmp0_0, tmp0_1, tmp0_2, tmp0_0, tmp0_1, tmp0_2, chall50_, chall51_, chall52_, chall5o0_, chall5o1_, chall5o2_);
                    Goldilocks3::add31_avx(tmp0_0, tmp0_1, tmp0_2, tmp0_0, tmp0_1, tmp0_2, &params.pols[args52[i_args] + i * args52[i_args + 1]], args52[i_args + 1]);
                    i_args += 2;
                    break;
               }
               case 17:
               {
                    // 1, 9, -> 17,: 138
                    Goldilocks3::mul_avx(tmp0_0, tmp0_1, tmp0_2, tmp0_0, tmp0_1, tmp0_2, chall50_, chall51_, chall52_, chall5o0_, chall5o1_, chall5o2_);
                    Goldilocks3::add_avx(tmp0_0, tmp0_1, tmp0_2, tmp0_0, tmp0_1, tmp0_2, &params.pols[args52[i_args] + i * args52[i_args + 1]], args52[i_args + 1]);
                    i_args += 2;
                    break;
               }
               case 18:
               {
                    //  2, 11, 7, -> 18,: 1237
                    Goldilocks3::mul_avx(tmp0_0, tmp0_1, tmp0_2, tmp0_0, tmp0_1, tmp0_2, chall60_, chall61_, chall62_, chall6o0_, chall6o1_, chall6o2_);
                    Goldilocks3::sub13c_avx(tmp2_0, tmp2_1, tmp2_2, &params.pols[args52[i_args] + i * args52[i_args + 1]], &evals_[args52[i_args + 2] * 3], args52[i_args + 1]);
                    Goldilocks3::add_avx(tmp0_0, tmp0_1, tmp0_2, tmp0_0, tmp0_1, tmp0_2, tmp2_0, tmp2_1, tmp2_2);
                    i_args += 3;
                    break;
               }
               case 19:
               {
                    // 2, 13, 7, -> 19: 338
                    Goldilocks3::mul_avx(tmp0_0, tmp0_1, tmp0_2, tmp0_0, tmp0_1, tmp0_2, chall60_, chall61_, chall62_, chall6o0_, chall6o1_, chall6o2_);
                    Goldilocks3::sub13c_avx(tmp2_0, tmp2_1, tmp2_2, &params.pConstPols2ns->getElement(args52[i_args], i), &evals_[args52[i_args + 1] * 3], params.pConstPols2ns->numPols());
                    Goldilocks3::add_avx(tmp0_0, tmp0_1, tmp0_2, tmp0_0, tmp0_1, tmp0_2, tmp2_0, tmp2_1, tmp2_2);
                    i_args += 2;
                    break;
               }
               case 20:
               {
                    // 2, 12, 7, -> 20: 205
                    Goldilocks3::mul_avx(tmp0_0, tmp0_1, tmp0_2, tmp0_0, tmp0_1, tmp0_2, chall60_, chall61_, chall62_, chall6o0_, chall6o1_, chall6o2_);
                    Goldilocks3::sub33c_avx(tmp2_0, tmp2_1, tmp2_2, &params.pols[args52[i_args] + i * args52[i_args + 1]], &evals_[args52[i_args + 2] * 3], args52[i_args + 1]);
                    Goldilocks3::add_avx(tmp0_0, tmp0_1, tmp0_2, tmp0_0, tmp0_1, tmp0_2, tmp2_0, tmp2_1, tmp2_2);
                    i_args += 3;
                    break;
               }
               default:
                    std::ostringstream message;
                    message << "Invalid operation in step52ns_first, component: " << kk << " value: " << op52[kk];
                    throw new std::invalid_argument(message.str());
               }
          }
          assert(i_args == NARGS_);
     }
}

void ZkevmSteps::step52ns_parser_first(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch)
{
#pragma omp parallel for
     for (uint64_t i = 0; i < nrows; i += nrowsBatch)
     {
          Goldilocks::Element tmp[AVX_SIZE_ * 3], tmp1[AVX_SIZE_ * 3], tmp2[AVX_SIZE_ * 3];

          Goldilocks::Element *challenge5 = params.challenges[5];
          Goldilocks::Element *challenge6 = params.challenges[6];
          Goldilocks::Element challenge5_ops[3];
          Goldilocks::Element challenge6_ops[3];

          challenge5_ops[0] = challenge5[0] + challenge5[1];
          challenge5_ops[1] = challenge5[0] + challenge5[2];
          challenge5_ops[2] = challenge5[1] + challenge5[2];

          challenge6_ops[0] = challenge6[0] + challenge6[1];
          challenge6_ops[1] = challenge6[0] + challenge6[2];
          challenge6_ops[2] = challenge6[1] + challenge6[2];

          Goldilocks::Element *evals_ = params.evals[0];

          int i_args = 0;

          for (int kk = 0; kk < NOPS_; ++kk)
          {
               switch (op52[kk])
               {
               case 0:
               {
                    Goldilocks3::mul13c_batch(tmp, &params.pols[args52[i_args] + i * args52[i_args + 1]], (Goldilocks3::Element &)*params.challenges[5], args52[i_args + 1]);
                    i_args += 2;
                    break;
               }
               case 1:
               {
                    Goldilocks3::mul_batch(tmp, tmp, challenge5, challenge5_ops);
                    break;
               }
               case 2:
               {
                    Goldilocks3::mul_batch(tmp, tmp, challenge6, challenge6_ops);
                    break;
               }
               case 3:
               {
                    Goldilocks3::mul_batch(tmp1, tmp, challenge5, challenge5_ops);
                    break;
               }
               case 4:
               {
                    Goldilocks3::mul_batch(tmp, tmp2, challenge6, challenge6_ops);
                    break;
               }
               case 5:
               {
                    Goldilocks3::mul_batch(tmp, tmp, params.xDivXSubXi[i]);
                    break;
               }
               case 6:
               {
                    Goldilocks3::mul_batch(tmp, tmp, params.xDivXSubWXi[i]);
                    break;
               }
               case 7:
               {
                    Goldilocks3::add_batch(tmp, tmp, tmp2);
                    break;
               }
               case 8:
               {
                    Goldilocks3::add_batch(tmp, tmp1, tmp);
                    break;
               }
               case 9:
               {
                    Goldilocks3::add_batch(tmp, tmp, &params.pols[args52[i_args] + i * args52[i_args + 1]], FIELD_EXTENSION, args52[i_args + 1]);
                    i_args += 2;
                    break;
               }
               case 10:
               {
                    Goldilocks3::add31_batch(tmp, tmp, &params.pols[args52[i_args] + i * args52[i_args + 1]], FIELD_EXTENSION, args52[i_args + 1]);
                    i_args += 2;
                    break;
               }
               case 11:
               {
                    Goldilocks3::sub13c_batch(tmp2, &params.pols[args52[i_args] + i * args52[i_args + 1]], &evals_[args52[i_args + 2] * 3], args52[i_args + 1]);
                    i_args += 3;
                    break;
               }
               case 12:
               {
                    Goldilocks3::sub33c_batch(tmp2, &params.pols[args52[i_args] + i * args52[i_args + 1]], &evals_[args52[i_args + 2] * 3], args52[i_args + 1]);
                    i_args += 3;
                    break;
               }
               case 13:
               {
                    Goldilocks3::sub13c_batch(tmp2, &params.pConstPols2ns->getElement(args52[i_args], i), &evals_[args52[i_args + 1] * 3], params.pConstPols2ns->numPols());
                    i_args += 2;
                    break;
               }
               case 14:
               {
                    Goldilocks3::sub13c_batch(tmp, &params.pConstPols2ns->getElement(5, i), evals_, params.pConstPols2ns->numPols());
                    break;
               }
               case 15:
               {
                    Goldilocks3::copy_batch(&(params.f_2ns[i * 3]), tmp);
                    break;
               }
               case 16:
               {
                    // 1, 10, -> 16,: 768
                    Goldilocks3::mul_batch(tmp, tmp, challenge5, challenge5_ops);
                    Goldilocks3::add31_batch(tmp, tmp, &params.pols[args52[i_args] + i * args52[i_args + 1]], FIELD_EXTENSION, args52[i_args + 1]);
                    i_args += 2;
                    break;
               }
               case 17:
               {
                    // 1, 9, -> 17,: 138
                    Goldilocks3::mul_batch(tmp, tmp, challenge5, challenge5_ops);
                    Goldilocks3::add_batch(tmp, tmp, &params.pols[args52[i_args] + i * args52[i_args + 1]], FIELD_EXTENSION, args52[i_args + 1]);
                    i_args += 2;
                    break;
               }
               case 18:
               {
                    //  2, 11, 7, -> 18,: 1237
                    Goldilocks3::mul_batch(tmp, tmp, challenge6, challenge6_ops);
                    Goldilocks3::sub13c_batch(tmp2, &params.pols[args52[i_args] + i * args52[i_args + 1]], &evals_[args52[i_args + 2] * 3], args52[i_args + 1]);
                    Goldilocks3::add_batch(tmp, tmp, tmp2);
                    i_args += 3;
                    break;
               }
               case 19:
               {
                    // 2, 13, 7, -> 19: 332
                    Goldilocks3::mul_batch(tmp, tmp, challenge6, challenge6_ops);
                    Goldilocks3::sub13c_batch(tmp2, &params.pConstPols2ns->getElement(args52[i_args], i), &evals_[args52[i_args + 1] * 3], params.pConstPols2ns->numPols());
                    Goldilocks3::add_batch(tmp, tmp, tmp2);
                    i_args += 2;
                    break;
               }
               case 20:
               {
                    // 2, 12, 7, -> 20: 205
                    Goldilocks3::mul_batch(tmp, tmp, challenge6, challenge6_ops);
                    Goldilocks3::sub33c_batch(tmp2, &params.pols[args52[i_args] + i * args52[i_args + 1]], &evals_[args52[i_args + 2] * 3], args52[i_args + 1]);
                    Goldilocks3::add_batch(tmp, tmp, tmp2);
                    i_args += 3;
                    break;
               }
               default:
                    std::ostringstream message;
                    message << "Invalid operation in step52ns_first, component: " << kk << " value: " << op52[kk];
                    throw new std::invalid_argument(message.str());
               }
          }
          assert(i_args == NARGS_);
     }
}

#ifdef __AVX512__
void ZkevmSteps::step52ns_parser_first_avx512(StepsParams &params, uint64_t nrows, uint64_t nrowsBatch)
{

#pragma omp parallel for
     for (uint64_t i = 0; i < nrows; i += nrowsBatch)
     {
          __m512i tmp0_0, tmp0_1, tmp0_2;
          __m512i tmp1_0, tmp1_1, tmp1_2;
          __m512i tmp2_0, tmp2_1, tmp2_2;

          tmp2_0 = _mm512_setzero_si512();
          tmp2_1 = _mm512_setzero_si512();
          tmp2_2 = _mm512_setzero_si512();

          // prepare constant arguments (challenge5, challenge6 and evals):
          Goldilocks::Element *challenge5 = params.challenges[5];
          Goldilocks::Element *challenge6 = params.challenges[6];
          Goldilocks::Element challenge5_ops[3];
          Goldilocks::Element challenge6_ops[3];

          challenge5_ops[0] = challenge5[0] + challenge5[1];
          challenge5_ops[1] = challenge5[0] + challenge5[2];
          challenge5_ops[2] = challenge5[1] + challenge5[2];

          challenge6_ops[0] = challenge6[0] + challenge6[1];
          challenge6_ops[1] = challenge6[0] + challenge6[2];
          challenge6_ops[2] = challenge6[1] + challenge6[2];

          Goldilocks::Element aux0_ops[AVX512_SIZE_], aux1_ops[AVX512_SIZE_], aux2_ops[AVX512_SIZE_];
          Goldilocks::Element aux0[AVX512_SIZE_], aux1[AVX512_SIZE_], aux2[AVX512_SIZE_];
          __m512i chall50_, chall51_, chall52_;
          __m512i chall5o0_, chall5o1_, chall5o2_;
          __m512i chall60_, chall61_, chall62_;
          __m512i chall6o0_, chall6o1_, chall6o2_;

          for (int k = 0; k < AVX512_SIZE_; ++k)
          {
               aux0_ops[k] = challenge5_ops[0];
               aux1_ops[k] = challenge5_ops[1];
               aux2_ops[k] = challenge5_ops[2];
               aux0[k] = challenge5[0];
               aux1[k] = challenge5[1];
               aux2[k] = challenge5[2];
          }
          Goldilocks::load_avx512(chall5o0_, aux0_ops);
          Goldilocks::load_avx512(chall5o1_, aux1_ops);
          Goldilocks::load_avx512(chall5o2_, aux2_ops);
          Goldilocks::load_avx512(chall50_, aux0);
          Goldilocks::load_avx512(chall51_, aux1);
          Goldilocks::load_avx512(chall52_, aux2);

          for (int k = 0; k < AVX512_SIZE_; ++k)
          {
               aux0_ops[k] = challenge6_ops[0];
               aux1_ops[k] = challenge6_ops[1];
               aux2_ops[k] = challenge6_ops[2];
               aux0[k] = challenge6[0];
               aux1[k] = challenge6[1];
               aux2[k] = challenge6[2];
          }
          Goldilocks::load_avx512(chall6o0_, aux0_ops);
          Goldilocks::load_avx512(chall6o1_, aux1_ops);
          Goldilocks::load_avx512(chall6o2_, aux2_ops);
          Goldilocks::load_avx512(chall60_, aux0);
          Goldilocks::load_avx512(chall61_, aux1);
          Goldilocks::load_avx512(chall62_, aux2);
          Goldilocks::Element *evals_ = params.evals[0];

          // Parser
          int i_args = 0;
          for (int kk = 0; kk < NOPS_; ++kk)
          {
               switch (op52[kk])
               {
               case 0:
               {
                    Goldilocks3::mul13c_avx512(tmp0_0, tmp0_1, tmp0_2, &params.pols[args52[i_args] + i * args52[i_args + 1]], (Goldilocks3::Element &)*params.challenges[5], args52[i_args + 1]);
                    i_args += 2;
                    break;
               }
               case 1:
               {
                    Goldilocks3::mul_avx512(tmp0_0, tmp0_1, tmp0_2, tmp0_0, tmp0_1, tmp0_2, chall50_, chall51_, chall52_, chall5o0_, chall5o1_, chall5o2_);
                    break;
               }
               case 2:
               {
                    Goldilocks3::mul_avx512(tmp0_0, tmp0_1, tmp0_2, tmp0_0, tmp0_1, tmp0_2, chall60_, chall61_, chall62_, chall6o0_, chall6o1_, chall6o2_);
                    break;
               }
               case 3:
               {
                    Goldilocks3::mul_avx512(tmp1_0, tmp1_1, tmp1_2, tmp0_0, tmp0_1, tmp0_2, chall50_, chall51_, chall52_, chall5o0_, chall5o1_, chall5o2_);
                    break;
               }
               case 4:
               {
                    Goldilocks3::mul_avx512(tmp0_0, tmp0_1, tmp0_2, tmp2_0, tmp2_1, tmp2_2, chall60_, chall61_, chall62_, chall6o0_, chall6o1_, chall6o2_);
                    break;
               }
               case 5:
               {
                    Goldilocks3::mul_avx512(tmp0_0, tmp0_1, tmp0_2, tmp0_0, tmp0_1, tmp0_2, params.xDivXSubXi[i]);
                    break;
               }
               case 6:
               {
                    Goldilocks3::mul_avx512(tmp0_0, tmp0_1, tmp0_2, tmp0_0, tmp0_1, tmp0_2, params.xDivXSubWXi[i]);
                    break;
               }
               case 7:
               {
                    Goldilocks3::add_avx512(tmp0_0, tmp0_1, tmp0_2, tmp0_0, tmp0_1, tmp0_2, tmp2_0, tmp2_1, tmp2_2);
                    break;
               }
               case 8:
               {
                    Goldilocks3::add_avx512(tmp0_0, tmp0_1, tmp0_2, tmp1_0, tmp1_1, tmp1_2, tmp0_0, tmp0_1, tmp0_2);
                    break;
               }
               case 9:
               {
                    Goldilocks3::add_avx512(tmp0_0, tmp0_1, tmp0_2, tmp0_0, tmp0_1, tmp0_2, &params.pols[args52[i_args] + i * args52[i_args + 1]], args52[i_args + 1]);
                    i_args += 2;
                    break;
               }
               case 10:
               {
                    Goldilocks3::add31_avx512(tmp0_0, tmp0_1, tmp0_2, tmp0_0, tmp0_1, tmp0_2, &params.pols[args52[i_args] + i * args52[i_args + 1]], args52[i_args + 1]);
                    i_args += 2;
                    break;
               }
               case 11:
               {
                    Goldilocks3::sub13c_avx512(tmp2_0, tmp2_1, tmp2_2, &params.pols[args52[i_args] + i * args52[i_args + 1]], &evals_[args52[i_args + 2] * 3], args52[i_args + 1]);
                    i_args += 3;
                    break;
               }
               case 12:
               {
                    Goldilocks3::sub33c_avx512(tmp2_0, tmp2_1, tmp2_2, &params.pols[args52[i_args] + i * args52[i_args + 1]], &evals_[args52[i_args + 2] * 3], args52[i_args + 1]);
                    i_args += 3;
                    break;
               }
               case 13:
               {
                    Goldilocks3::sub13c_avx512(tmp2_0, tmp2_1, tmp2_2, &params.pConstPols2ns->getElement(args52[i_args], i), &evals_[args52[i_args + 1] * 3], params.pConstPols2ns->numPols());
                    i_args += 2;
                    break;
               }
               case 14:
               {
                    Goldilocks3::sub13c_avx512(tmp0_0, tmp0_1, tmp0_2, &params.pConstPols2ns->getElement(5, i), evals_, params.pConstPols2ns->numPols());
                    break;
               }
               case 15:
               {
                    Goldilocks3::copy_avx512(&(params.f_2ns[i * 3]), tmp0_0, tmp0_1, tmp0_2);
                    break;
               }
               case 16:
               {
                    // 1, 10, -> 16,: 768
                    Goldilocks3::mul_avx512(tmp0_0, tmp0_1, tmp0_2, tmp0_0, tmp0_1, tmp0_2, chall50_, chall51_, chall52_, chall5o0_, chall5o1_, chall5o2_);
                    Goldilocks3::add31_avx512(tmp0_0, tmp0_1, tmp0_2, tmp0_0, tmp0_1, tmp0_2, &params.pols[args52[i_args] + i * args52[i_args + 1]], args52[i_args + 1]);
                    i_args += 2;
                    break;
               }
               case 17:
               {
                    // 1, 9, -> 17,: 138
                    Goldilocks3::mul_avx512(tmp0_0, tmp0_1, tmp0_2, tmp0_0, tmp0_1, tmp0_2, chall50_, chall51_, chall52_, chall5o0_, chall5o1_, chall5o2_);
                    Goldilocks3::add_avx512(tmp0_0, tmp0_1, tmp0_2, tmp0_0, tmp0_1, tmp0_2, &params.pols[args52[i_args] + i * args52[i_args + 1]], args52[i_args + 1]);
                    i_args += 2;
                    break;
               }
               case 18:
               {
                    //  2, 11, 7, -> 18,: 1237
                    Goldilocks3::mul_avx512(tmp0_0, tmp0_1, tmp0_2, tmp0_0, tmp0_1, tmp0_2, chall60_, chall61_, chall62_, chall6o0_, chall6o1_, chall6o2_);
                    Goldilocks3::sub13c_avx512(tmp2_0, tmp2_1, tmp2_2, &params.pols[args52[i_args] + i * args52[i_args + 1]], &evals_[args52[i_args + 2] * 3], args52[i_args + 1]);
                    Goldilocks3::add_avx512(tmp0_0, tmp0_1, tmp0_2, tmp0_0, tmp0_1, tmp0_2, tmp2_0, tmp2_1, tmp2_2);
                    i_args += 3;
                    break;
               }
               case 19:
               {
                    // 2, 13, 7, -> 19: 338
                    Goldilocks3::mul_avx512(tmp0_0, tmp0_1, tmp0_2, tmp0_0, tmp0_1, tmp0_2, chall60_, chall61_, chall62_, chall6o0_, chall6o1_, chall6o2_);
                    Goldilocks3::sub13c_avx512(tmp2_0, tmp2_1, tmp2_2, &params.pConstPols2ns->getElement(args52[i_args], i), &evals_[args52[i_args + 1] * 3], params.pConstPols2ns->numPols());
                    Goldilocks3::add_avx512(tmp0_0, tmp0_1, tmp0_2, tmp0_0, tmp0_1, tmp0_2, tmp2_0, tmp2_1, tmp2_2);
                    i_args += 2;
                    break;
               }
               case 20:
               {
                    // 2, 12, 7, -> 20: 205
                    Goldilocks3::mul_avx512(tmp0_0, tmp0_1, tmp0_2, tmp0_0, tmp0_1, tmp0_2, chall60_, chall61_, chall62_, chall6o0_, chall6o1_, chall6o2_);
                    Goldilocks3::sub33c_avx512(tmp2_0, tmp2_1, tmp2_2, &params.pols[args52[i_args] + i * args52[i_args + 1]], &evals_[args52[i_args + 2] * 3], args52[i_args + 1]);
                    Goldilocks3::add_avx512(tmp0_0, tmp0_1, tmp0_2, tmp0_0, tmp0_1, tmp0_2, tmp2_0, tmp2_1, tmp2_2);
                    i_args += 3;
                    break;
               }
               default:
                    std::ostringstream message;
                    message << "Invalid operation in step52ns_first, component: " << kk << " value: " << op52[kk];
                    throw new std::invalid_argument(message.str());
               }
          }
          assert(i_args == NARGS_);
     }
}
#endif