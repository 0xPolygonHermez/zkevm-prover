
#ifndef CHELPERS_STEPS_GPU_HPP
#define CHELPERS_STEPS_GPU_HPP
#include "chelpers.hpp"
#include "chelpers_steps.hpp"
#include "steps.hpp"
#ifdef __USE_CUDA__
#include <cuda_runtime.h>
#endif



class gl64_t;
class CHelpersStepsGPU : public CHelpersSteps {
public:
    uint64_t nCols;
    vector<uint64_t> nColsStages;
    vector<uint64_t> nColsStagesAcc;
    vector<uint64_t> offsetsStages;

    //Device polynomials

    uint32_t *ops_d;
    uint32_t *args_d;
    Goldilocks::Element *numbers_d;
    Goldilocks::Element *challenges_d;
    Goldilocks::Element *challenges_ops_d;
    Goldilocks::Element *publics_d;
    Goldilocks::Element *evals_d;
    uint32_t *storePol_d;

    Goldilocks::Element *constPols_d;
    Goldilocks::Element *constPols2ns_d;
    Goldilocks::Element *x_d;
    Goldilocks::Element *x_2ns_d;
    Goldilocks::Element *zi_d;
    Goldilocks::Element *pols_d;
    Goldilocks::Element *xDivXSubXi_d;

    uint32_t numBlocks;
    Goldilocks::Element ** bufferT_d;
    Goldilocks::Element ** tmp1_d;
    Goldilocks::Element ** tmp3_d;

#ifdef __USE_CUDA__     

    uint64_t* nColsStages_d_;
    uint64_t* nColsStagesAcc_d_;
    uint64_t* offsetsStages_d_;

    uint32_t *ops_d_;
    uint32_t *args_d_;
    gl64_t *numbers_d_;
    gl64_t *challenges_d_;
    gl64_t *challenges_ops_d_;
    gl64_t *publics_d_;
    gl64_t *evals_d_;
    uint32_t *storePol_d_;

    gl64_t *constPols_d_;
    gl64_t *constPols2ns_d_;
    gl64_t *x_d_;
    gl64_t *x_2ns_d_;
    gl64_t *zi_d_;
    gl64_t *pols_d_;
    gl64_t *xDivXSubXi_d_;

    gl64_t ** bufferT_d_;
    gl64_t ** tmp1_d_;
    gl64_t ** tmp3_d_;
#endif


    void setBufferTInfo(StarkInfo& starkInfo, uint64_t stage) override;
    void storePolinomials(StarkInfo &starkInfo, StepsParams &params, uint32_t* storePol, uint64_t row, uint64_t nrowsPack, uint64_t domainExtended)  {
        
        Goldilocks::Element *bufferT_ = bufferT_d[omp_get_thread_num()];

        if(domainExtended) {
            // Store either polinomial f or polinomial q
            for(uint64_t k = 0; k < nColsStages[10]; ++k) {
                Goldilocks::Element *buffT = &bufferT_[(nColsStagesAcc[10] + k)* nrowsPack];
                Goldilocks::copy_pack(nrowsPack, &params.pols[offsetsStages[10] + k + row * nColsStages[10]], nColsStages[10], buffT);
            }
        } else {
            uint64_t nStages = 3;
            uint64_t domainSize = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
            for(uint64_t s = 2; s <= nStages + 1; ++s) {
                bool isTmpPol = !domainExtended && s == 4;
                for(uint64_t k = 0; k < nColsStages[s]; ++k) {
                    uint64_t dim = storePol[nColsStagesAcc[s] + k];
                    if(storePol[nColsStagesAcc[s] + k]) {
                        Goldilocks::Element *buffT = &bufferT_[(nColsStagesAcc[s] + k)* nrowsPack];
                        if(isTmpPol) {
                            for(uint64_t i = 0; i < dim; ++i) {
                                Goldilocks::copy_pack(nrowsPack, &params.pols[offsetsStages[s] + k * domainSize + row * dim + i], uint64_t(dim), &buffT[i*nrowsPack]);
                            }
                        } else {
                            Goldilocks::copy_pack(nrowsPack, &params.pols[offsetsStages[s] + k + row * nColsStages[s]], nColsStages[s], buffT);
                        }
                    }
                }
            }
        }
    }

#ifdef __USE_CUDA__     
__device__  void storePolinomials_(StarkInfo &starkInfo, StepsParams &params, uint64_t row, uint64_t domainExtended);
#endif

    inline  void loadPolinomials(StarkInfo &starkInfo, StepsParams &params, uint64_t row, uint64_t stage, uint64_t nrowsPack, uint64_t domainExtended)  {
    
        Goldilocks::Element *bufferT_ = bufferT_d[omp_get_thread_num()];

        // buffered data
        Goldilocks::Element *constPols_aux = domainExtended ? constPols2ns_d : constPols_d;
        uint64_t domainSize = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
        Goldilocks::Element *x_aux = domainExtended ? x_2ns_d : x_d;

        uint64_t nStages = 3;
        uint64_t nextStride = domainExtended ?  1 << (starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits) : 1;
        std::vector<uint64_t> nextStrides = {0, nextStride};
        for(uint64_t k = 0; k < starkInfo.nConstants; ++k) {
            for(uint64_t o = 0; o < 2; ++o) {
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    uint64_t l = (row + j + nextStrides[o]) % domainSize;
                    bufferT_[(nColsStagesAcc[5*o] + k)*nrowsPack + j] = constPols_aux[l * starkInfo.nConstants + k];
                }
            }
        }

        // Load x and Zi
        for(uint64_t j = 0; j < nrowsPack; ++j) {
            bufferT_[starkInfo.nConstants*nrowsPack + j] = x_aux[row + j];
        }
        for(uint64_t j = 0; j < nrowsPack; ++j) {
            bufferT_[(starkInfo.nConstants + 1)*nrowsPack + j] = zi_d[row + j];
        }

        for(uint64_t s = 1; s <= nStages; ++s) {
            for(uint64_t k = 0; k < nColsStages[s]; ++k) {
                for(uint64_t o = 0; o < 2; ++o) {
                    for(uint64_t j = 0; j < nrowsPack; ++j) {
                        uint64_t l = (row + j + nextStrides[o]) % domainSize;
                        bufferT_[(nColsStagesAcc[5*o + s] + k)*nrowsPack + j] = pols_d[offsetsStages[s] + l * nColsStages[s] + k];
                    }
                }
            }
        }

        if(stage == 5) {
            for(uint64_t k = 0; k < nColsStages[nStages + 1]; ++k) {
                for(uint64_t o = 0; o < 2; ++o) {
                    for(uint64_t j = 0; j < nrowsPack; ++j) {
                        uint64_t l = (row + j + nextStrides[o]) % domainSize;
                        bufferT_[(nColsStagesAcc[5*o + nStages + 1] + k)*nrowsPack + j] = pols_d[offsetsStages[nStages + 1] + l * nColsStages[nStages + 1] + k];
                    }
                }
            }

           // Load xDivXSubXi & xDivXSubWXi
           for(uint64_t d = 0; d < 2; ++d) {
               for(uint64_t i = 0; i < FIELD_EXTENSION; ++i) {
                   for(uint64_t j = 0; j < nrowsPack; ++j) {
                      bufferT_[(nColsStagesAcc[11] + FIELD_EXTENSION*d + i)*nrowsPack + j] = xDivXSubXi_d[(d*domainSize + row + j)*FIELD_EXTENSION+i];
                   }
               }
           }
        }
    }

#ifdef __USE_CUDA__//rick: not coalesced accesses!
__device__ __forceinline__  void loadPolinomials_(StarkInfo &starkInfo, StepsParams &params, uint64_t row, uint64_t stage, uint64_t domainExtended);
#endif

 void calculateExpressions(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams);

#ifdef __USE_CUDA__

 void calculateExpressions_(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams); 

#endif

void optcodeIteration(uint32_t nrowsPack, uint32_t nOps, uint32_t nArgs) {

    uint64_t i_args = 0;
    Goldilocks::Element *bufferT_ = bufferT_d[omp_get_thread_num()];
    Goldilocks::Element *tmp1 = tmp1_d[omp_get_thread_num()];
    Goldilocks::Element *tmp3 = tmp3_d[omp_get_thread_num()];

    for (uint64_t kk = 0; kk < nOps; ++kk) {
        
        switch (ops_d[kk]) {
        case 0: {
            // COPY commit1 to commit1
            Goldilocks::copy_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args_d[i_args]] + args_d[i_args + 1]) * nrowsPack], &bufferT_[(nColsStagesAcc[args_d[i_args + 2]] + args_d[i_args + 3]) * nrowsPack]);
            i_args += 4;
            break;
        }
        case 1: {
            // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: commit1
            Goldilocks::op_pack(nrowsPack, args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args_d[i_args + 3]] + args_d[i_args + 4]) * nrowsPack], &bufferT_[(nColsStagesAcc[args_d[i_args + 5]] + args_d[i_args + 6]) * nrowsPack]);
            i_args += 7;
            break;
        }
        case 2: {
            // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: tmp1
            Goldilocks::op_pack(nrowsPack, args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args_d[i_args + 3]] + args_d[i_args + 4]) * nrowsPack], &tmp1[args_d[i_args + 5] * nrowsPack]);
            i_args += 6;
            break;
        }
        case 3: {
            // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: public
            Goldilocks::op_pack(nrowsPack, args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args_d[i_args + 3]] + args_d[i_args + 4]) * nrowsPack], &publics_d[args_d[i_args + 5] * nrowsPack]);
            i_args += 6;
            break;
        }
        case 4: {
            // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: number
            Goldilocks::op_pack(nrowsPack, args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args_d[i_args + 3]] + args_d[i_args + 4]) * nrowsPack], &numbers_d[args_d[i_args + 5]*nrowsPack]);
            i_args += 6;
            break;
        }
        case 5: {
            // COPY tmp1 to commit1
            Goldilocks::copy_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args_d[i_args]] + args_d[i_args + 1]) * nrowsPack], &tmp1[args_d[i_args + 2] * nrowsPack]);
            i_args += 3;
            break;
        }
        case 6: {
            // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: tmp1
            Goldilocks::op_pack(nrowsPack, args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * nrowsPack], &tmp1[args_d[i_args + 3] * nrowsPack], &tmp1[args_d[i_args + 4] * nrowsPack]);
            i_args += 5;
            break;
        }
        case 7: {
            // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: public
            Goldilocks::op_pack(nrowsPack, args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * nrowsPack], &tmp1[args_d[i_args + 3] * nrowsPack], &publics_d[args_d[i_args + 4] * nrowsPack]);
            i_args += 5;
            break;
        }
        case 8: {
            // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: number
            Goldilocks::op_pack(nrowsPack, args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * nrowsPack], &tmp1[args_d[i_args + 3] * nrowsPack], &numbers_d[args_d[i_args + 4]*nrowsPack]);
            i_args += 5;
            break;
        }
        case 9: {
            // COPY public to commit1
            Goldilocks::copy_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args_d[i_args]] + args_d[i_args + 1]) * nrowsPack], &publics_d[args_d[i_args + 2] * nrowsPack]);
            i_args += 3;
            break;
        }
        case 10: {
            // OPERATION WITH DEST: commit1 - SRC0: public - SRC1: public
            Goldilocks::op_pack(nrowsPack, args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * nrowsPack], &publics_d[args_d[i_args + 3] * nrowsPack], &publics_d[args_d[i_args + 4] * nrowsPack]);
            i_args += 5;
            break;
        }
        case 11: {
            // OPERATION WITH DEST: commit1 - SRC0: public - SRC1: number
            Goldilocks::op_pack(nrowsPack, args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * nrowsPack], &publics_d[args_d[i_args + 3] * nrowsPack], &numbers_d[args_d[i_args + 4]*nrowsPack]);
            i_args += 5;
            break;
        }
        case 12: {
            // COPY number to commit1
            Goldilocks::copy_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args_d[i_args]] + args_d[i_args + 1]) * nrowsPack], &numbers_d[args_d[i_args + 2]*nrowsPack]);
            i_args += 3;
            break;
        }
        case 13: {
            // OPERATION WITH DEST: commit1 - SRC0: number - SRC1: number
            Goldilocks::op_pack(nrowsPack, args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * nrowsPack], &numbers_d[args_d[i_args + 3]*nrowsPack], &numbers_d[args_d[i_args + 4]*nrowsPack]);
            i_args += 5;
            break;
        }
        case 14: {
            // COPY commit1 to tmp1
            Goldilocks::copy_pack(nrowsPack, &tmp1[args_d[i_args] * nrowsPack], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * nrowsPack]);
            i_args += 3;
            break;
        }
        case 15: {
            // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: commit1
            Goldilocks::op_pack(nrowsPack, args_d[i_args], &tmp1[args_d[i_args + 1] * nrowsPack], &bufferT_[(nColsStagesAcc[args_d[i_args + 2]] + args_d[i_args + 3]) * nrowsPack], &bufferT_[(nColsStagesAcc[args_d[i_args + 4]] + args_d[i_args + 5]) * nrowsPack]);
            i_args += 6;
            break;
        }
        case 16: {
            // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: tmp1
            Goldilocks::op_pack(nrowsPack, args_d[i_args], &tmp1[args_d[i_args + 1] * nrowsPack], &bufferT_[(nColsStagesAcc[args_d[i_args + 2]] + args_d[i_args + 3]) * nrowsPack], &tmp1[args_d[i_args + 4] * nrowsPack]);
            i_args += 5;
            break;
        }
        case 17: {
            // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: public
            Goldilocks::op_pack(nrowsPack, args_d[i_args], &tmp1[args_d[i_args + 1] * nrowsPack], &bufferT_[(nColsStagesAcc[args_d[i_args + 2]] + args_d[i_args + 3]) * nrowsPack], &publics_d[args_d[i_args + 4] * nrowsPack]);
            i_args += 5;
            break;
        }
        case 18: {
            // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: number
            Goldilocks::op_pack(nrowsPack, args_d[i_args], &tmp1[args_d[i_args + 1] * nrowsPack], &bufferT_[(nColsStagesAcc[args_d[i_args + 2]] + args_d[i_args + 3]) * nrowsPack], &numbers_d[args_d[i_args + 4]*nrowsPack]);
            i_args += 5;
            break;
        }
        case 19: {
            // COPY tmp1 to tmp1
            Goldilocks::copy_pack(nrowsPack, &tmp1[args_d[i_args] * nrowsPack], &tmp1[args_d[i_args + 1] * nrowsPack]);
            i_args += 2;
            break;
        }
        case 20: {
            // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: tmp1
            Goldilocks::op_pack(nrowsPack, args_d[i_args], &tmp1[args_d[i_args + 1] * nrowsPack], &tmp1[args_d[i_args + 2] * nrowsPack], &tmp1[args_d[i_args + 3] * nrowsPack]);
            i_args += 4;
            break;
        }
        case 21: {
            // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: public
            Goldilocks::op_pack(nrowsPack, args_d[i_args], &tmp1[args_d[i_args + 1] * nrowsPack], &tmp1[args_d[i_args + 2] * nrowsPack], &publics_d[args_d[i_args + 3] * nrowsPack]);
            i_args += 4;
            break;
        }
        case 22: {
            // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: number
            Goldilocks::op_pack(nrowsPack, args_d[i_args], &tmp1[args_d[i_args + 1] * nrowsPack], &tmp1[args_d[i_args + 2] * nrowsPack], &numbers_d[args_d[i_args + 3]*nrowsPack]);
            i_args += 4;
            break;
        }
        case 23: {
            // COPY public to tmp1
            Goldilocks::copy_pack(nrowsPack, &tmp1[args_d[i_args] * nrowsPack], &publics_d[args_d[i_args + 1] * nrowsPack]);
            i_args += 2;
            break;
        }
        case 24: {
            // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: public
            Goldilocks::op_pack(nrowsPack, args_d[i_args], &tmp1[args_d[i_args + 1] * nrowsPack], &publics_d[args_d[i_args + 2] * nrowsPack], &publics_d[args_d[i_args + 3] * nrowsPack]);
            i_args += 4;
            break;
        }
        case 25: {
            // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: number
            Goldilocks::op_pack(nrowsPack, args_d[i_args], &tmp1[args_d[i_args + 1] * nrowsPack], &publics_d[args_d[i_args + 2] * nrowsPack], &numbers_d[args_d[i_args + 3]*nrowsPack]);
            i_args += 4;
            break;
        }
        case 26: {
            // COPY number to tmp1
            Goldilocks::copy_pack(nrowsPack, &tmp1[args_d[i_args] * nrowsPack], &numbers_d[args_d[i_args + 1]*nrowsPack]);
            i_args += 2;
            break;
        }
        case 27: {
            // OPERATION WITH DEST: tmp1 - SRC0: number - SRC1: number
            Goldilocks::op_pack(nrowsPack, args_d[i_args], &tmp1[args_d[i_args + 1] * nrowsPack], &numbers_d[args_d[i_args + 2]*nrowsPack], &numbers_d[args_d[i_args + 3]*nrowsPack]);
            i_args += 4;
            break;
        }
        case 28: {
            // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: commit1
            Goldilocks3::op_31_pack(nrowsPack, args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args_d[i_args + 3]] + args_d[i_args + 4]) * nrowsPack], &bufferT_[(nColsStagesAcc[args_d[i_args + 5]] + args_d[i_args + 6]) * nrowsPack]);
            i_args += 7;
            break;
        }
        case 29: {
            // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: tmp1
            Goldilocks3::op_31_pack(nrowsPack, args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args_d[i_args + 3]] + args_d[i_args + 4]) * nrowsPack], &tmp1[args_d[i_args + 5] * nrowsPack]);
            i_args += 6;
            break;
        }
        case 30: {
            // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: public
            Goldilocks3::op_31_pack(nrowsPack, args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args_d[i_args + 3]] + args_d[i_args + 4]) * nrowsPack], &publics_d[args_d[i_args + 5] * nrowsPack]);
            i_args += 6;
            break;
        }
        case 31: {
            // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: number
            Goldilocks3::op_31_pack(nrowsPack, args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args_d[i_args + 3]] + args_d[i_args + 4]) * nrowsPack], &numbers_d[args_d[i_args + 5]*nrowsPack]);
            i_args += 6;
            break;
        }
        case 32: {
            // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: commit1
            Goldilocks3::op_31_pack(nrowsPack, args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * nrowsPack], &tmp3[args_d[i_args + 3] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args_d[i_args + 4]] + args_d[i_args + 5]) * nrowsPack]);
            i_args += 6;
            break;
        }
        case 33: {
            // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: tmp1
            Goldilocks3::op_31_pack(nrowsPack, args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * nrowsPack], &tmp3[args_d[i_args + 3] * nrowsPack * FIELD_EXTENSION], &tmp1[args_d[i_args + 4] * nrowsPack]);
            i_args += 5;
            break;
        }
        case 34: {
            // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: public
            Goldilocks3::op_31_pack(nrowsPack, args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * nrowsPack], &tmp3[args_d[i_args + 3] * nrowsPack * FIELD_EXTENSION], &publics_d[args_d[i_args + 4] * nrowsPack]);
            i_args += 5;
            break;
        }
        case 35: {
            // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: number
            Goldilocks3::op_31_pack(nrowsPack, args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * nrowsPack], &tmp3[args_d[i_args + 3] * nrowsPack * FIELD_EXTENSION], &numbers_d[args_d[i_args + 4]*nrowsPack]);
            i_args += 5;
            break;
        }
        case 36: {
            // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: commit1
            Goldilocks3::op_31_pack(nrowsPack, args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * nrowsPack], &challenges_d[args_d[i_args + 3]*FIELD_EXTENSION*nrowsPack], &bufferT_[(nColsStagesAcc[args_d[i_args + 4]] + args_d[i_args + 5]) * nrowsPack]);
            i_args += 6;
            break;
        }
        case 37: {
            // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: tmp1
            Goldilocks3::op_31_pack(nrowsPack, args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * nrowsPack], &challenges_d[args_d[i_args + 3]*FIELD_EXTENSION*nrowsPack], &tmp1[args_d[i_args + 4] * nrowsPack]);
            i_args += 5;
            break;
        }
        case 38: {
            // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: public
            Goldilocks3::op_31_pack(nrowsPack, args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * nrowsPack], &challenges_d[args_d[i_args + 3]*FIELD_EXTENSION*nrowsPack], &publics_d[args_d[i_args + 4] * nrowsPack]);
            i_args += 5;
            break;
        }
        case 39: {
            // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: number
            Goldilocks3::op_31_pack(nrowsPack, args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * nrowsPack], &challenges_d[args_d[i_args + 3]*FIELD_EXTENSION*nrowsPack], &numbers_d[args_d[i_args + 4]*nrowsPack]);
            i_args += 5;
            break;
        }
        case 40: {
            // COPY commit3 to commit3
            Goldilocks3::copy_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args_d[i_args]] + args_d[i_args + 1]) * nrowsPack], &bufferT_[(nColsStagesAcc[args_d[i_args + 2]] + args_d[i_args + 3]) * nrowsPack]);
            i_args += 4;
            break;
        }
        case 41: {
            // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: commit3
            Goldilocks3::op_pack(nrowsPack, args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args_d[i_args + 3]] + args_d[i_args + 4]) * nrowsPack], &bufferT_[(nColsStagesAcc[args_d[i_args + 5]] + args_d[i_args + 6]) * nrowsPack]);
            i_args += 7;
            break;
        }
        case 42: {
            // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: tmp3
            Goldilocks3::op_pack(nrowsPack, args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args_d[i_args + 3]] + args_d[i_args + 4]) * nrowsPack], &tmp3[args_d[i_args + 5] * nrowsPack * FIELD_EXTENSION]);
            i_args += 6;
            break;
        }
        case 43: {
            // MULTIPLICATION WITH DEST: commit3 - SRC0: commit3 - SRC1: challenge
            Goldilocks3::mul_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args_d[i_args + 3]] + args_d[i_args + 4]) * nrowsPack], &challenges_d[args_d[i_args + 5]*FIELD_EXTENSION*nrowsPack], &challenges_ops_d[args_d[i_args + 5]*FIELD_EXTENSION*nrowsPack]);
            i_args += 6;
            break;
        }
        case 44: {
            // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: challenge
            Goldilocks3::op_pack(nrowsPack, args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args_d[i_args + 3]] + args_d[i_args + 4]) * nrowsPack], &challenges_d[args_d[i_args + 5]*FIELD_EXTENSION*nrowsPack]);
            i_args += 6;
            break;
        }
        case 45: {
            // COPY tmp3 to commit3
            Goldilocks3::copy_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args_d[i_args]] + args_d[i_args + 1]) * nrowsPack], &tmp3[args_d[i_args + 2] * nrowsPack * FIELD_EXTENSION]);
            i_args += 3;
            break;
        }
        case 46: {
            // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: tmp3
            Goldilocks3::op_pack(nrowsPack, args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * nrowsPack], &tmp3[args_d[i_args + 3] * nrowsPack * FIELD_EXTENSION], &tmp3[args_d[i_args + 4] * nrowsPack * FIELD_EXTENSION]);
            i_args += 5;
            break;
        }
        case 47: {
            // MULTIPLICATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: challenge
            Goldilocks3::mul_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * nrowsPack], &tmp3[args_d[i_args + 3] * nrowsPack * FIELD_EXTENSION], &challenges_d[args_d[i_args + 4]*FIELD_EXTENSION*nrowsPack], &challenges_ops_d[args_d[i_args + 4]*FIELD_EXTENSION*nrowsPack]);
            i_args += 5;
            break;
        }
        case 48: {
            // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: challenge
            Goldilocks3::op_pack(nrowsPack, args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * nrowsPack], &tmp3[args_d[i_args + 3] * nrowsPack * FIELD_EXTENSION], &challenges_d[args_d[i_args + 4]*FIELD_EXTENSION*nrowsPack]);
            i_args += 5;
            break;
        }
        case 49: {
            // MULTIPLICATION WITH DEST: commit3 - SRC0: challenge - SRC1: challenge
            Goldilocks3::mul_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * nrowsPack], &challenges_d[args_d[i_args + 3]*FIELD_EXTENSION*nrowsPack], &challenges_d[args_d[i_args + 4]*FIELD_EXTENSION*nrowsPack], &challenges_ops_d[args_d[i_args + 4]*FIELD_EXTENSION*nrowsPack]);
            i_args += 5;
            break;
        }
        case 50: {
            // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: challenge
            Goldilocks3::op_pack(nrowsPack, args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * nrowsPack], &challenges_d[args_d[i_args + 3]*FIELD_EXTENSION*nrowsPack], &challenges_d[args_d[i_args + 4]*FIELD_EXTENSION*nrowsPack]);
            i_args += 5;
            break;
        }
        case 51: {
            // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit1
            Goldilocks3::op_31_pack(nrowsPack, args_d[i_args], &tmp3[args_d[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args_d[i_args + 2]] + args_d[i_args + 3]) * nrowsPack], &bufferT_[(nColsStagesAcc[args_d[i_args + 4]] + args_d[i_args + 5]) * nrowsPack]);
            i_args += 6;
            break;
        }
        case 52: {
            // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp1
            Goldilocks3::op_31_pack(nrowsPack, args_d[i_args], &tmp3[args_d[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args_d[i_args + 2]] + args_d[i_args + 3]) * nrowsPack], &tmp1[args_d[i_args + 4] * nrowsPack]);
            i_args += 5;
            break;
        }
        case 53: {
            // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: public
            Goldilocks3::op_31_pack(nrowsPack, args_d[i_args], &tmp3[args_d[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args_d[i_args + 2]] + args_d[i_args + 3]) * nrowsPack], &publics_d[args_d[i_args + 4] * nrowsPack]);
            i_args += 5;
            break;
        }
        case 54: {
            // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: number
            Goldilocks3::op_31_pack(nrowsPack, args_d[i_args], &tmp3[args_d[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args_d[i_args + 2]] + args_d[i_args + 3]) * nrowsPack], &numbers_d[args_d[i_args + 4]*nrowsPack]);
            i_args += 5;
            break;
        }
        case 55: {
            // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: commit1
            Goldilocks3::op_31_pack(nrowsPack, args_d[i_args], &tmp3[args_d[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args_d[i_args + 2] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args_d[i_args + 3]] + args_d[i_args + 4]) * nrowsPack]);
            i_args += 5;
            break;
        }
        case 56: {
            // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp1
            Goldilocks3::op_31_pack(nrowsPack, args_d[i_args], &tmp3[args_d[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args_d[i_args + 2] * nrowsPack * FIELD_EXTENSION], &tmp1[args_d[i_args + 3] * nrowsPack]);
            i_args += 4;
            break;
        }
        case 57: {
            // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: public
            Goldilocks3::op_31_pack(nrowsPack, args_d[i_args], &tmp3[args_d[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args_d[i_args + 2] * nrowsPack * FIELD_EXTENSION], &publics_d[args_d[i_args + 3] * nrowsPack]);
            i_args += 4;
            break;
        }
        case 58: {
            // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: number
            Goldilocks3::op_31_pack(nrowsPack, args_d[i_args], &tmp3[args_d[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args_d[i_args + 2] * nrowsPack * FIELD_EXTENSION], &numbers_d[args_d[i_args + 3]*nrowsPack]);
            i_args += 4;
            break;
        }
        case 59: {
            // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: commit1
            Goldilocks3::op_31_pack(nrowsPack, args_d[i_args], &tmp3[args_d[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges_d[args_d[i_args + 2]*FIELD_EXTENSION*nrowsPack], &bufferT_[(nColsStagesAcc[args_d[i_args + 3]] + args_d[i_args + 4]) * nrowsPack]);
            i_args += 5;
            break;
        }
        case 60: {
            // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: tmp1
            Goldilocks3::op_31_pack(nrowsPack, args_d[i_args], &tmp3[args_d[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges_d[args_d[i_args + 2]*FIELD_EXTENSION*nrowsPack], &tmp1[args_d[i_args + 3] * nrowsPack]);
            i_args += 4;
            break;
        }
        case 61: {
            // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: public
            Goldilocks3::op_31_pack(nrowsPack, args_d[i_args], &tmp3[args_d[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges_d[args_d[i_args + 2]*FIELD_EXTENSION*nrowsPack], &publics_d[args_d[i_args + 3] * nrowsPack]);
            i_args += 4;
            break;
        }
        case 62: {
            // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: number
            Goldilocks3::op_31_pack(nrowsPack, args_d[i_args], &tmp3[args_d[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges_d[args_d[i_args + 2]*FIELD_EXTENSION*nrowsPack], &numbers_d[args_d[i_args + 3]*nrowsPack]);
            i_args += 4;
            break;
        }
        case 63: {
            // COPY commit3 to tmp3
            Goldilocks3::copy_pack(nrowsPack, &tmp3[args_d[i_args] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * nrowsPack]);
            i_args += 3;
            break;
        }
        case 64: {
            // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit3
            Goldilocks3::op_pack(nrowsPack, args_d[i_args], &tmp3[args_d[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args_d[i_args + 2]] + args_d[i_args + 3]) * nrowsPack], &bufferT_[(nColsStagesAcc[args_d[i_args + 4]] + args_d[i_args + 5]) * nrowsPack]);
            i_args += 6;
            break;
        }
        case 65: {
            // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp3
            Goldilocks3::op_pack(nrowsPack, args_d[i_args], &tmp3[args_d[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args_d[i_args + 2]] + args_d[i_args + 3]) * nrowsPack], &tmp3[args_d[i_args + 4] * nrowsPack * FIELD_EXTENSION]);
            i_args += 5;
            break;
        }
        case 66: {
            // MULTIPLICATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: challenge
            Goldilocks3::mul_pack(nrowsPack, &tmp3[args_d[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args_d[i_args + 2]] + args_d[i_args + 3]) * nrowsPack], &challenges_d[args_d[i_args + 4]*FIELD_EXTENSION*nrowsPack], &challenges_ops_d[args_d[i_args + 4]*FIELD_EXTENSION*nrowsPack]);
            i_args += 5;
            break;
        }
        case 67: {
            // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: challenge
            Goldilocks3::op_pack(nrowsPack, args_d[i_args], &tmp3[args_d[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args_d[i_args + 2]] + args_d[i_args + 3]) * nrowsPack], &challenges_d[args_d[i_args + 4]*FIELD_EXTENSION*nrowsPack]);
            i_args += 5;
            break;
        }
        case 68: {
            // COPY tmp3 to tmp3
            Goldilocks3::copy_pack(nrowsPack, &tmp3[args_d[i_args] * nrowsPack * FIELD_EXTENSION], &tmp3[args_d[i_args + 1] * nrowsPack * FIELD_EXTENSION]);
            i_args += 2;
            break;
        }
        case 69: {
            // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp3
            Goldilocks3::op_pack(nrowsPack, args_d[i_args], &tmp3[args_d[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args_d[i_args + 2] * nrowsPack * FIELD_EXTENSION], &tmp3[args_d[i_args + 3] * nrowsPack * FIELD_EXTENSION]);
            i_args += 4;
            break;
        }
        case 70: {
            // MULTIPLICATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: challenge
            Goldilocks3::mul_pack(nrowsPack, &tmp3[args_d[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args_d[i_args + 2] * nrowsPack * FIELD_EXTENSION], &challenges_d[args_d[i_args + 3]*FIELD_EXTENSION*nrowsPack], &challenges_ops_d[args_d[i_args + 3]*FIELD_EXTENSION*nrowsPack]);
            i_args += 4;
            break;
        }
        case 71: {
            // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: challenge
            Goldilocks3::op_pack(nrowsPack, args_d[i_args], &tmp3[args_d[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args_d[i_args + 2] * nrowsPack * FIELD_EXTENSION], &challenges_d[args_d[i_args + 3]*FIELD_EXTENSION*nrowsPack]);
            i_args += 4;
            break;
        }
        case 72: {
            // MULTIPLICATION WITH DEST: tmp3 - SRC0: challenge - SRC1: challenge
            Goldilocks3::mul_pack(nrowsPack, &tmp3[args_d[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges_d[args_d[i_args + 2]*FIELD_EXTENSION*nrowsPack], &challenges_d[args_d[i_args + 3]*FIELD_EXTENSION*nrowsPack], &challenges_ops_d[args_d[i_args + 3]*FIELD_EXTENSION*nrowsPack]);
            i_args += 4;
            break;
        }
        case 73: {
            // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: challenge
            Goldilocks3::op_pack(nrowsPack, args_d[i_args], &tmp3[args_d[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges_d[args_d[i_args + 2]*FIELD_EXTENSION*nrowsPack], &challenges_d[args_d[i_args + 3]*FIELD_EXTENSION*nrowsPack]);
            i_args += 4;
            break;
        }
        case 74: {
            // COPY eval to tmp3
            Goldilocks3::copy_pack(nrowsPack, &tmp3[args_d[i_args] * nrowsPack * FIELD_EXTENSION], &evals_d[args_d[i_args + 1]*FIELD_EXTENSION*nrowsPack]);
            i_args += 2;
            break;
        }
        case 75: {
            // MULTIPLICATION WITH DEST: tmp3 - SRC0: eval - SRC1: challenge
            Goldilocks3::mul_pack(nrowsPack, &tmp3[args_d[i_args + 1] * nrowsPack * FIELD_EXTENSION], &evals_d[args_d[i_args + 2]*FIELD_EXTENSION*nrowsPack], &challenges_d[args_d[i_args + 3]*FIELD_EXTENSION*nrowsPack], &challenges_ops_d[args_d[i_args + 3]*FIELD_EXTENSION*nrowsPack]);
            i_args += 4;
            break;
        }
        case 76: {
            // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: eval
            Goldilocks3::op_pack(nrowsPack, args_d[i_args], &tmp3[args_d[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges_d[args_d[i_args + 2]*FIELD_EXTENSION*nrowsPack], &evals_d[args_d[i_args + 3]*FIELD_EXTENSION*nrowsPack]);
            i_args += 4;
            break;
        }
        case 77: {
            // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: eval
            Goldilocks3::op_pack(nrowsPack, args_d[i_args], &tmp3[args_d[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args_d[i_args + 2] * nrowsPack * FIELD_EXTENSION], &evals_d[args_d[i_args + 3]*FIELD_EXTENSION*nrowsPack]);
            i_args += 4;
            break;
        }
        case 78: {
            // OPERATION WITH DEST: tmp3 - SRC0: eval - SRC1: commit1
            Goldilocks3::op_31_pack(nrowsPack, args_d[i_args], &tmp3[args_d[i_args + 1] * nrowsPack * FIELD_EXTENSION], &evals_d[args_d[i_args + 2]*FIELD_EXTENSION*nrowsPack], &bufferT_[(nColsStagesAcc[args_d[i_args + 3]] + args_d[i_args + 4]) * nrowsPack]);
            i_args += 5;
            break;
        }
        case 79: {
            // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: eval
            Goldilocks3::op_pack(nrowsPack, args_d[i_args], &tmp3[args_d[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args_d[i_args + 2]] + args_d[i_args + 3]) * nrowsPack], &evals_d[args_d[i_args + 4]*FIELD_EXTENSION*nrowsPack]);
            i_args += 5;
            break;
        }
            default: {
                std::cout << " Wrong operation!" << std::endl;
                exit(1);
            }
        }
    }
    assert(i_args == nArgs);

}

#ifdef __USE_CUDA__
__device__  void optcodeIteration_(uint32_t nOps, uint32_t nArgs);
#endif

void dataSetup(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams){

        uint32_t nrowsPack =  4;

        /*
            non-buffered data
        */
        uint8_t *ops = &parserArgs.ops[parserParams.opsOffset];
        ops_d = new uint32_t[parserParams.nOps];
        for(uint64_t i = 0; i < parserParams.nOps; ++i) ops_d[i] = uint32_t(ops[i]);
        
        uint16_t *args = &parserArgs.args[parserParams.argsOffset];
        args_d = new uint32_t[parserParams.nArgs];
        for(uint64_t i = 0; i < parserParams.nArgs; ++i) args_d[i] = uint32_t(args[i]);

        uint8_t *storePol = &parserArgs.storePols[parserParams.storePolsOffset];
        storePol_d = new uint32_t[parserParams.nStorePols];
        for(uint64_t i = 0; i < parserParams.nStorePols; ++i) storePol_d[i] = uint32_t(storePol[i]);

        Goldilocks::Element challenges[params.challenges.degree()*FIELD_EXTENSION*nrowsPack];
        Goldilocks::Element challenges_ops[params.challenges.degree()*FIELD_EXTENSION*nrowsPack];
        for(uint64_t i = 0; i < params.challenges.degree(); ++i) {
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                challenges[(i*FIELD_EXTENSION)*nrowsPack + j] = params.challenges[i][0];
                challenges[(i*FIELD_EXTENSION + 1)*nrowsPack + j] = params.challenges[i][1];
                challenges[(i*FIELD_EXTENSION + 2)*nrowsPack + j] = params.challenges[i][2];
                challenges_ops[(i*FIELD_EXTENSION)*nrowsPack + j] = params.challenges[i][0] + params.challenges[i][1];
                challenges_ops[(i*FIELD_EXTENSION + 1)*nrowsPack + j] = params.challenges[i][0] + params.challenges[i][2];
                challenges_ops[(i*FIELD_EXTENSION + 2)*nrowsPack + j] = params.challenges[i][1] + params.challenges[i][2];
            }
        }
        challenges_d = new Goldilocks::Element[params.challenges.degree()*FIELD_EXTENSION*nrowsPack];
        memcpy(challenges_d, challenges, params.challenges.degree()*FIELD_EXTENSION*nrowsPack*sizeof(Goldilocks::Element));
        challenges_ops_d = new Goldilocks::Element[params.challenges.degree()*FIELD_EXTENSION*nrowsPack];
        memcpy(challenges_ops_d, challenges_ops, params.challenges.degree()*FIELD_EXTENSION*nrowsPack*sizeof(Goldilocks::Element));
        
        uint64_t *numbers = &parserArgs.numbers[parserParams.numbersOffset];
        Goldilocks::Element numbers_[parserParams.nNumbers*nrowsPack];
        for(uint64_t i = 0; i < parserParams.nNumbers; ++i) {
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                numbers_[i*nrowsPack + j] = Goldilocks::fromU64(numbers[i]);
            }
        }
        numbers_d = new Goldilocks::Element[parserParams.nNumbers*nrowsPack];
        memcpy(numbers_d, numbers_, parserParams.nNumbers*nrowsPack*sizeof(Goldilocks::Element));

        Goldilocks::Element publics[starkInfo.nPublics*nrowsPack];
        for(uint64_t i = 0; i < starkInfo.nPublics; ++i) {
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                publics[i*nrowsPack + j] = params.publicInputs[i];
            }
        }
        publics_d = new Goldilocks::Element[starkInfo.nPublics*nrowsPack];
        memcpy(publics_d, publics, starkInfo.nPublics*nrowsPack*sizeof(Goldilocks::Element));

        Goldilocks::Element evals[params.evals.degree()*FIELD_EXTENSION*nrowsPack];
        for(uint64_t i = 0; i < params.evals.degree(); ++i) {
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                evals[(i*FIELD_EXTENSION)*nrowsPack + j] = params.evals[i][0];
                evals[(i*FIELD_EXTENSION + 1)*nrowsPack + j] = params.evals[i][1];
                evals[(i*FIELD_EXTENSION + 2)*nrowsPack + j] = params.evals[i][2];
            }
        }
        evals_d = new Goldilocks::Element[params.evals.degree()*FIELD_EXTENSION*nrowsPack];
        memcpy(evals_d, evals, params.evals.degree()*FIELD_EXTENSION*nrowsPack*sizeof(Goldilocks::Element));

        /* 
            buffered data
        */
        constPols_d = new Goldilocks::Element[params.pConstPols->numPols()*params.pConstPols->degree()];
        memcpy(constPols_d, params.pConstPols->address(), params.pConstPols->size());

        constPols2ns_d = new Goldilocks::Element[params.pConstPols2ns->numPols()*params.pConstPols2ns->degree()];
        memcpy(constPols2ns_d, params.pConstPols2ns->address(), params.pConstPols2ns->size());

        x_d = new Goldilocks::Element[params.x_n.dim()*params.x_n.degree()];
        memcpy(x_d, params.x_n.address(), params.x_n.size());

        x_2ns_d = new Goldilocks::Element[params.x_2ns.dim()*params.x_2ns.degree()];
        memcpy(x_2ns_d, params.x_2ns.address(), params.x_2ns.size());
        
        zi_d = new Goldilocks::Element[params.zi.dim()*params.zi.degree()];
        memcpy(zi_d, params.zi.address(), params.zi.size());

        xDivXSubXi_d = new Goldilocks::Element[params.xDivXSubXi.dim()*params.xDivXSubXi.degree()];
        memcpy(xDivXSubXi_d, params.xDivXSubXi.address(), params.xDivXSubXi.size());
        
        pols_d = new Goldilocks::Element[starkInfo.mapTotalN];
        memcpy(pols_d, params.pols, starkInfo.mapTotalN*sizeof(Goldilocks::Element));
        /*
            temporal buffers
        */
        numBlocks = omp_get_max_threads();
        bufferT_d = new Goldilocks::Element* [numBlocks];
        for(uint64_t i = 0; i < numBlocks; ++i) {
            bufferT_d[i] = new Goldilocks::Element[2*nCols*nrowsPack];
        }
        tmp1_d = new Goldilocks::Element*[numBlocks];
        for(uint64_t i = 0; i < numBlocks; ++i) {
            tmp1_d[i] = new Goldilocks::Element[parserParams.nTemp1*nrowsPack];
        }
        tmp3_d = new Goldilocks::Element*[numBlocks];
        for(uint64_t i = 0; i < numBlocks; ++i) {
            tmp3_d[i] = new Goldilocks::Element[parserParams.nTemp3*nrowsPack*FIELD_EXTENSION];
        }
    }    

#ifdef __USE_CUDA__
void dataSetup_(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams);
#endif

};

#ifdef __USE_CUDA__     
__global__ void blockCalculation(CHelpersStepsGPU* chelpers_,StarkInfo &starkInfo, StepsParams &params, uint64_t domainSize, bool domainExtended, uint64_t stage, uint32_t nOps, uint32_t nArgs);
#endif

#endif