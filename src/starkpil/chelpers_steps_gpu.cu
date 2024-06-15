#include "chelpers_steps_gpu.hpp"

__device__  void CHelpersStepsGPU::storePolinomials_(StarkInfo &starkInfo, StepsParams &params, uint64_t row, uint64_t domainExtended) {

        /*bufferT_ = bufferT_d[blockIdx.x];
        if(domainExtended) {
            // Store either polinomial f or polinomial q
            for(uint64_t k = 0; k < nColsStages[10]; ++k) {
                Goldilocks::Element *buffT = &bufferT_[(nColsStagesAcc[10] + k)* nrowsPack];
                Goldilocks::copy_pack( &pols_d[offsetsStages[10] + k + row * nColsStages[10]], nColsStages[10], buffT);
            }
        } else {
            uint64_t nStages = 3;
            uint64_t domainSize =  1 << starkInfo.starkStruct.nBits;
            for(uint64_t s = 2; s <= nStages + 1; ++s) {
                bool isTmpPol =  s == 4;
                for(uint64_t k = 0; k < nColsStages[s]; ++k) {
                    uint64_t dim = storePol[nColsStagesAcc[s] + k];
                    if(storePol[nColsStagesAcc[s] + k]) {
                        Goldilocks::Element *buffT = &bufferT_[(nColsStagesAcc[s] + k)* nrowsPack];
                        if(isTmpPol) {
                            for(uint64_t i = 0; i < dim; ++i) {
                                Goldilocks::copy_pack(&params.pols[offsetsStages[s] + k * domainSize + row * dim + i], uint64_t(dim), &buffT[i*nrowsPack]);
                            }
                        } else {
                            Goldilocks::copy_pack(&params.pols[offsetsStages[s] + k + row * nColsStages[s]], nColsStages[s], buffT);
                        }
                    }
                }
            }
        }*/
    }



__device__   void CHelpersStepsGPU::loadPolinomials_(StarkInfo &starkInfo, StepsParams &params, uint64_t row, uint64_t stage, uint64_t domainExtended) {
    /*
        // buffered data
        Goldilocks::Element *constPols_aux = domainExtended ? constPols2ns_d : constPols_d;
        uint64_t domainSize = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
        Goldilocks::Element *x_aux = domainExtended ? x_2ns_d : x_d;

        uint64_t nStages = 3;
        uint64_t nextStride = domainExtended ?  1 << (starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits) : 1;
        std::vector<uint64_t> nextStrides = {0, nextStride};
        for(uint64_t k = 0; k < starkInfo.nConstants; ++k) {
            for(uint64_t o = 0; o < 2; ++o) {
                uint64_t l = (row + threadIdx.x + nextStrides[o]) % domainSize;
                bufferT_[(nColsStagesAcc[5*o] + k)*nrowsPack + threadIdx.x] = constPols_aux[l * starkInfo.nConstants + k];
            }
        }

        // Load x and Zi
        bufferT_[starkInfo.nConstants*nrowsPack + threadIdx.x] = x_aux[row + threadIdx.x];
        bufferT_[(starkInfo.nConstants + 1)*nrowsPack + threadIdx.x] = zi_d[row + threadIdx.x];
        

        for(uint64_t s = 1; s <= nStages; ++s) {
            for(uint64_t k = 0; k < nColsStages[s]; ++k) {
                for(uint64_t o = 0; o < 2; ++o) {
                        uint64_t l = (row + j + nextStrides[o]) % domainSize;
                        bufferT_[(nColsStagesAcc[5*o + s] + k)*nrowsPack + threadIdx.x] = pols_d[offsetsStages[s] + l * nColsStages[s] + k];
                }
            }
        }

        if(stage == 5) {
            for(uint64_t k = 0; k < nColsStages[nStages + 1]; ++k) {
                for(uint64_t o = 0; o < 2; ++o) {
                    uint64_t l = (row + j + nextStrides[o]) % domainSize;
                    bufferT_[(nColsStagesAcc[5*o + nStages + 1] + k)*nrowsPack + threadIdx.x] = pols_d[offsetsStages[nStages + 1] + l * nColsStages[nStages + 1] + k];
                }
            }

           // Load xDivXSubXi & xDivXSubWXi
           for(uint64_t d = 0; d < 2; ++d) {
               for(uint64_t i = 0; i < FIELD_EXTENSION; ++i) {
                    bufferT_[(nColsStagesAcc[11] + FIELD_EXTENSION*d + i)*nrowsPack + threadIdx.x] = xDivXSubXi_d[(d*domainSize + row + j)*FIELD_EXTENSION+i];SSSSS                   
               }
           }
        }
        */
    }

 void CHelpersStepsGPU::calculateExpressions_(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams)  {
    bool domainExtended = parserParams.stage > 3 ? true : false;
    uint64_t domainSize = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
    

    /*setBufferTInfo(starkInfo, parserParams.stage);
    dataSetup(starkInfo, params, parserArgs, parserParams);
    blockCalculation<<<numBlocks, 256>>>(starkInfo,params, domainSize, dominExtended, stage, nOps, nArgs);
    */
}

__global__ void blockCalculation(StarkInfo &starkInfo, StepsParams &params, uint64_t domainSize, bool domainExtended, uint64_t stage, uint32_t nOps, uint32_t nArgs) {
    uint32_t i = threadIdx.x;
    /*while( i < domainSize) {
        loadPolinomials(starkInfo, params, i, stage, nrowsPack, domainExtended);
        optcodeIteration(nrowsPack,  nOps, nArgs);
        storePolinomials(starkInfo, params, storePol_d, i, nrowsPack, domainExtended);
        i += blockDim.x;
    }*/
}


    __device__  void CHelpersStepsGPU::optcodeIteration_(uint32_t nOps, uint32_t nArgs) {

    uint64_t i_args = 0;
    Goldilocks::Element *bufferT_ = bufferT_d[blockIdx.x];
    Goldilocks::Element *tmp1 = tmp1_d[blockIdx.x];
    Goldilocks::Element *tmp3 = tmp3_d[blockIdx.x];
#if 0
    for (uint64_t kk = 0; kk < nOps; ++kk) {
        
        switch (ops_d[kk]) {
            case 0: {
                // COPY commit1 to commit1
                Goldilocks::copy_gpu(&bufferT_[(nColsStagesAcc[args_d[i_args]] + args_d[i_args + 1]) * blockDim.x], &bufferT_[(nColsStagesAcc[args_d[i_args + 2]] + args_d[i_args + 3]) * blockDim.x]);
                i_args += 4;
                break;
            }
            case 1: {
                // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: commit1
                Goldilocks::op_gpu(args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * blockDim.x], &bufferT_[(nColsStagesAcc[args_d[i_args + 3]] + args_d[i_args + 4]) * blockDim.x], &bufferT_[(nColsStagesAcc[args_d[i_args + 5]] + args_d[i_args + 6]) * blockDim.x]);
                i_args += 7;
                break;
            }
            case 2: {
                // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: tmp1
                Goldilocks::op_gpu(args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * blockDim.x], &bufferT_[(nColsStagesAcc[args_d[i_args + 3]] + args_d[i_args + 4]) * blockDim.x], &tmp1[args_d[i_args + 5] * blockDim.x]);
                i_args += 6;
                break;
            }
            case 3: {
                // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: public
                Goldilocks::op_gpu(args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * blockDim.x], &bufferT_[(nColsStagesAcc[args_d[i_args + 3]] + args_d[i_args + 4]) * blockDim.x], &publics_d[args_d[i_args + 5] * blockDim.x]);
                i_args += 6;
                break;
            }
            case 4: {
                // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: number
                Goldilocks::op_gpu(args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * blockDim.x], &bufferT_[(nColsStagesAcc[args_d[i_args + 3]] + args_d[i_args + 4]) * blockDim.x], &numbers_d[args_d[i_args + 5]*blockDim.x]);
                i_args += 6;
                break;
            }
            case 5: {
                // COPY tmp1 to commit1
                Goldilocks::copy_gpu(&bufferT_[(nColsStagesAcc[args_d[i_args]] + args_d[i_args + 1]) * blockDim.x], &tmp1[args_d[i_args + 2] * blockDim.x]);
                i_args += 3;
                break;
            }
            case 6: {
                // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: tmp1
                Goldilocks::op_gpu(args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * blockDim.x], &tmp1[args_d[i_args + 3] * blockDim.x], &tmp1[args_d[i_args + 4] * blockDim.x]);
                i_args += 5;
                break;
            }
            case 7: {
                // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: public
                Goldilocks::op_gpu(args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * blockDim.x], &tmp1[args_d[i_args + 3] * blockDim.x], &publics_d[args_d[i_args + 4] * blockDim.x]);
                i_args += 5;
                break;
            }
            case 8: {
                // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: number
                Goldilocks::op_gpu(args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * blockDim.x], &tmp1[args_d[i_args + 3] * blockDim.x], &numbers_d[args_d[i_args + 4]*blockDim.x]);
                i_args += 5;
                break;
            }
            case 9: {
                // COPY public to commit1
                Goldilocks::copy_gpu(&bufferT_[(nColsStagesAcc[args_d[i_args]] + args_d[i_args + 1]) * blockDim.x], &publics_d[args_d[i_args + 2] * blockDim.x]);
                i_args += 3;
                break;
            }
            case 10: {
                // OPERATION WITH DEST: commit1 - SRC0: public - SRC1: public
                Goldilocks::op_gpu(args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * blockDim.x], &publics_d[args_d[i_args + 3] * blockDim.x], &publics_d[args_d[i_args + 4] * blockDim.x]);
                i_args += 5;
                break;
            }
            case 11: {
                // OPERATION WITH DEST: commit1 - SRC0: public - SRC1: number
                Goldilocks::op_gpu(args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * blockDim.x], &publics_d[args_d[i_args + 3] * blockDim.x], &numbers_d[args_d[i_args + 4]*blockDim.x]);
                i_args += 5;
                break;
            }
            case 12: {
                // COPY number to commit1
                Goldilocks::copy_gpu(&bufferT_[(nColsStagesAcc[args_d[i_args]] + args_d[i_args + 1]) * blockDim.x], &numbers_d[args_d[i_args + 2]*blockDim.x]);
                i_args += 3;
                break;
            }
            case 13: {
                // OPERATION WITH DEST: commit1 - SRC0: number - SRC1: number
                Goldilocks::op_gpu(args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * blockDim.x], &numbers_d[args_d[i_args + 3]*blockDim.x], &numbers_d[args_d[i_args + 4]*blockDim.x]);
                i_args += 5;
                break;
            }
            case 14: {
                // COPY commit1 to tmp1
                Goldilocks::copy_gpu(&tmp1[args_d[i_args] * blockDim.x], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * blockDim.x]);
                i_args += 3;
                break;
            }
            case 15: {
                // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: commit1
                Goldilocks::op_gpu(args_d[i_args], &tmp1[args_d[i_args + 1] * blockDim.x], &bufferT_[(nColsStagesAcc[args_d[i_args + 2]] + args_d[i_args + 3]) * blockDim.x], &bufferT_[(nColsStagesAcc[args_d[i_args + 4]] + args_d[i_args + 5]) * blockDim.x]);
                i_args += 6;
                break;
            }
            case 16: {
                // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: tmp1
                Goldilocks::op_gpu(args_d[i_args], &tmp1[args_d[i_args + 1] * blockDim.x], &bufferT_[(nColsStagesAcc[args_d[i_args + 2]] + args_d[i_args + 3]) * blockDim.x], &tmp1[args_d[i_args + 4] * blockDim.x]);
                i_args += 5;
                break;
            }
            case 17: {
                // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: public
                Goldilocks::op_gpu(args_d[i_args], &tmp1[args_d[i_args + 1] * blockDim.x], &bufferT_[(nColsStagesAcc[args_d[i_args + 2]] + args_d[i_args + 3]) * blockDim.x], &publics_d[args_d[i_args + 4] * blockDim.x]);
                i_args += 5;
                break;
            }
            case 18: {
                // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: number
                Goldilocks::op_gpu(args_d[i_args], &tmp1[args_d[i_args + 1] * blockDim.x], &bufferT_[(nColsStagesAcc[args_d[i_args + 2]] + args_d[i_args + 3]) * blockDim.x], &numbers_d[args_d[i_args + 4]*blockDim.x]);
                i_args += 5;
                break;
            }
            case 19: {
                // COPY tmp1 to tmp1
                Goldilocks::copy_gpu(&tmp1[args_d[i_args] * blockDim.x], &tmp1[args_d[i_args + 1] * blockDim.x]);
                i_args += 2;
                break;
            }
            case 20: {
                // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: tmp1
                Goldilocks::op_gpu(args_d[i_args], &tmp1[args_d[i_args + 1] * blockDim.x], &tmp1[args_d[i_args + 2] * blockDim.x], &tmp1[args_d[i_args + 3] * blockDim.x]);
                i_args += 4;
                break;
            }
            case 21: {
                // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: public
                Goldilocks::op_gpu(args_d[i_args], &tmp1[args_d[i_args + 1] * blockDim.x], &tmp1[args_d[i_args + 2] * blockDim.x], &publics_d[args_d[i_args + 3] * blockDim.x]);
                i_args += 4;
                break;
            }
            case 22: {
                // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: number
                Goldilocks::op_gpu(args_d[i_args], &tmp1[args_d[i_args + 1] * blockDim.x], &tmp1[args_d[i_args + 2] * blockDim.x], &numbers_d[args_d[i_args + 3]*blockDim.x]);
                i_args += 4;
                break;
            }
            case 23: {
                // COPY public to tmp1
                Goldilocks::copy_gpu(&tmp1[args_d[i_args] * blockDim.x], &publics_d[args_d[i_args + 1] * blockDim.x]);
                i_args += 2;
                break;
            }
            case 24: {
                // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: public
                Goldilocks::op_gpu(args_d[i_args], &tmp1[args_d[i_args + 1] * blockDim.x], &publics_d[args_d[i_args + 2] * blockDim.x], &publics_d[args_d[i_args + 3] * blockDim.x]);
                i_args += 4;
                break;
            }
            case 25: {
                // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: number
                Goldilocks::op_gpu(args_d[i_args], &tmp1[args_d[i_args + 1] * blockDim.x], &publics_d[args_d[i_args + 2] * blockDim.x], &numbers_d[args_d[i_args + 3]*blockDim.x]);
                i_args += 4;
                break;
            }
            case 26: {
                // COPY number to tmp1
                Goldilocks::copy_gpu(&tmp1[args_d[i_args] * blockDim.x], &numbers_d[args_d[i_args + 1]*blockDim.x]);
                i_args += 2;
                break;
            }
            case 27: {
                // OPERATION WITH DEST: tmp1 - SRC0: number - SRC1: number
                Goldilocks::op_gpu(args_d[i_args], &tmp1[args_d[i_args + 1] * blockDim.x], &numbers_d[args_d[i_args + 2]*blockDim.x], &numbers_d[args_d[i_args + 3]*blockDim.x]);
                i_args += 4;
                break;
            }
            case 28: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: commit1
                Goldilocks3::op_31_gpu(args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * blockDim.x], &bufferT_[(nColsStagesAcc[args_d[i_args + 3]] + args_d[i_args + 4]) * blockDim.x], &bufferT_[(nColsStagesAcc[args_d[i_args + 5]] + args_d[i_args + 6]) * blockDim.x]);
                i_args += 7;
                break;
            }
            case 29: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: tmp1
                Goldilocks3::op_31_gpu(args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * blockDim.x], &bufferT_[(nColsStagesAcc[args_d[i_args + 3]] + args_d[i_args + 4]) * blockDim.x], &tmp1[args_d[i_args + 5] * blockDim.x]);
                i_args += 6;
                break;
            }
            case 30: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: public
                Goldilocks3::op_31_gpu(args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * blockDim.x], &bufferT_[(nColsStagesAcc[args_d[i_args + 3]] + args_d[i_args + 4]) * blockDim.x], &publics_d[args_d[i_args + 5] * blockDim.x]);
                i_args += 6;
                break;
            }
            case 31: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: number
                Goldilocks3::op_31_gpu(args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * blockDim.x], &bufferT_[(nColsStagesAcc[args_d[i_args + 3]] + args_d[i_args + 4]) * blockDim.x], &numbers_d[args_d[i_args + 5]*blockDim.x]);
                i_args += 6;
                break;
            }
            case 32: {
                // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: commit1
                Goldilocks3::op_31_gpu(args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * blockDim.x], &tmp3[args_d[i_args + 3] * blockDim.x * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args_d[i_args + 4]] + args_d[i_args + 5]) * blockDim.x]);
                i_args += 6;
                break;
            }
            case 33: {
                // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: tmp1
                Goldilocks3::op_31_gpu(args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * blockDim.x], &tmp3[args_d[i_args + 3] * blockDim.x * FIELD_EXTENSION], &tmp1[args_d[i_args + 4] * blockDim.x]);
                i_args += 5;
                break;
            }
            case 34: {
                // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: public
                Goldilocks3::op_31_gpu(args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * blockDim.x], &tmp3[args_d[i_args + 3] * blockDim.x * FIELD_EXTENSION], &publics_d[args_d[i_args + 4] * blockDim.x]);
                i_args += 5;
                break;
            }
            case 35: {
                // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: number
                Goldilocks3::op_31_gpu(args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * blockDim.x], &tmp3[args_d[i_args + 3] * blockDim.x * FIELD_EXTENSION], &numbers_d[args_d[i_args + 4]*blockDim.x]);
                i_args += 5;
                break;
            }
            case 36: {
                // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: commit1
                Goldilocks3::op_31_gpu(args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * blockDim.x], &challenges_d[args_d[i_args + 3]*FIELD_EXTENSION*blockDim.x], &bufferT_[(nColsStagesAcc[args_d[i_args + 4]] + args_d[i_args + 5]) * blockDim.x]);
                i_args += 6;
                break;
            }
            case 37: {
                // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: tmp1
                Goldilocks3::op_31_gpu(args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * blockDim.x], &challenges_d[args_d[i_args + 3]*FIELD_EXTENSION*blockDim.x], &tmp1[args_d[i_args + 4] * blockDim.x]);
                i_args += 5;
                break;
            }
            case 38: {
                // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: public
                Goldilocks3::op_31_gpu(args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * blockDim.x], &challenges_d[args_d[i_args + 3]*FIELD_EXTENSION*blockDim.x], &publics_d[args_d[i_args + 4] * blockDim.x]);
                i_args += 5;
                break;
            }
            case 39: {
                // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: number
                Goldilocks3::op_31_gpu(args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * blockDim.x], &challenges_d[args_d[i_args + 3]*FIELD_EXTENSION*blockDim.x], &numbers_d[args_d[i_args + 4]*blockDim.x]);
                i_args += 5;
                break;
            }
            case 40: {
                // COPY commit3 to commit3
                Goldilocks3::copy_gpu(&bufferT_[(nColsStagesAcc[args_d[i_args]] + args_d[i_args + 1]) * blockDim.x], &bufferT_[(nColsStagesAcc[args_d[i_args + 2]] + args_d[i_args + 3]) * blockDim.x]);
                i_args += 4;
                break;
            }
            case 41: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: commit3
                Goldilocks3::op_gpu(args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * blockDim.x], &bufferT_[(nColsStagesAcc[args_d[i_args + 3]] + args_d[i_args + 4]) * blockDim.x], &bufferT_[(nColsStagesAcc[args_d[i_args + 5]] + args_d[i_args + 6]) * blockDim.x]);
                i_args += 7;
                break;
            }
            case 42: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: tmp3
                Goldilocks3::op_gpu(args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * blockDim.x], &bufferT_[(nColsStagesAcc[args_d[i_args + 3]] + args_d[i_args + 4]) * blockDim.x], &tmp3[args_d[i_args + 5] * blockDim.x * FIELD_EXTENSION]);
                i_args += 6;
                break;
            }
            case 43: {
                // MULTIPLICATION WITH DEST: commit3 - SRC0: commit3 - SRC1: challenge
                Goldilocks3::mul_gpu(&bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * blockDim.x], &bufferT_[(nColsStagesAcc[args_d[i_args + 3]] + args_d[i_args + 4]) * blockDim.x], &challenges_d[args_d[i_args + 5]*FIELD_EXTENSION*blockDim.x], &challenges_ops_d[args_d[i_args + 5]*FIELD_EXTENSION*blockDim.x]);
                i_args += 6;
                break;
            }
            case 44: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: challenge
                Goldilocks3::op_gpu(args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * blockDim.x], &bufferT_[(nColsStagesAcc[args_d[i_args + 3]] + args_d[i_args + 4]) * blockDim.x], &challenges_d[args_d[i_args + 5]*FIELD_EXTENSION*blockDim.x]);
                i_args += 6;
                break;
            }
            case 45: {
                // COPY tmp3 to commit3
                Goldilocks3::copy_gpu(&bufferT_[(nColsStagesAcc[args_d[i_args]] + args_d[i_args + 1]) * blockDim.x], &tmp3[args_d[i_args + 2] * blockDim.x * FIELD_EXTENSION]);
                i_args += 3;
                break;
            }
            case 46: {
                // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: tmp3
                Goldilocks3::op_gpu(args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * blockDim.x], &tmp3[args_d[i_args + 3] * blockDim.x * FIELD_EXTENSION], &tmp3[args_d[i_args + 4] * blockDim.x * FIELD_EXTENSION]);
                i_args += 5;
                break;
            }
            case 47: {
                // MULTIPLICATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: challenge
                Goldilocks3::mul_gpu(&bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * blockDim.x], &tmp3[args_d[i_args + 3] * blockDim.x * FIELD_EXTENSION], &challenges_d[args_d[i_args + 4]*FIELD_EXTENSION*blockDim.x], &challenges_ops_d[args_d[i_args + 4]*FIELD_EXTENSION*blockDim.x]);
                i_args += 5;
                break;
            }
            case 48: {
                // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: challenge
                Goldilocks3::op_gpu(args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * blockDim.x], &tmp3[args_d[i_args + 3] * blockDim.x * FIELD_EXTENSION], &challenges_d[args_d[i_args + 4]*FIELD_EXTENSION*blockDim.x]);
                i_args += 5;
                break;
            }
            case 49: {
                // MULTIPLICATION WITH DEST: commit3 - SRC0: challenge - SRC1: challenge
                Goldilocks3::mul_gpu(&bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * blockDim.x], &challenges_d[args_d[i_args + 3]*FIELD_EXTENSION*blockDim.x], &challenges_d[args_d[i_args + 4]*FIELD_EXTENSION*blockDim.x], &challenges_ops_d[args_d[i_args + 4]*FIELD_EXTENSION*blockDim.x]);
                i_args += 5;
                break;
            }
            case 50: {
                // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: challenge
                Goldilocks3::op_gpu(args_d[i_args], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * blockDim.x], &challenges_d[args_d[i_args + 3]*FIELD_EXTENSION*blockDim.x], &challenges_d[args_d[i_args + 4]*FIELD_EXTENSION*blockDim.x]);
                i_args += 5;
                break;
            }
            case 51: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit1
                Goldilocks3::op_31_gpu(args_d[i_args], &tmp3[args_d[i_args + 1] * blockDim.x * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args_d[i_args + 2]] + args_d[i_args + 3]) * blockDim.x], &bufferT_[(nColsStagesAcc[args_d[i_args + 4]] + args_d[i_args + 5]) * blockDim.x]);
                i_args += 6;
                break;
            }
            case 52: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp1
                Goldilocks3::op_31_gpu(args_d[i_args], &tmp3[args_d[i_args + 1] * blockDim.x * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args_d[i_args + 2]] + args_d[i_args + 3]) * blockDim.x], &tmp1[args_d[i_args + 4] * blockDim.x]);
                i_args += 5;
                break;
            }
            case 53: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: public
                Goldilocks3::op_31_gpu(args_d[i_args], &tmp3[args_d[i_args + 1] * blockDim.x * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args_d[i_args + 2]] + args_d[i_args + 3]) * blockDim.x], &publics_d[args_d[i_args + 4] * blockDim.x]);
                i_args += 5;
                break;
            }
            case 54: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: number
                Goldilocks3::op_31_gpu(args_d[i_args], &tmp3[args_d[i_args + 1] * blockDim.x * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args_d[i_args + 2]] + args_d[i_args + 3]) * blockDim.x], &numbers_d[args_d[i_args + 4]*blockDim.x]);
                i_args += 5;
                break;
            }
            case 55: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: commit1
                Goldilocks3::op_31_gpu(args_d[i_args], &tmp3[args_d[i_args + 1] * blockDim.x * FIELD_EXTENSION], &tmp3[args_d[i_args + 2] * blockDim.x * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args_d[i_args + 3]] + args_d[i_args + 4]) * blockDim.x]);
                i_args += 5;
                break;
            }
            case 56: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp1
                Goldilocks3::op_31_gpu(args_d[i_args], &tmp3[args_d[i_args + 1] * blockDim.x * FIELD_EXTENSION], &tmp3[args_d[i_args + 2] * blockDim.x * FIELD_EXTENSION], &tmp1[args_d[i_args + 3] * blockDim.x]);
                i_args += 4;
                break;
            }
            case 57: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: public
                Goldilocks3::op_31_gpu(args_d[i_args], &tmp3[args_d[i_args + 1] * blockDim.x * FIELD_EXTENSION], &tmp3[args_d[i_args + 2] * blockDim.x * FIELD_EXTENSION], &publics_d[args_d[i_args + 3] * blockDim.x]);
                i_args += 4;
                break;
            }
            case 58: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: number
                Goldilocks3::op_31_gpu(args_d[i_args], &tmp3[args_d[i_args + 1] * blockDim.x * FIELD_EXTENSION], &tmp3[args_d[i_args + 2] * blockDim.x * FIELD_EXTENSION], &numbers_d[args_d[i_args + 3]*blockDim.x]);
                i_args += 4;
                break;
            }
            case 59: {
                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: commit1
                Goldilocks3::op_31_gpu(args_d[i_args], &tmp3[args_d[i_args + 1] * blockDim.x * FIELD_EXTENSION], &challenges_d[args_d[i_args + 2]*FIELD_EXTENSION*blockDim.x], &bufferT_[(nColsStagesAcc[args_d[i_args + 3]] + args_d[i_args + 4]) * blockDim.x]);
                i_args += 5;
                break;
            }
            case 60: {
                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: tmp1
                Goldilocks3::op_31_gpu(args_d[i_args], &tmp3[args_d[i_args + 1] * blockDim.x * FIELD_EXTENSION], &challenges_d[args_d[i_args + 2]*FIELD_EXTENSION*blockDim.x], &tmp1[args_d[i_args + 3] * blockDim.x]);
                i_args += 4;
                break;
            }
            case 61: {
                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: public
                Goldilocks3::op_31_gpu(args_d[i_args], &tmp3[args_d[i_args + 1] * blockDim.x * FIELD_EXTENSION], &challenges_d[args_d[i_args + 2]*FIELD_EXTENSION*blockDim.x], &publics_d[args_d[i_args + 3] * blockDim.x]);
                i_args += 4;
                break;
            }
            case 62: {
                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: number
                Goldilocks3::op_31_gpu(args_d[i_args], &tmp3[args_d[i_args + 1] * blockDim.x * FIELD_EXTENSION], &challenges_d[args_d[i_args + 2]*FIELD_EXTENSION*blockDim.x], &numbers_d[args_d[i_args + 3]*blockDim.x]);
                i_args += 4;
                break;
            }
            case 63: {
                // COPY commit3 to tmp3
                Goldilocks3::copy_gpu(&tmp3[args_d[i_args] * blockDim.x * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args_d[i_args + 1]] + args_d[i_args + 2]) * blockDim.x]);
                i_args += 3;
                break;
            }
            case 64: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit3
                Goldilocks3::op_gpu(args_d[i_args], &tmp3[args_d[i_args + 1] * blockDim.x * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args_d[i_args + 2]] + args_d[i_args + 3]) * blockDim.x], &bufferT_[(nColsStagesAcc[args_d[i_args + 4]] + args_d[i_args + 5]) * blockDim.x]);
                i_args += 6;
                break;
            }
            case 65: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp3
                Goldilocks3::op_gpu(args_d[i_args], &tmp3[args_d[i_args + 1] * blockDim.x * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args_d[i_args + 2]] + args_d[i_args + 3]) * blockDim.x], &tmp3[args_d[i_args + 4] * blockDim.x * FIELD_EXTENSION]);
                i_args += 5;
                break;
            }
            case 66: {
                // MULTIPLICATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: challenge
                Goldilocks3::mul_gpu(&tmp3[args_d[i_args + 1] * blockDim.x * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args_d[i_args + 2]] + args_d[i_args + 3]) * blockDim.x], &challenges_d[args_d[i_args + 4]*FIELD_EXTENSION*blockDim.x], &challenges_ops_d[args_d[i_args + 4]*FIELD_EXTENSION*blockDim.x]);
                i_args += 5;
                break;
            }
            case 67: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: challenge
                Goldilocks3::op_gpu(args_d[i_args], &tmp3[args_d[i_args + 1] * blockDim.x * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args_d[i_args + 2]] + args_d[i_args + 3]) * blockDim.x], &challenges_d[args_d[i_args + 4]*FIELD_EXTENSION*blockDim.x]);
                i_args += 5;
                break;
            }
            case 68: {
                // COPY tmp3 to tmp3
                Goldilocks3::copy_gpu(&tmp3[args_d[i_args] * blockDim.x * FIELD_EXTENSION], &tmp3[args_d[i_args + 1] * blockDim.x * FIELD_EXTENSION]);
                i_args += 2;
                break;
            }
            case 69: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp3
                Goldilocks3::op_gpu(args_d[i_args], &tmp3[args_d[i_args + 1] * blockDim.x * FIELD_EXTENSION], &tmp3[args_d[i_args + 2] * blockDim.x * FIELD_EXTENSION], &tmp3[args_d[i_args + 3] * blockDim.x * FIELD_EXTENSION]);
                i_args += 4;
                break;
            }
            case 70: {
                // MULTIPLICATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: challenge
                Goldilocks3::mul_gpu(&tmp3[args_d[i_args + 1] * blockDim.x * FIELD_EXTENSION], &tmp3[args_d[i_args + 2] * blockDim.x * FIELD_EXTENSION], &challenges_d[args_d[i_args + 3]*FIELD_EXTENSION*blockDim.x], &challenges_ops_d[args_d[i_args + 3]*FIELD_EXTENSION*blockDim.x]);
                i_args += 4;
                break;
            }
            case 71: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: challenge
                Goldilocks3::op_gpu(args_d[i_args], &tmp3[args_d[i_args + 1] * blockDim.x * FIELD_EXTENSION], &tmp3[args_d[i_args + 2] * blockDim.x * FIELD_EXTENSION], &challenges_d[args_d[i_args + 3]*FIELD_EXTENSION*blockDim.x]);
                i_args += 4;
                break;
            }
            case 72: {
                // MULTIPLICATION WITH DEST: tmp3 - SRC0: challenge - SRC1: challenge
                Goldilocks3::mul_gpu(&tmp3[args_d[i_args + 1] * blockDim.x * FIELD_EXTENSION], &challenges_d[args_d[i_args + 2]*FIELD_EXTENSION*blockDim.x], &challenges_d[args_d[i_args + 3]*FIELD_EXTENSION*blockDim.x], &challenges_ops_d[args_d[i_args + 3]*FIELD_EXTENSION*blockDim.x]);
                i_args += 4;
                break;
            }
            case 73: {
                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: challenge
                Goldilocks3::op_gpu(args_d[i_args], &tmp3[args_d[i_args + 1] * blockDim.x * FIELD_EXTENSION], &challenges_d[args_d[i_args + 2]*FIELD_EXTENSION*blockDim.x], &challenges_d[args_d[i_args + 3]*FIELD_EXTENSION*blockDim.x]);
                i_args += 4;
                break;
            }
            case 74: {
                // COPY eval to tmp3
                Goldilocks3::copy_gpu(&tmp3[args_d[i_args] * blockDim.x * FIELD_EXTENSION], &evals_d[args_d[i_args + 1]*FIELD_EXTENSION*blockDim.x]);
                i_args += 2;
                break;
            }
            case 75: {
                // MULTIPLICATION WITH DEST: tmp3 - SRC0: eval - SRC1: challenge
                Goldilocks3::mul_gpu(&tmp3[args_d[i_args + 1] * blockDim.x * FIELD_EXTENSION], &evals_d[args_d[i_args + 2]*FIELD_EXTENSION*blockDim.x], &challenges_d[args_d[i_args + 3]*FIELD_EXTENSION*blockDim.x], &challenges_ops_d[args_d[i_args + 3]*FIELD_EXTENSION*blockDim.x]);
                i_args += 4;
                break;
            }
            case 76: {
                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: eval
                Goldilocks3::op_gpu(args_d[i_args], &tmp3[args_d[i_args + 1] * blockDim.x * FIELD_EXTENSION], &challenges_d[args_d[i_args + 2]*FIELD_EXTENSION*blockDim.x], &evals_d[args_d[i_args + 3]*FIELD_EXTENSION*blockDim.x]);
                i_args += 4;
                break;
            }
            case 77: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: eval
                Goldilocks3::op_gpu(args_d[i_args], &tmp3[args_d[i_args + 1] * blockDim.x * FIELD_EXTENSION], &tmp3[args_d[i_args + 2] * blockDim.x * FIELD_EXTENSION], &evals_d[args_d[i_args + 3]*FIELD_EXTENSION*blockDim.x]);
                i_args += 4;
                break;
            }
            case 78: {
                // OPERATION WITH DEST: tmp3 - SRC0: eval - SRC1: commit1
                Goldilocks3::op_31_gpu(args_d[i_args], &tmp3[args_d[i_args + 1] * blockDim.x * FIELD_EXTENSION], &evals_d[args_d[i_args + 2]*FIELD_EXTENSION*blockDim.x], &bufferT_[(nColsStagesAcc[args_d[i_args + 3]] + args_d[i_args + 4]) * blockDim.x]);
                i_args += 5;
                break;
            }
            case 79: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: eval
                Goldilocks3::op_gpu(args_d[i_args], &tmp3[args_d[i_args + 1] * blockDim.x * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args_d[i_args + 2]] + args_d[i_args + 3]) * blockDim.x], &evals_d[args_d[i_args + 4]*FIELD_EXTENSION*blockDim.x]);
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
    #endif
    }
 
void CHelpersStepsGPU::dataSetup_(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams){

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
        cudaMalloc(&challenges_d, params.challenges.degree()*FIELD_EXTENSION*nrowsPack*sizeof(Goldilocks::Element));
        cudaMemcpy(challenges_d, challenges, params.challenges.degree()*FIELD_EXTENSION*nrowsPack*sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);
        
        cudaMalloc(&challenges_ops_d, params.challenges.degree()*FIELD_EXTENSION*nrowsPack*sizeof(Goldilocks::Element));
        cudaMemcpy(challenges_ops_d, challenges_ops, params.challenges.degree()*FIELD_EXTENSION*nrowsPack*sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);
        
        uint64_t *numbers = &parserArgs.numbers[parserParams.numbersOffset];
        Goldilocks::Element numbers_[parserParams.nNumbers*nrowsPack];
        for(uint64_t i = 0; i < parserParams.nNumbers; ++i) {
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                numbers_[i*nrowsPack + j] = Goldilocks::fromU64(numbers[i]);
            }
        }
        cudaMalloc(&numbers_d, parserParams.nNumbers*nrowsPack*sizeof(Goldilocks::Element));
        cudaMemcpy(numbers_d, numbers_, parserParams.nNumbers*nrowsPack*sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);

        Goldilocks::Element publics[starkInfo.nPublics*nrowsPack];
        for(uint64_t i = 0; i < starkInfo.nPublics; ++i) {
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                publics[i*nrowsPack + j] = params.publicInputs[i];
            }
        }

        cudaMalloc(&publics_d, starkInfo.nPublics*nrowsPack*sizeof(Goldilocks::Element));
        cudaMemcpy(publics_d, publics, starkInfo.nPublics*nrowsPack*sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);

        Goldilocks::Element evals[params.evals.degree()*FIELD_EXTENSION*nrowsPack];
        for(uint64_t i = 0; i < params.evals.degree(); ++i) {
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                evals[(i*FIELD_EXTENSION)*nrowsPack + j] = params.evals[i][0];
                evals[(i*FIELD_EXTENSION + 1)*nrowsPack + j] = params.evals[i][1];
                evals[(i*FIELD_EXTENSION + 2)*nrowsPack + j] = params.evals[i][2];
            }
        }
        cudaMalloc(&evals_d, params.evals.degree()*FIELD_EXTENSION*nrowsPack*sizeof(Goldilocks::Element));
        cudaMemcpy(evals_d, evals, params.evals.degree()*FIELD_EXTENSION*nrowsPack*sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);

        /* 
            buffered data
        */
        cudaMalloc(&constPols_d, params.pConstPols->numPols()*params.pConstPols->degree()*sizeof(Goldilocks::Element));
        cudaMemcpy(constPols_d, params.pConstPols->address(), params.pConstPols->size(), cudaMemcpyHostToDevice);
        
        cudaMalloc(&constPols2ns_d, params.pConstPols2ns->numPols()*params.pConstPols2ns->degree()*sizeof(Goldilocks::Element));
        cudaMemcpy(constPols2ns_d, params.pConstPols2ns->address(), params.pConstPols2ns->size(), cudaMemcpyHostToDevice);

        cudaMalloc(&x_d, params.x_n.dim()*params.x_n.degree()*sizeof(Goldilocks::Element));
        cudaMemcpy(x_d, params.x_n.address(), params.x_n.size(), cudaMemcpyHostToDevice);

        cudaMalloc(&x_2ns_d, params.x_2ns.dim()*params.x_2ns.degree()*sizeof(Goldilocks::Element));
        cudaMemcpy(x_2ns_d, params.x_2ns.address(), params.x_2ns.size(), cudaMemcpyHostToDevice);

        cudaMalloc(&zi_d, params.zi.dim()*params.zi.degree()*sizeof(Goldilocks::Element));
        cudaMemcpy(zi_d, params.zi.address(), params.zi.size(), cudaMemcpyHostToDevice);

        cudaMalloc(&xDivXSubXi_d, params.xDivXSubXi.dim()*params.xDivXSubXi.degree()*sizeof(Goldilocks::Element));
        cudaMemcpy(xDivXSubXi_d, params.xDivXSubXi.address(), params.xDivXSubXi.size(), cudaMemcpyHostToDevice);

        cudaMalloc(&pols_d, starkInfo.mapTotalN*sizeof(Goldilocks::Element));
        cudaMemcpy(pols_d, params.pols, starkInfo.mapTotalN*sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);

        /*
            temporal buffers
        */
        cudaMalloc(&bufferT_d, numBlocks*sizeof(Goldilocks::Element*));
        for(uint64_t i = 0; i < numBlocks; ++i) {
            cudaMalloc(&bufferT_d[i], 2*nCols*nrowsPack*sizeof(Goldilocks::Element));
        }
        cudaMalloc(&tmp1_d, numBlocks*sizeof(Goldilocks::Element*));
        for(uint64_t i = 0; i < numBlocks; ++i) {
            cudaMalloc(&tmp1_d[i], parserParams.nTemp1*nrowsPack*sizeof(Goldilocks::Element));
        }
        cudaMalloc(&tmp3_d, numBlocks*sizeof(Goldilocks::Element*));
        for(uint64_t i = 0; i < numBlocks; ++i) {
            cudaMalloc(&tmp3_d[i], parserParams.nTemp3*nrowsPack*FIELD_EXTENSION*sizeof(Goldilocks::Element));
        }
}    
