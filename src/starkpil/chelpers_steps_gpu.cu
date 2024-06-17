#include "chelpers_steps_gpu.hpp"
#ifdef __USE_CUDA__
#include "gl64_t.cuh"
#include "goldilocks_cubic_extension.cuh"
#endif


void CHelpersStepsGPU::setBufferTInfo(StarkInfo& starkInfo, uint64_t stage) 
{
        bool domainExtended = stage <= 3 ? false : true;
        nColsStagesAcc.resize(10 + 2);
        nColsStages.resize(10 + 2);
        offsetsStages.resize(10 + 2);

        nColsStages[0] = starkInfo.nConstants + 2;
        offsetsStages[0] = 0;

        for(uint64_t s = 1; s <= 3; ++s) {
            nColsStages[s] = starkInfo.mapSectionsN.section[string2section("cm" + to_string(s) + "_n")];
            if(domainExtended) {
                offsetsStages[s] = starkInfo.mapOffsets.section[string2section("cm" + to_string(s) + "_2ns")];
            } else {
                offsetsStages[s] = starkInfo.mapOffsets.section[string2section("cm" + to_string(s) + "_n")];
            }
        }
        if(domainExtended) {
            nColsStages[4] = starkInfo.mapSectionsN.section[eSection::cm4_2ns];
            offsetsStages[4] = starkInfo.mapOffsets.section[eSection::cm4_2ns];
        } else {
            nColsStages[4] = starkInfo.mapSectionsN.section[eSection::tmpExp_n];
            offsetsStages[4] = starkInfo.mapOffsets.section[eSection::tmpExp_n];
        }
        for(uint64_t o = 0; o < 2; ++o) {
            for(uint64_t s = 0; s < 5; ++s) {
                if(s == 0) {
                    if(o == 0) {
                        nColsStagesAcc[0] = 0;
                    } else {
                        nColsStagesAcc[5*o] = nColsStagesAcc[5*o - 1] + nColsStages[4];
                    }
                } else {
                    nColsStagesAcc[5*o + s] = nColsStagesAcc[5*o + (s - 1)] + nColsStages[(s - 1)];
                }
            }
        }
        nColsStagesAcc[10] = nColsStagesAcc[9] + nColsStages[9]; // Polinomials f & q
        if(stage == 4) {
            offsetsStages[10] = starkInfo.mapOffsets.section[eSection::q_2ns];
            nColsStages[10] = starkInfo.qDim;
        } else if(stage == 5) {
            offsetsStages[10] = starkInfo.mapOffsets.section[eSection::f_2ns];
            nColsStages[10] = 3;
        }
        nColsStagesAcc[11] = nColsStagesAcc[10] + 3; // xDivXSubXi
        nCols = nColsStagesAcc[11] + 6;
    
#ifdef __USE_CUDA__
        cudaMalloc((void**)&nColsStages_d_, nColsStages.size() * sizeof(uint64_t));
        cudaMemcpy(nColsStages_d_, nColsStages.data(), nColsStages.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMalloc((void**)&nColsStagesAcc_d_, nColsStagesAcc.size() * sizeof(uint64_t));
        cudaMemcpy(nColsStagesAcc_d_, nColsStagesAcc.data(), nColsStagesAcc.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMalloc((void**)&offsetsStages_d_, offsetsStages.size() * sizeof(uint64_t));
        cudaMemcpy(offsetsStages_d_, offsetsStages.data(), offsetsStages.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
#endif
        
    }

void CHelpersStepsGPU::calculateExpressions(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams)  {

    uint32_t nrowsPack =  4;
    bool domainExtended = parserParams.stage > 3 ? true : false;
    uint64_t domainSize = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
    

    setBufferTInfo(starkInfo, parserParams.stage);
#ifdef __USE_CUDA__
    dataSetup_(starkInfo, params, parserArgs, parserParams);
#endif
    dataSetup(starkInfo, params, parserArgs, parserParams);

    #pragma omp parallel for
    for (uint64_t i = 0; i < domainSize; i+= nrowsPack) {

        loadPolinomials(starkInfo, params, i, parserParams.stage, nrowsPack, domainExtended);
        optcodeIteration(nrowsPack,  parserParams.nOps, parserParams.nArgs);
        storePolinomials(starkInfo, params, storePol_d, i, nrowsPack, domainExtended);
    }
}



__device__  void CHelpersStepsGPU::storePolinomials_(StarkInfo &starkInfo, StepsParams &params, uint64_t row, uint64_t domainExtended) {

        gl64_t *bufferT_ = bufferT_d_[blockIdx.x];
        if(domainExtended) {
            // Store either polinomial f or polinomial q
            for(uint64_t k = 0; k < nColsStages_d_[10]; ++k) {
                gl64_t *buffT = &bufferT_[(nColsStagesAcc_d_[10] + k)* blockDim.x];
                gl64_t::copy_gpu( &pols_d_[offsetsStages_d_[10] + k + row * nColsStages_d_[10]], nColsStages_d_[10], buffT);
            }
        } else {
            uint64_t nStages = 3;
            uint64_t domainSize =  1 << starkInfo.starkStruct.nBits;
            for(uint64_t s = 2; s <= nStages + 1; ++s) {
                bool isTmpPol =  s == 4;
                for(uint64_t k = 0; k < nColsStages_d_[s]; ++k) {
                    uint64_t dim = storePol_d_[nColsStagesAcc_d_[s] + k];
                    if(storePol_d_[nColsStagesAcc_d_[s] + k]) {
                        gl64_t *buffT = &bufferT_[(nColsStagesAcc_d_[s] + k)* blockDim.x];
                        if(isTmpPol) {
                            for(uint64_t i = 0; i < dim; ++i) {
                                gl64_t::copy_gpu(&pols_d_[offsetsStages_d_[s] + k * domainSize + row * dim + i], uint64_t(dim), &buffT[i*blockDim.x]);
                            }
                        } else {
                            gl64_t::copy_gpu(&pols_d_[offsetsStages_d_[s] + k + row * nColsStages_d_[s]], nColsStages_d_[s], buffT);
                        }
                    }
                }
            }
        }
    }

__device__   void CHelpersStepsGPU::loadPolinomials_(StarkInfo &starkInfo, StepsParams &params, uint64_t row, uint64_t stage, uint64_t domainExtended) {
    
    // buffered data
    gl64_t *constPols_aux = domainExtended ? constPols2ns_d_ : constPols_d_;
    uint64_t domainSize = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
    gl64_t *x_aux = domainExtended ? x_2ns_d_ : x_d_;
    
    uint64_t nStages = 3;
    uint64_t nextStride = domainExtended ?  1 << (starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits) : 1;
    uint64_t nextStrides[2] = {0, nextStride};
    for(uint64_t k = 0; k < starkInfo.nConstants; ++k) {
        for(uint64_t o = 0; o < 2; ++o) {
            uint64_t l = (row + nextStrides[o]) % domainSize;
            bufferT_d_[blockIdx.x][(nColsStagesAcc_d_[5*o] + k)*blockDim.x + threadIdx.x] = constPols_aux[l * starkInfo.nConstants + k];
        }
    }

    // Load x and Zi
    bufferT_d_[blockIdx.x][starkInfo.nConstants*blockDim.x + threadIdx.x] = x_aux[row];
    bufferT_d_[blockIdx.x][(starkInfo.nConstants + 1)*blockDim.x + threadIdx.x] = zi_d_[row];
    

    for(uint64_t s = 1; s <= nStages; ++s) {
        for(uint64_t k = 0; k < nColsStages_d_[s]; ++k) {
            for(uint64_t o = 0; o < 2; ++o) {
                    uint64_t l = (row + nextStrides[o]) % domainSize;
                    bufferT_d_[blockIdx.x][(nColsStagesAcc_d_[5*o + s] + k)* blockDim.x + threadIdx.x] = pols_d_[offsetsStages_d_[s] + l * nColsStages_d_[s] + k];
            }
        }
    }

    if(stage == 5) {
        for(uint64_t k = 0; k < nColsStages_d_[nStages + 1]; ++k) {
            for(uint64_t o = 0; o < 2; ++o) {
                uint64_t l = (row + nextStrides[o]) % domainSize;
                bufferT_d_[blockIdx.x][(nColsStagesAcc_d_[5*o + nStages + 1] + k)*blockDim.x + threadIdx.x] = pols_d_[offsetsStages_d_[nStages + 1] + l * nColsStages_d_[nStages + 1] + k];
            }
        }

        // Load xDivXSubXi & xDivXSubWXi
        for(uint64_t d = 0; d < 2; ++d) {
            for(uint64_t i = 0; i < FIELD_EXTENSION; ++i) {
                bufferT_d_[blockIdx.x][(nColsStagesAcc_d_[11] + FIELD_EXTENSION*d + i)*blockDim.x + threadIdx.x] = xDivXSubXi_d_[(d*domainSize + row)*FIELD_EXTENSION+i];                   
            }
        }
    }    
    
}

 void CHelpersStepsGPU::calculateExpressions_(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams)  {
    bool domainExtended = parserParams.stage > 3 ? true : false;
    uint64_t domainSize = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
    
    setBufferTInfo(starkInfo, parserParams.stage);
    dataSetup_(starkInfo, params, parserArgs, parserParams);
    blockCalculation<<<numBlocks, 256>>>(this, starkInfo,params, domainSize, domainExtended, parserParams.stage, parserParams.nOps, parserParams.nArgs); 
}

__global__ void blockCalculation(CHelpersStepsGPU*  chelpers_, StarkInfo &starkInfo, StepsParams &params, uint64_t domainSize, bool domainExtended, uint64_t stage, uint32_t nOps, uint32_t nArgs) {
    uint32_t i = threadIdx.x;
    while( i < domainSize) {
        chelpers_->loadPolinomials_(starkInfo, params, i, stage, domainExtended);
        chelpers_->optcodeIteration_(nOps, nArgs);
        chelpers_->storePolinomials_(starkInfo, params, i, domainExtended);
        i += blockDim.x;
    }
}

    __device__  void CHelpersStepsGPU::optcodeIteration_(uint32_t nOps, uint32_t nArgs) {

    uint64_t i_args = 0;
    gl64_t *bufferT_ = bufferT_d_[blockIdx.x];
    gl64_t *tmp1 = tmp1_d_[blockIdx.x];
    gl64_t *tmp3 = tmp3_d_[blockIdx.x];

    for (uint64_t kk = 0; kk < nOps; ++kk) {
        
        switch (ops_d[kk]) {
            case 0: {
                // COPY commit1 to commit1
                gl64_t::copy_gpu(&bufferT_[(nColsStagesAcc_d_[args_d_[i_args]] + args_d_[i_args + 1]) * blockDim.x], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 2]] + args_d_[i_args + 3]) * blockDim.x]);
                i_args += 4;
                break;
            }
            case 1: {
                // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: commit1
                gl64_t::op_gpu(args_d_[i_args], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 1]] + args_d_[i_args + 2]) * blockDim.x], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 3]] + args_d_[i_args + 4]) * blockDim.x], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 5]] + args_d_[i_args + 6]) * blockDim.x]);
                i_args += 7;
                break;
            }
            case 2: {
                // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: tmp1
                gl64_t::op_gpu(args_d_[i_args], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 1]] + args_d_[i_args + 2]) * blockDim.x], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 3]] + args_d_[i_args + 4]) * blockDim.x], &tmp1[args_d_[i_args + 5] * blockDim.x]);
                i_args += 6;
                break;
            }
            case 3: {
                // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: public
                gl64_t::op_gpu(args_d_[i_args], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 1]] + args_d_[i_args + 2]) * blockDim.x], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 3]] + args_d_[i_args + 4]) * blockDim.x], &publics_d_[args_d_[i_args + 5] * blockDim.x]);
                i_args += 6;
                break;
            }
            case 4: {
                // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: number
                gl64_t::op_gpu(args_d_[i_args], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 1]] + args_d_[i_args + 2]) * blockDim.x], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 3]] + args_d_[i_args + 4]) * blockDim.x], &numbers_d_[args_d_[i_args + 5]*blockDim.x]);
                i_args += 6;
                break;
            }
            case 5: {
                // COPY tmp1 to commit1
                gl64_t::copy_gpu(&bufferT_[(nColsStagesAcc_d_[args_d_[i_args]] + args_d_[i_args + 1]) * blockDim.x], &tmp1[args_d_[i_args + 2] * blockDim.x]);
                i_args += 3;
                break;
            }
            case 6: {
                // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: tmp1
                gl64_t::op_gpu(args_d_[i_args], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 1]] + args_d_[i_args + 2]) * blockDim.x], &tmp1[args_d_[i_args + 3] * blockDim.x], &tmp1[args_d_[i_args + 4] * blockDim.x]);
                i_args += 5;
                break;
            }
            case 7: {
                // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: public
                gl64_t::op_gpu(args_d_[i_args], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 1]] + args_d_[i_args + 2]) * blockDim.x], &tmp1[args_d_[i_args + 3] * blockDim.x], &publics_d_[args_d_[i_args + 4] * blockDim.x]);
                i_args += 5;
                break;
            }
            case 8: {
                // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: number
                gl64_t::op_gpu(args_d_[i_args], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 1]] + args_d_[i_args + 2]) * blockDim.x], &tmp1[args_d_[i_args + 3] * blockDim.x], &numbers_d_[args_d_[i_args + 4]*blockDim.x]);
                i_args += 5;
                break;
            }
            case 9: {
                // COPY public to commit1
                gl64_t::copy_gpu(&bufferT_[(nColsStagesAcc_d_[args_d_[i_args]] + args_d_[i_args + 1]) * blockDim.x], &publics_d_[args_d_[i_args + 2] * blockDim.x]);
                i_args += 3;
                break;
            }
            case 10: {
                // OPERATION WITH DEST: commit1 - SRC0: public - SRC1: public
                gl64_t::op_gpu(args_d_[i_args], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 1]] + args_d_[i_args + 2]) * blockDim.x], &publics_d_[args_d_[i_args + 3] * blockDim.x], &publics_d_[args_d_[i_args + 4] * blockDim.x]);
                i_args += 5;
                break;
            }
            case 11: {
                // OPERATION WITH DEST: commit1 - SRC0: public - SRC1: number
                gl64_t::op_gpu(args_d_[i_args], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 1]] + args_d_[i_args + 2]) * blockDim.x], &publics_d_[args_d_[i_args + 3] * blockDim.x], &numbers_d_[args_d_[i_args + 4]*blockDim.x]);
                i_args += 5;
                break;
            }
            case 12: {
                // COPY number to commit1
                gl64_t::copy_gpu(&bufferT_[(nColsStagesAcc_d_[args_d_[i_args]] + args_d_[i_args + 1]) * blockDim.x], &numbers_d_[args_d_[i_args + 2]*blockDim.x]);
                i_args += 3;
                break;
            }
            case 13: {
                // OPERATION WITH DEST: commit1 - SRC0: number - SRC1: number
                gl64_t::op_gpu(args_d_[i_args], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 1]] + args_d_[i_args + 2]) * blockDim.x], &numbers_d_[args_d_[i_args + 3]*blockDim.x], &numbers_d_[args_d_[i_args + 4]*blockDim.x]);
                i_args += 5;
                break;
            }
            case 14: {
                // COPY commit1 to tmp1
                gl64_t::copy_gpu(&tmp1[args_d_[i_args] * blockDim.x], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 1]] + args_d_[i_args + 2]) * blockDim.x]);
                i_args += 3;
                break;
            }
            case 15: {
                // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: commit1
                gl64_t::op_gpu(args_d_[i_args], &tmp1[args_d_[i_args + 1] * blockDim.x], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 2]] + args_d_[i_args + 3]) * blockDim.x], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 4]] + args_d_[i_args + 5]) * blockDim.x]);
                i_args += 6;
                break;
            }
            case 16: {
                // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: tmp1
                gl64_t::op_gpu(args_d_[i_args], &tmp1[args_d_[i_args + 1] * blockDim.x], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 2]] + args_d_[i_args + 3]) * blockDim.x], &tmp1[args_d_[i_args + 4] * blockDim.x]);
                i_args += 5;
                break;
            }
            case 17: {
                // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: public
                gl64_t::op_gpu(args_d_[i_args], &tmp1[args_d_[i_args + 1] * blockDim.x], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 2]] + args_d_[i_args + 3]) * blockDim.x], &publics_d_[args_d_[i_args + 4] * blockDim.x]);
                i_args += 5;
                break;
            }
            case 18: {
                // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: number
                gl64_t::op_gpu(args_d_[i_args], &tmp1[args_d_[i_args + 1] * blockDim.x], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 2]] + args_d_[i_args + 3]) * blockDim.x], &numbers_d_[args_d_[i_args + 4]*blockDim.x]);
                i_args += 5;
                break;
            }
            case 19: {
                // COPY tmp1 to tmp1
                gl64_t::copy_gpu(&tmp1[args_d_[i_args] * blockDim.x], &tmp1[args_d_[i_args + 1] * blockDim.x]);
                i_args += 2;
                break;
            }
            case 20: {
                // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: tmp1
                gl64_t::op_gpu(args_d_[i_args], &tmp1[args_d_[i_args + 1] * blockDim.x], &tmp1[args_d_[i_args + 2] * blockDim.x], &tmp1[args_d_[i_args + 3] * blockDim.x]);
                i_args += 4;
                break;
            }
            case 21: {
                // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: public
                gl64_t::op_gpu(args_d_[i_args], &tmp1[args_d_[i_args + 1] * blockDim.x], &tmp1[args_d_[i_args + 2] * blockDim.x], &publics_d_[args_d_[i_args + 3] * blockDim.x]);
                i_args += 4;
                break;
            }
            case 22: {
                // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: number
                gl64_t::op_gpu(args_d_[i_args], &tmp1[args_d_[i_args + 1] * blockDim.x], &tmp1[args_d_[i_args + 2] * blockDim.x], &numbers_d_[args_d_[i_args + 3]*blockDim.x]);
                i_args += 4;
                break;
            }
            case 23: {
                // COPY public to tmp1
                gl64_t::copy_gpu(&tmp1[args_d_[i_args] * blockDim.x], &publics_d_[args_d_[i_args + 1] * blockDim.x]);
                i_args += 2;
                break;
            }
            case 24: {
                // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: public
                gl64_t::op_gpu(args_d_[i_args], &tmp1[args_d_[i_args + 1] * blockDim.x], &publics_d_[args_d_[i_args + 2] * blockDim.x], &publics_d_[args_d_[i_args + 3] * blockDim.x]);
                i_args += 4;
                break;
            }
            case 25: {
                // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: number
                gl64_t::op_gpu(args_d_[i_args], &tmp1[args_d_[i_args + 1] * blockDim.x], &publics_d_[args_d_[i_args + 2] * blockDim.x], &numbers_d_[args_d_[i_args + 3]*blockDim.x]);
                i_args += 4;
                break;
            }
            case 26: {
                // COPY number to tmp1
                gl64_t::copy_gpu(&tmp1[args_d_[i_args] * blockDim.x], &numbers_d_[args_d_[i_args + 1]*blockDim.x]);
                i_args += 2;
                break;
            }
            case 27: {
                // OPERATION WITH DEST: tmp1 - SRC0: number - SRC1: number
                gl64_t::op_gpu(args_d_[i_args], &tmp1[args_d_[i_args + 1] * blockDim.x], &numbers_d_[args_d_[i_args + 2]*blockDim.x], &numbers_d_[args_d_[i_args + 3]*blockDim.x]);
                i_args += 4;
                break;
            }
            case 28: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: commit1
                Goldilocks3GPU::op_31_gpu(args_d_[i_args], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 1]] + args_d_[i_args + 2]) * blockDim.x], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 3]] + args_d_[i_args + 4]) * blockDim.x], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 5]] + args_d_[i_args + 6]) * blockDim.x]);
                i_args += 7;
                break;
            }
            case 29: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: tmp1
                Goldilocks3GPU::op_31_gpu(args_d_[i_args], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 1]] + args_d_[i_args + 2]) * blockDim.x], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 3]] + args_d_[i_args + 4]) * blockDim.x], &tmp1[args_d_[i_args + 5] * blockDim.x]);
                i_args += 6;
                break;
            }
            case 30: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: public
                Goldilocks3GPU::op_31_gpu(args_d_[i_args], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 1]] + args_d_[i_args + 2]) * blockDim.x], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 3]] + args_d_[i_args + 4]) * blockDim.x], &publics_d_[args_d_[i_args + 5] * blockDim.x]);
                i_args += 6;
                break;
            }
            case 31: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: number
                Goldilocks3GPU::op_31_gpu(args_d_[i_args], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 1]] + args_d_[i_args + 2]) * blockDim.x], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 3]] + args_d_[i_args + 4]) * blockDim.x], &numbers_d_[args_d_[i_args + 5]*blockDim.x]);
                i_args += 6;
                break;
            }
            case 32: {
                // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: commit1
                Goldilocks3GPU::op_31_gpu(args_d_[i_args], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 1]] + args_d_[i_args + 2]) * blockDim.x], &tmp3[args_d_[i_args + 3] * blockDim.x * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 4]] + args_d_[i_args + 5]) * blockDim.x]);
                i_args += 6;
                break;
            }
            case 33: {
                // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: tmp1
                Goldilocks3GPU::op_31_gpu(args_d_[i_args], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 1]] + args_d_[i_args + 2]) * blockDim.x], &tmp3[args_d_[i_args + 3] * blockDim.x * FIELD_EXTENSION], &tmp1[args_d_[i_args + 4] * blockDim.x]);
                i_args += 5;
                break;
            }
            case 34: {
                // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: public
                Goldilocks3GPU::op_31_gpu(args_d_[i_args], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 1]] + args_d_[i_args + 2]) * blockDim.x], &tmp3[args_d_[i_args + 3] * blockDim.x * FIELD_EXTENSION], &publics_d_[args_d_[i_args + 4] * blockDim.x]);
                i_args += 5;
                break;
            }
            case 35: {
                // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: number
                Goldilocks3GPU::op_31_gpu(args_d_[i_args], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 1]] + args_d_[i_args + 2]) * blockDim.x], &tmp3[args_d_[i_args + 3] * blockDim.x * FIELD_EXTENSION], &numbers_d_[args_d_[i_args + 4]*blockDim.x]);
                i_args += 5;
                break;
            }
            case 36: {
                // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: commit1
                Goldilocks3GPU::op_31_gpu(args_d_[i_args], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 1]] + args_d_[i_args + 2]) * blockDim.x], &challenges_d_[args_d_[i_args + 3]*FIELD_EXTENSION*blockDim.x], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 4]] + args_d_[i_args + 5]) * blockDim.x]);
                i_args += 6;
                break;
            }
            case 37: {
                // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: tmp1
                Goldilocks3GPU::op_31_gpu(args_d_[i_args], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 1]] + args_d_[i_args + 2]) * blockDim.x], &challenges_d_[args_d_[i_args + 3]*FIELD_EXTENSION*blockDim.x], &tmp1[args_d_[i_args + 4] * blockDim.x]);
                i_args += 5;
                break;
            }
            case 38: {
                // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: public
                Goldilocks3GPU::op_31_gpu(args_d_[i_args], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 1]] + args_d_[i_args + 2]) * blockDim.x], &challenges_d_[args_d_[i_args + 3]*FIELD_EXTENSION*blockDim.x], &publics_d_[args_d_[i_args + 4] * blockDim.x]);
                i_args += 5;
                break;
            }
            case 39: {
                // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: number
                Goldilocks3GPU::op_31_gpu(args_d_[i_args], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 1]] + args_d_[i_args + 2]) * blockDim.x], &challenges_d_[args_d_[i_args + 3]*FIELD_EXTENSION*blockDim.x], &numbers_d_[args_d_[i_args + 4]*blockDim.x]);
                i_args += 5;
                break;
            }
            case 40: {
                // COPY commit3 to commit3
                Goldilocks3GPU::copy_gpu(&bufferT_[(nColsStagesAcc_d_[args_d_[i_args]] + args_d_[i_args + 1]) * blockDim.x], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 2]] + args_d_[i_args + 3]) * blockDim.x]);
                i_args += 4;
                break;
            }
            case 41: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: commit3
                Goldilocks3GPU::op_gpu(args_d_[i_args], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 1]] + args_d_[i_args + 2]) * blockDim.x], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 3]] + args_d_[i_args + 4]) * blockDim.x], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 5]] + args_d_[i_args + 6]) * blockDim.x]);
                i_args += 7;
                break;
            }
            case 42: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: tmp3
                Goldilocks3GPU::op_gpu(args_d_[i_args], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 1]] + args_d_[i_args + 2]) * blockDim.x], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 3]] + args_d_[i_args + 4]) * blockDim.x], &tmp3[args_d_[i_args + 5] * blockDim.x * FIELD_EXTENSION]);
                i_args += 6;
                break;
            }
            case 43: {
                // MULTIPLICATION WITH DEST: commit3 - SRC0: commit3 - SRC1: challenge
                Goldilocks3GPU::mul_gpu(&bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 1]] + args_d_[i_args + 2]) * blockDim.x], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 3]] + args_d_[i_args + 4]) * blockDim.x], &challenges_d_[args_d_[i_args + 5]*FIELD_EXTENSION*blockDim.x], &challenges_ops_d_[args_d_[i_args + 5]*FIELD_EXTENSION*blockDim.x]);
                i_args += 6;
                break;
            }
            case 44: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: challenge
                Goldilocks3GPU::op_gpu(args_d_[i_args], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 1]] + args_d_[i_args + 2]) * blockDim.x], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 3]] + args_d_[i_args + 4]) * blockDim.x], &challenges_d_[args_d_[i_args + 5]*FIELD_EXTENSION*blockDim.x]);
                i_args += 6;
                break;
            }
            case 45: {
                // COPY tmp3 to commit3
                Goldilocks3GPU::copy_gpu(&bufferT_[(nColsStagesAcc_d_[args_d_[i_args]] + args_d_[i_args + 1]) * blockDim.x], &tmp3[args_d_[i_args + 2] * blockDim.x * FIELD_EXTENSION]);
                i_args += 3;
                break;
            }
            case 46: {
                // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: tmp3
                Goldilocks3GPU::op_gpu(args_d_[i_args], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 1]] + args_d_[i_args + 2]) * blockDim.x], &tmp3[args_d_[i_args + 3] * blockDim.x * FIELD_EXTENSION], &tmp3[args_d_[i_args + 4] * blockDim.x * FIELD_EXTENSION]);
                i_args += 5;
                break;
            }
            case 47: {
                // MULTIPLICATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: challenge
                Goldilocks3GPU::mul_gpu(&bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 1]] + args_d_[i_args + 2]) * blockDim.x], &tmp3[args_d_[i_args + 3] * blockDim.x * FIELD_EXTENSION], &challenges_d_[args_d_[i_args + 4]*FIELD_EXTENSION*blockDim.x], &challenges_ops_d_[args_d_[i_args + 4]*FIELD_EXTENSION*blockDim.x]);
                i_args += 5;
                break;
            }
            case 48: {
                // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: challenge
                Goldilocks3GPU::op_gpu(args_d_[i_args], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 1]] + args_d_[i_args + 2]) * blockDim.x], &tmp3[args_d_[i_args + 3] * blockDim.x * FIELD_EXTENSION], &challenges_d_[args_d_[i_args + 4]*FIELD_EXTENSION*blockDim.x]);
                i_args += 5;
                break;
            }
            case 49: {
                // MULTIPLICATION WITH DEST: commit3 - SRC0: challenge - SRC1: challenge
                Goldilocks3GPU::mul_gpu(&bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 1]] + args_d_[i_args + 2]) * blockDim.x], &challenges_d_[args_d_[i_args + 3]*FIELD_EXTENSION*blockDim.x], &challenges_d_[args_d_[i_args + 4]*FIELD_EXTENSION*blockDim.x], &challenges_ops_d_[args_d_[i_args + 4]*FIELD_EXTENSION*blockDim.x]);
                i_args += 5;
                break;
            }
            case 50: {
                // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: challenge
                Goldilocks3GPU::op_gpu(args_d_[i_args], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 1]] + args_d_[i_args + 2]) * blockDim.x], &challenges_d_[args_d_[i_args + 3]*FIELD_EXTENSION*blockDim.x], &challenges_d_[args_d_[i_args + 4]*FIELD_EXTENSION*blockDim.x]);
                i_args += 5;
                break;
            }
            case 51: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit1
                Goldilocks3GPU::op_31_gpu(args_d_[i_args], &tmp3[args_d_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 2]] + args_d_[i_args + 3]) * blockDim.x], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 4]] + args_d_[i_args + 5]) * blockDim.x]);
                i_args += 6;
                break;
            }
            case 52: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp1
                Goldilocks3GPU::op_31_gpu(args_d_[i_args], &tmp3[args_d_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 2]] + args_d_[i_args + 3]) * blockDim.x], &tmp1[args_d_[i_args + 4] * blockDim.x]);
                i_args += 5;
                break;
            }
            case 53: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: public
                Goldilocks3GPU::op_31_gpu(args_d_[i_args], &tmp3[args_d_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 2]] + args_d_[i_args + 3]) * blockDim.x], &publics_d_[args_d_[i_args + 4] * blockDim.x]);
                i_args += 5;
                break;
            }
            case 54: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: number
                Goldilocks3GPU::op_31_gpu(args_d_[i_args], &tmp3[args_d_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 2]] + args_d_[i_args + 3]) * blockDim.x], &numbers_d_[args_d_[i_args + 4]*blockDim.x]);
                i_args += 5;
                break;
            }
            case 55: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: commit1
                Goldilocks3GPU::op_31_gpu(args_d_[i_args], &tmp3[args_d_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &tmp3[args_d_[i_args + 2] * blockDim.x * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 3]] + args_d_[i_args + 4]) * blockDim.x]);
                i_args += 5;
                break;
            }
            case 56: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp1
                Goldilocks3GPU::op_31_gpu(args_d_[i_args], &tmp3[args_d_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &tmp3[args_d_[i_args + 2] * blockDim.x * FIELD_EXTENSION], &tmp1[args_d_[i_args + 3] * blockDim.x]);
                i_args += 4;
                break;
            }
            case 57: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: public
                Goldilocks3GPU::op_31_gpu(args_d_[i_args], &tmp3[args_d_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &tmp3[args_d_[i_args + 2] * blockDim.x * FIELD_EXTENSION], &publics_d_[args_d_[i_args + 3] * blockDim.x]);
                i_args += 4;
                break;
            }
            case 58: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: number
                Goldilocks3GPU::op_31_gpu(args_d_[i_args], &tmp3[args_d_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &tmp3[args_d_[i_args + 2] * blockDim.x * FIELD_EXTENSION], &numbers_d_[args_d_[i_args + 3]*blockDim.x]);
                i_args += 4;
                break;
            }
            case 59: {
                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: commit1
                Goldilocks3GPU::op_31_gpu(args_d_[i_args], &tmp3[args_d_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &challenges_d_[args_d_[i_args + 2]*FIELD_EXTENSION*blockDim.x], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 3]] + args_d_[i_args + 4]) * blockDim.x]);
                i_args += 5;
                break;
            }
            case 60: {
                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: tmp1
                Goldilocks3GPU::op_31_gpu(args_d_[i_args], &tmp3[args_d_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &challenges_d_[args_d_[i_args + 2]*FIELD_EXTENSION*blockDim.x], &tmp1[args_d_[i_args + 3] * blockDim.x]);
                i_args += 4;
                break;
            }
            case 61: {
                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: public
                Goldilocks3GPU::op_31_gpu(args_d_[i_args], &tmp3[args_d_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &challenges_d_[args_d_[i_args + 2]*FIELD_EXTENSION*blockDim.x], &publics_d_[args_d_[i_args + 3] * blockDim.x]);
                i_args += 4;
                break;
            }
            case 62: {
                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: number
                Goldilocks3GPU::op_31_gpu(args_d_[i_args], &tmp3[args_d_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &challenges_d_[args_d_[i_args + 2]*FIELD_EXTENSION*blockDim.x], &numbers_d_[args_d_[i_args + 3]*blockDim.x]);
                i_args += 4;
                break;
            }
            case 63: {
                // COPY commit3 to tmp3
                Goldilocks3GPU::copy_gpu(&tmp3[args_d_[i_args] * blockDim.x * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 1]] + args_d_[i_args + 2]) * blockDim.x]);
                i_args += 3;
                break;
            }
            case 64: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit3
                Goldilocks3GPU::op_gpu(args_d_[i_args], &tmp3[args_d_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 2]] + args_d_[i_args + 3]) * blockDim.x], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 4]] + args_d_[i_args + 5]) * blockDim.x]);
                i_args += 6;
                break;
            }
            case 65: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp3
                Goldilocks3GPU::op_gpu(args_d_[i_args], &tmp3[args_d_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 2]] + args_d_[i_args + 3]) * blockDim.x], &tmp3[args_d_[i_args + 4] * blockDim.x * FIELD_EXTENSION]);
                i_args += 5;
                break;
            }
            case 66: {
                // MULTIPLICATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: challenge
                Goldilocks3GPU::mul_gpu(&tmp3[args_d_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 2]] + args_d_[i_args + 3]) * blockDim.x], &challenges_d_[args_d_[i_args + 4]*FIELD_EXTENSION*blockDim.x], &challenges_ops_d_[args_d_[i_args + 4]*FIELD_EXTENSION*blockDim.x]);
                i_args += 5;
                break;
            }
            case 67: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: challenge
                Goldilocks3GPU::op_gpu(args_d_[i_args], &tmp3[args_d_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 2]] + args_d_[i_args + 3]) * blockDim.x], &challenges_d_[args_d_[i_args + 4]*FIELD_EXTENSION*blockDim.x]);
                i_args += 5;
                break;
            }
            case 68: {
                // COPY tmp3 to tmp3
                Goldilocks3GPU::copy_gpu(&tmp3[args_d_[i_args] * blockDim.x * FIELD_EXTENSION], &tmp3[args_d_[i_args + 1] * blockDim.x * FIELD_EXTENSION]);
                i_args += 2;
                break;
            }
            case 69: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp3
                Goldilocks3GPU::op_gpu(args_d_[i_args], &tmp3[args_d_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &tmp3[args_d_[i_args + 2] * blockDim.x * FIELD_EXTENSION], &tmp3[args_d_[i_args + 3] * blockDim.x * FIELD_EXTENSION]);
                i_args += 4;
                break;
            }
            case 70: {
                // MULTIPLICATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: challenge
                Goldilocks3GPU::mul_gpu(&tmp3[args_d_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &tmp3[args_d_[i_args + 2] * blockDim.x * FIELD_EXTENSION], &challenges_d_[args_d_[i_args + 3]*FIELD_EXTENSION*blockDim.x], &challenges_ops_d_[args_d_[i_args + 3]*FIELD_EXTENSION*blockDim.x]);
                i_args += 4;
                break;
            }
            case 71: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: challenge
                Goldilocks3GPU::op_gpu(args_d_[i_args], &tmp3[args_d_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &tmp3[args_d_[i_args + 2] * blockDim.x * FIELD_EXTENSION], &challenges_d_[args_d_[i_args + 3]*FIELD_EXTENSION*blockDim.x]);
                i_args += 4;
                break;
            }
            case 72: {
                // MULTIPLICATION WITH DEST: tmp3 - SRC0: challenge - SRC1: challenge
                Goldilocks3GPU::mul_gpu(&tmp3[args_d_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &challenges_d_[args_d_[i_args + 2]*FIELD_EXTENSION*blockDim.x], &challenges_d_[args_d_[i_args + 3]*FIELD_EXTENSION*blockDim.x], &challenges_ops_d_[args_d_[i_args + 3]*FIELD_EXTENSION*blockDim.x]);
                i_args += 4;
                break;
            }
            case 73: {
                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: challenge
                Goldilocks3GPU::op_gpu(args_d_[i_args], &tmp3[args_d_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &challenges_d_[args_d_[i_args + 2]*FIELD_EXTENSION*blockDim.x], &challenges_d_[args_d_[i_args + 3]*FIELD_EXTENSION*blockDim.x]);
                i_args += 4;
                break;
            }
            case 74: {
                // COPY eval to tmp3
                Goldilocks3GPU::copy_gpu(&tmp3[args_d_[i_args] * blockDim.x * FIELD_EXTENSION], &evals_d_[args_d_[i_args + 1]*FIELD_EXTENSION*blockDim.x]);
                i_args += 2;
                break;
            }
            case 75: {
                // MULTIPLICATION WITH DEST: tmp3 - SRC0: eval - SRC1: challenge
                Goldilocks3GPU::mul_gpu(&tmp3[args_d_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &evals_d_[args_d_[i_args + 2]*FIELD_EXTENSION*blockDim.x], &challenges_d_[args_d_[i_args + 3]*FIELD_EXTENSION*blockDim.x], &challenges_ops_d_[args_d_[i_args + 3]*FIELD_EXTENSION*blockDim.x]);
                i_args += 4;
                break;
            }
            case 76: {
                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: eval
                Goldilocks3GPU::op_gpu(args_d_[i_args], &tmp3[args_d_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &challenges_d_[args_d_[i_args + 2]*FIELD_EXTENSION*blockDim.x], &evals_d_[args_d_[i_args + 3]*FIELD_EXTENSION*blockDim.x]);
                i_args += 4;
                break;
            }
            case 77: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: eval
                Goldilocks3GPU::op_gpu(args_d_[i_args], &tmp3[args_d_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &tmp3[args_d_[i_args + 2] * blockDim.x * FIELD_EXTENSION], &evals_d_[args_d_[i_args + 3]*FIELD_EXTENSION*blockDim.x]);
                i_args += 4;
                break;
            }
            case 78: {
                // OPERATION WITH DEST: tmp3 - SRC0: eval - SRC1: commit1
                Goldilocks3GPU::op_31_gpu(args_d_[i_args], &tmp3[args_d_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &evals_d_[args_d_[i_args + 2]*FIELD_EXTENSION*blockDim.x], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 3]] + args_d_[i_args + 4]) * blockDim.x]);
                i_args += 5;
                break;
            }
            case 79: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: eval
                Goldilocks3GPU::op_gpu(args_d_[i_args], &tmp3[args_d_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc_d_[args_d_[i_args + 2]] + args_d_[i_args + 3]) * blockDim.x], &evals_d_[args_d_[i_args + 4]*FIELD_EXTENSION*blockDim.x]);
                i_args += 5;
                break;
            }
            default: {
                //exit(1); 
                return;                
            }
            }
        }
        assert(i_args == nArgs);
    }
 
void CHelpersStepsGPU::dataSetup_(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams){

        uint32_t nrowsPack =  4;

        /*
            non-buffered data
        */
        uint8_t *ops = &parserArgs.ops[parserParams.opsOffset];
        uint32_t *ops_aux = new uint32_t[parserParams.nOps];
        for(uint64_t i = 0; i < parserParams.nOps; ++i) ops_aux[i] = uint32_t(ops[i]);
        cudaMalloc((void **) &(this->ops_d_), parserParams.nOps*sizeof(uint32_t));
        cudaMemcpy(this->ops_d_, ops_aux, parserParams.nOps*sizeof(uint32_t), cudaMemcpyHostToDevice);
        delete[] ops_aux;

        uint16_t *args = &parserArgs.args[parserParams.argsOffset];
        uint32_t *args_aux = new uint32_t[parserParams.nArgs];
        for(uint64_t i = 0; i < parserParams.nArgs; ++i) args_aux[i] = uint32_t(args[i]);
        cudaMalloc((void **) &(this->args_d_), parserParams.nArgs*sizeof(uint32_t));
        cudaMemcpy(this->args_d_, args_aux, parserParams.nArgs*sizeof(uint32_t), cudaMemcpyHostToDevice);
        delete[] args_aux;

        uint8_t *storePol = &parserArgs.storePols[parserParams.storePolsOffset];
        uint32_t *storePol_aux = new uint32_t[parserParams.nStorePols];
        for(uint64_t i = 0; i < parserParams.nStorePols; ++i) storePol_aux[i] = uint32_t(storePol[i]);
        cudaMalloc((void **) &(this->storePol_d_), parserParams.nStorePols*sizeof(uint32_t));
        cudaMemcpy(this->storePol_d_, storePol_aux, parserParams.nStorePols*sizeof(uint32_t), cudaMemcpyHostToDevice); 
        delete[] storePol_aux;

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

        cudaMalloc((void**) &(this->challenges_d_), params.challenges.degree()*FIELD_EXTENSION*nrowsPack*sizeof(gl64_t));
        cudaMemcpy(this->challenges_d_, challenges, params.challenges.degree()*FIELD_EXTENSION*nrowsPack*sizeof(gl64_t), cudaMemcpyHostToDevice);

        cudaMalloc((void**) &(this->challenges_ops_d_), params.challenges.degree()*FIELD_EXTENSION*nrowsPack*sizeof(gl64_t));
        cudaMemcpy(this->challenges_ops_d_, challenges_ops, params.challenges.degree()*FIELD_EXTENSION*nrowsPack*sizeof(gl64_t), cudaMemcpyHostToDevice);
    
        uint64_t *numbers = &parserArgs.numbers[parserParams.numbersOffset];
        Goldilocks::Element *numbers_aux = new Goldilocks::Element[parserParams.nNumbers*nrowsPack];
        for(uint64_t i = 0; i < parserParams.nNumbers; ++i) {
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                numbers_aux[i*nrowsPack + j] = Goldilocks::fromU64(numbers[i]);
            }
        }
        cudaMalloc((void **) &(this->numbers_d_), parserParams.nNumbers*nrowsPack*sizeof(gl64_t));
        cudaMemcpy(this->numbers_d_, numbers_aux, parserParams.nNumbers*nrowsPack*sizeof(gl64_t), cudaMemcpyHostToDevice);
        delete[] numbers_aux;

  #if 0       
        Goldilocks::Element publics_aux[starkInfo.nPublics*nrowsPack];
        for(uint64_t i = 0; i < starkInfo.nPublics; ++i) {
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                publics_aux[i*nrowsPack + j] = params.publicInputs[i];
            }
        }
        cudaMalloc((void **) &(this->publics_d_), starkInfo.nPublics*nrowsPack*sizeof(gl64_t));
        cudaMemcpy(this->publics_d_, publics_aux, starkInfo.nPublics*nrowsPack*sizeof(gl64_t), cudaMemcpyHostToDevice);
        delete[] publics_aux;

        Goldilocks::Element evals_aux[params.evals.degree()*FIELD_EXTENSION*nrowsPack];
        for(uint64_t i = 0; i < params.evals.degree(); ++i) {
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                evals_aux[(i*FIELD_EXTENSION)*nrowsPack + j] = params.evals[i][0];
                evals_aux[(i*FIELD_EXTENSION + 1)*nrowsPack + j] = params.evals[i][1];
                evals_aux[(i*FIELD_EXTENSION + 2)*nrowsPack + j] = params.evals[i][2];
            }
        }
        cudaMalloc((void **) &(this->evals_d_), params.evals.degree()*FIELD_EXTENSION*nrowsPack*sizeof(gl64_t));
        cudaMemcpy(this->evals_d_, evals_aux, params.evals.degree()*FIELD_EXTENSION*nrowsPack*sizeof(gl64_t), cudaMemcpyHostToDevice);
        delete[] evals_aux;

        /* 
            buffered data //assuming all fits in the GPU
        */
        cudaMalloc((void**) &(this->constPols_d_), params.pConstPols->numPols()*params.pConstPols->degree()*sizeof(gl64_t));
        cudaMemcpy(this->constPols_d_, params.pConstPols->address(), params.pConstPols->size(), cudaMemcpyHostToDevice);
        
        cudaMalloc((void**) &(this->constPols2ns_d_), params.pConstPols2ns->numPols()*params.pConstPols2ns->degree()*sizeof(gl64_t));
        cudaMemcpy(this->constPols2ns_d_, params.pConstPols2ns->address(), params.pConstPols2ns->size(), cudaMemcpyHostToDevice);

        cudaMalloc((void**) &(this->x_d_), params.x_n.dim()*params.x_n.degree()*sizeof(gl64_t));
        cudaMemcpy(this->x_d_, params.x_n.address(), params.x_n.size(), cudaMemcpyHostToDevice);

        cudaMalloc((void**) &(this->x_2ns_d_), params.x_2ns.dim()*params.x_2ns.degree()*sizeof(gl64_t));
        cudaMemcpy(this->x_2ns_d_, params.x_2ns.address(), params.x_2ns.size(), cudaMemcpyHostToDevice);

        cudaMalloc((void**) &(this->zi_d_), params.zi.dim()*params.zi.degree()*sizeof(gl64_t));
        cudaMemcpy(this->zi_d_, params.zi.address(), params.zi.size(), cudaMemcpyHostToDevice);

        cudaMalloc((void**) &(this->xDivXSubXi_d_), params.xDivXSubXi.dim()*params.xDivXSubXi.degree()*sizeof(gl64_t));
        cudaMemcpy(this->xDivXSubXi_d_, params.xDivXSubXi.address(), params.xDivXSubXi.size(), cudaMemcpyHostToDevice);

        cudaMalloc((void**) &(this->pols_d_), starkInfo.mapTotalN*sizeof(gl64_t));
        cudaMemcpy(this->pols_d_, params.pols, starkInfo.mapTotalN*sizeof(gl64_t), cudaMemcpyHostToDevice);
        
        /*
            temporal buffers
        */
        cudaMalloc(&bufferT_d_, numBlocks*sizeof(gl64_t*));
        for(uint64_t i = 0; i < numBlocks; ++i) {
            cudaMalloc(&bufferT_d[i], 2*nCols*nrowsPack*sizeof(gl64_t));
        }
        cudaMalloc(&tmp1_d, numBlocks*sizeof(gl64_t*));
        for(uint64_t i = 0; i < numBlocks; ++i) {
            cudaMalloc(&tmp1_d[i], parserParams.nTemp1*nrowsPack*sizeof(gl64_t));
        }
        cudaMalloc(&tmp3_d, numBlocks*sizeof(gl64_t*));
        for(uint64_t i = 0; i < numBlocks; ++i) {
            cudaMalloc(&tmp3_d[i], parserParams.nTemp3*nrowsPack*FIELD_EXTENSION*sizeof(gl64_t));
        }
#endif
}    
