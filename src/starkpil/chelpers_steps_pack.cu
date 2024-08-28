#include "zklog.hpp"
#include <inttypes.h>

#if defined(__USE_CUDA__) && defined(ENABLE_EXPERIMENTAL_CODE)
#ifdef __AVX512__
#include "chelpers_steps_avx512.hpp"
#endif
#include "chelpers_steps_pack.cuh"
#include "goldilocks_cubic_extension.cuh"
#include "cuda_utils.cuh"
#include "cuda_utils.hpp"
#include "timer.hpp"

const uint64_t MAX_U64 = 0xFFFFFFFFFFFFFFFF;

CHelpersStepsPackGPU *cHelpersSteps[MAX_GPUS];
uint64_t *gpuSharedStorage[MAX_GPUS];
uint64_t *streamExclusiveStorage[nStreams*MAX_GPUS];
cudaStream_t streams[nStreams*MAX_GPUS];

void CHelpersStepsPackGPU::prepareGPU(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams) {

    prepare(starkInfo, params, parserArgs, parserParams);


    CHECKCUDAERR(cudaGetDeviceCount(&nDevices));
    printf("nDevices: %d\n", nDevices);
    nCudaThreads = 1<<15;
    domainExtended = parserParams.stage > 3 ? true : false;
    domainSize = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
    subDomainSize = nrowsPack * nCudaThreads;
    nextStride = domainExtended ? 1 << (starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits) : 1;

    nOps = parserParams.nOps;
    nArgs = parserParams.nArgs;
    nBufferT = 2*nCols*nrowsPack;
    nTemp1 = parserParams.nTemp1*nrowsPack;
    nTemp3 = parserParams.nTemp3*FIELD_EXTENSION*nrowsPack;

    printf("nCols:%lu\n", nCols);
    printf("nrowsPack:%lu\n", nrowsPack);

    offsetsStagesGPU.resize(offsetsStages.size());
    uint64_t total_pols = 0;
    for (uint64_t s = 1; s < 11; s++) {
        if (s < 4 || (s == 4 && parserParams.stage != 4) || (s == 10 && domainExtended)) {
            printf("s=%lu, offsets=%lu\n", s, total_pols);
            offsetsStagesGPU[s] = total_pols;
            total_pols += nColsStages[s] * (nrowsPack * nCudaThreads + nextStride);
        } else {
            offsetsStagesGPU[s] = MAX_U64;
        }
    }

    printf("total_pols:%lu\n", total_pols);

    sharedStorageSize = 0;
    ops_offset = sharedStorageSize;
    sharedStorageSize += nOps;

    args_offset = sharedStorageSize;
    sharedStorageSize += nArgs;

    offsetsStages_offset = sharedStorageSize;
    sharedStorageSize += offsetsStages.size();

    nColsStages_offset = sharedStorageSize;
    sharedStorageSize += nColsStages.size();

    nColsStagesAcc_offset = sharedStorageSize;
    sharedStorageSize += nColsStagesAcc.size();

    challenges_offset = sharedStorageSize;
    sharedStorageSize += challenges.size();

    challenges_ops_offset = sharedStorageSize;
    sharedStorageSize += challenges_ops.size();

    numbers_offset = sharedStorageSize;
    sharedStorageSize += numbers_.size();

    publics_offset = sharedStorageSize;
    sharedStorageSize += publics.size();

    evals_offset = sharedStorageSize;
    sharedStorageSize += evals.size();

    uint64_t *ops64 = (uint64_t *)malloc(nOps * sizeof(uint64_t));
    for (uint32_t i=0; i<nOps; i++) {
        ops64[i] = uint64_t(parserArgs.ops[parserParams.opsOffset+i]);
    }
    uint64_t *args64 = (uint64_t *)malloc(nArgs * sizeof(uint64_t));
    for (uint32_t i=0; i<nArgs; i++) {
        args64[i] = uint64_t(parserArgs.args[parserParams.argsOffset+i]);
    }

    for (int d=0;d<nDevices;d++) {
        CHECKCUDAERR(cudaSetDevice(d));
        CHECKCUDAERR(cudaMalloc(&gpuSharedStorage[d], sharedStorageSize * sizeof(uint64_t)));
        CHECKCUDAERR(cudaMemcpy(gpuSharedStorage[d]+ops_offset, ops64, nOps * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpy(gpuSharedStorage[d]+args_offset, args64, nArgs * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpy(gpuSharedStorage[d]+offsetsStages_offset, offsetsStagesGPU.data(), offsetsStagesGPU.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpy(gpuSharedStorage[d]+nColsStages_offset, nColsStages.data(), nColsStages.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpy(gpuSharedStorage[d]+nColsStagesAcc_offset, nColsStagesAcc.data(), nColsStagesAcc.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpy(gpuSharedStorage[d]+challenges_offset, challenges.data(), challenges.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpy(gpuSharedStorage[d]+challenges_ops_offset, challenges_ops.data(), challenges_ops.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpy(gpuSharedStorage[d]+numbers_offset, numbers_.data(), numbers_.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpy(gpuSharedStorage[d]+publics_offset, publics.data(), publics.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CHECKCUDAERR(cudaMemcpy(gpuSharedStorage[d]+evals_offset, evals.data(), evals.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    }

    free(ops64);
    free(args64);


    exclusiveStorageSize = 0;
    constPols_offset = exclusiveStorageSize;
    exclusiveStorageSize += starkInfo.nConstants * (subDomainSize + nextStride);

    x_offset = exclusiveStorageSize;
    exclusiveStorageSize += subDomainSize;

    zi_offset = exclusiveStorageSize;
    exclusiveStorageSize += subDomainSize;

    pols_offset = exclusiveStorageSize;
    exclusiveStorageSize += total_pols;

    xDivXSubXi_offset = exclusiveStorageSize;
    exclusiveStorageSize += 2 * subDomainSize * FIELD_EXTENSION;

    bufferT_offset = exclusiveStorageSize;
    exclusiveStorageSize += nBufferT * nCudaThreads;

    tmp1_offset = exclusiveStorageSize;
    exclusiveStorageSize += nTemp1 * nCudaThreads;

    tmp3_offset = exclusiveStorageSize;
    exclusiveStorageSize += nTemp3 * nCudaThreads;

    for (uint32_t s = 0; s < nStreams*nDevices; s++) {
        CHECKCUDAERR(cudaSetDevice(s/nStreams));
        CHECKCUDAERR(cudaMalloc(&streamExclusiveStorage[s], exclusiveStorageSize * sizeof(uint64_t)));
    }

    for (int d=0;d<nDevices;d++) {
        CHECKCUDAERR(cudaSetDevice(d));
        CHECKCUDAERR(cudaMalloc((void **)&(cHelpersSteps[d]), sizeof(CHelpersStepsPackGPU)));
        CHECKCUDAERR(cudaMemcpy(cHelpersSteps[d], this, sizeof(CHelpersStepsPackGPU), cudaMemcpyHostToDevice));
    }

    for (uint32_t s = 0; s < nStreams*nDevices; s++) {
        CHECKCUDAERR(cudaSetDevice(s/nStreams));
        CHECKCUDAERR(cudaStreamCreate(&streams[s]));
    }
}

void CHelpersStepsPackGPU::cleanupGPU() {
    CHECKCUDAERR(cudaGetDeviceCount(&nDevices));
    for (int d=0;d<nDevices;d++) {
        cudaFree(gpuSharedStorage[d]);
        cudaFree(cHelpersSteps[d]);
    }

    for (uint32_t s = 0; s < nStreams*nDevices; s++) {
        cudaFree(streamExclusiveStorage[s]);
    }

    for (uint32_t s = 0; s < nStreams*nDevices; s++) {
        CHECKCUDAERR(cudaStreamDestroy(streams[s]));
    }
}


void CHelpersStepsPackGPU::calculateExpressions(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams) {

    if (!starkInfo.reduceMemory || parserParams.stage == 2) { // in these cases, cpu version is faster
#ifdef __AVX512__
        CHelpersStepsAvx512 cHelpersSteps;
#else
        CHelpersSteps cHelpersSteps;
#endif
        return cHelpersSteps.calculateExpressions(starkInfo, params, parserArgs, parserParams);
    }

    prepareGPU(starkInfo, params, parserArgs, parserParams);
    calculateExpressionsRowsGPU(starkInfo, params, parserArgs, parserParams, 0, domainSize);
    cleanupGPU();
}

void CHelpersStepsPackGPU::calculateExpressionsRowsGPU(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams,
    uint64_t rowIni, uint64_t rowEnd){

    if(rowEnd < rowIni || rowEnd > domainSize || (rowEnd -rowIni) % nrowsPack != 0) {
        zklog.info("Invalid range for rowIni " + to_string(rowIni) + " and rowEnd " + to_string(rowEnd));
        exitProcess();
    }

    assert((rowEnd - rowIni) % (nrowsPack*nCudaThreads*nStreams*nDevices) == 0);
    uint64_t nrowPerStream = (rowEnd - rowIni) / nStreams /nDevices;

    for (int s=0; s<nStreams*nDevices; s++) {
        int d = s/nStreams;
        CHECKCUDAERR(cudaSetDevice(d));
        CHelpersStepsPackGPU *cHelpersSteps_d = cHelpersSteps[d];
        uint64_t *sharedStorage = gpuSharedStorage[d];
        uint64_t *exclusiveStorage = streamExclusiveStorage[s];
        cudaStream_t stream = streams[s];
        //TimerStart(STREAM_OPS);
        for (uint64_t i = rowIni+s*nrowPerStream; i < rowIni+(s+1)*nrowPerStream; i+= nrowsPack*nCudaThreads) {
            //TimerStart(Memcpy_H_to_D);
            loadData(starkInfo, params, i, s);
            //TimerStopAndLog(Memcpy_H_to_D);

            //TimerStart(EXP_Kernel);
            loadPolinomialsGPU<<<(nCudaThreads+15)/16,16,0,stream>>>(cHelpersSteps_d, sharedStorage, exclusiveStorage, starkInfo.nConstants, parserParams.stage);
            pack_kernel<<<(nCudaThreads+15)/16,16,0,stream>>>(cHelpersSteps_d, sharedStorage, exclusiveStorage);
            storePolinomialsGPU<<<(nCudaThreads+15)/16,16,0,stream>>>(cHelpersSteps_d, sharedStorage, exclusiveStorage);
            //TimerStopAndLog(EXP_Kernel);

            //TimerStart(Memcpy_D_to_H);
            storeData(starkInfo, params, i, s);
            //TimerStopAndLog(Memcpy_D_to_H);
        }
        //TimerStopAndLog(STREAM_OPS);
    }


    TimerStart(WAIT_STREAM);
    for (uint32_t s = 0; s < nStreams*nDevices; s++) {
        CHECKCUDAERR(cudaStreamSynchronize(streams[s]));
    }
    TimerStopAndLog(WAIT_STREAM);
}

void CHelpersStepsPackGPU::loadData(StarkInfo &starkInfo, StepsParams &params, uint64_t row, uint32_t s) {

    ConstantPolsStarks *constPols = domainExtended ? params.pConstPols2ns : params.pConstPols;
    Polinomial &x = domainExtended ? params.x_2ns : params.x_n;

    uint64_t *exclusiveStorage = streamExclusiveStorage[s];
    uint64_t *constPols_d = exclusiveStorage + constPols_offset;
    uint64_t *x_d = exclusiveStorage + x_offset;
    uint64_t *zi_d = exclusiveStorage + zi_offset;
    uint64_t *pols_d = exclusiveStorage + pols_offset;
    uint64_t *xDivXSubXi_d = exclusiveStorage + xDivXSubXi_offset;

    cudaStream_t stream = streams[s];

    if (row + subDomainSize != domainSize) {
        CHECKCUDAERR(cudaMemcpyAsync(constPols_d, ((Goldilocks::Element *)constPols->address()) + row * starkInfo.nConstants, starkInfo.nConstants * (subDomainSize + nextStride) * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
    } else {
        CHECKCUDAERR(cudaMemcpyAsync(constPols_d, ((Goldilocks::Element *)constPols->address()) + row * starkInfo.nConstants, starkInfo.nConstants * subDomainSize * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
        CHECKCUDAERR(cudaMemcpyAsync(constPols_d + starkInfo.nConstants * subDomainSize, (Goldilocks::Element *)constPols->address(), starkInfo.nConstants * nextStride * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
    }

    CHECKCUDAERR(cudaMemcpyAsync(x_d, x[row], subDomainSize * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
    CHECKCUDAERR(cudaMemcpyAsync(zi_d, params.zi[row], subDomainSize * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));

    for (uint64_t s = 1; s < 11; s++) {
        if (offsetsStagesGPU[s] != MAX_U64) {
            if (row + subDomainSize != domainSize) {
                CHECKCUDAERR(cudaMemcpyAsync(pols_d + offsetsStagesGPU[s], &params.pols[offsetsStages[s] + row*nColsStages[s]], (subDomainSize+nextStride) *nColsStages[s] * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
            } else {
                CHECKCUDAERR(cudaMemcpyAsync(pols_d + offsetsStagesGPU[s], &params.pols[offsetsStages[s] + row*nColsStages[s]], subDomainSize *nColsStages[s] * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
                CHECKCUDAERR(cudaMemcpyAsync(pols_d + offsetsStagesGPU[s] + subDomainSize *nColsStages[s], &params.pols[offsetsStages[s]], nextStride *nColsStages[s] * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
            }
        }
    }

    CHECKCUDAERR(cudaMemcpyAsync(xDivXSubXi_d, params.xDivXSubXi[row], subDomainSize *FIELD_EXTENSION * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
    CHECKCUDAERR(cudaMemcpyAsync(xDivXSubXi_d + subDomainSize *FIELD_EXTENSION, params.xDivXSubXi[domainSize + row], subDomainSize *FIELD_EXTENSION * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
}

void CHelpersStepsPackGPU::storeData(StarkInfo &starkInfo, StepsParams &params, uint64_t row, uint32_t s) {
    uint64_t *pols_d = streamExclusiveStorage[s] + pols_offset;
    cudaStream_t stream = streams[s];
    for (uint64_t s = 1; s < 11; s++) {
        if (offsetsStagesGPU[s] != MAX_U64) {
            CHECKCUDAERR(cudaMemcpyAsync(&params.pols[offsetsStages[s] + row*nColsStages[s]], pols_d + offsetsStagesGPU[s], subDomainSize *nColsStages[s] * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
        }
    }
}

__global__ void loadPolinomialsGPU(CHelpersStepsPackGPU *cHelpersSteps, uint64_t *sharedStorage, uint64_t *exclusiveStorage, uint64_t nConstants, uint64_t stage) {

    uint64_t nCudaThreads = cHelpersSteps->nCudaThreads;

    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nCudaThreads) {
        return;
    }

    uint64_t nrowsPack = cHelpersSteps->nrowsPack;
    uint64_t nextStride = cHelpersSteps->nextStride;
    uint64_t subDomainSize = cHelpersSteps->subDomainSize;
    uint64_t nBufferT = cHelpersSteps->nBufferT;

    uint64_t *nColsStages = sharedStorage + cHelpersSteps->nColsStages_offset;
    uint64_t *nColsStagesAcc = sharedStorage + cHelpersSteps->nColsStagesAcc_offset;
    uint64_t *offsetsStages = sharedStorage + cHelpersSteps->offsetsStages_offset;

    gl64_t *bufferT_ = (gl64_t *)exclusiveStorage + cHelpersSteps->bufferT_offset + idx * nBufferT;
    gl64_t *pols = (gl64_t *)exclusiveStorage + cHelpersSteps->pols_offset;
    gl64_t *constPols = (gl64_t *)exclusiveStorage + cHelpersSteps->constPols_offset;

    uint64_t row = idx*nrowsPack;
    uint64_t nStages = 3;
    uint64_t nextStrides[2] = {0, nextStride};

    for(uint64_t k = 0; k < nConstants; ++k) {
        for(uint64_t o = 0; o < 2; ++o) {
            for(uint64_t j = 0; j < nrowsPack; ++j) {
                uint64_t l = (row + j + nextStrides[o]);
                bufferT_[(nColsStagesAcc[5*o] + k)*nrowsPack + j] = constPols[l * nConstants + k];
            }
        }
    }

    // Load x and Zi
    for(uint64_t j = 0; j < nrowsPack; ++j) {
        bufferT_[nConstants*nrowsPack + j] = (exclusiveStorage + cHelpersSteps->x_offset)[row + j];
    }
    for(uint64_t j = 0; j < nrowsPack; ++j) {
        bufferT_[(nConstants + 1)*nrowsPack + j] = (exclusiveStorage + cHelpersSteps->zi_offset)[row + j];
    }

    for(uint64_t s = 1; s <= nStages; ++s) {
        for(uint64_t k = 0; k < nColsStages[s]; ++k) {
            for(uint64_t o = 0; o < 2; ++o) {
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    uint64_t l = (row + j + nextStrides[o]);
                    bufferT_[(nColsStagesAcc[5*o + s] + k)*nrowsPack + j] = pols[offsetsStages[s] + l * nColsStages[s] + k];
                }
            }
        }
    }

    if(stage == 5) {
        for(uint64_t k = 0; k < nColsStages[nStages + 1]; ++k) {
            for(uint64_t o = 0; o < 2; ++o) {
                for(uint64_t j = 0; j < nrowsPack; ++j) {
                    uint64_t l = (row + j + nextStrides[o]); // % domainSize;
                    bufferT_[(nColsStagesAcc[5*o + nStages + 1] + k)*nrowsPack + j] = pols[offsetsStages[nStages + 1] + l * nColsStages[nStages + 1] + k];
                }
            }
        }

       // Load xDivXSubXi & xDivXSubWXi
       for(uint64_t d = 0; d < 2; ++d) {
           for(uint64_t i = 0; i < FIELD_EXTENSION; ++i) {
               for(uint64_t j = 0; j < nrowsPack; ++j) {
                  bufferT_[(nColsStagesAcc[11] + FIELD_EXTENSION*d + i)*nrowsPack + j] = (exclusiveStorage + cHelpersSteps->xDivXSubXi_offset)[(d*subDomainSize + row + j) * FIELD_EXTENSION + i];
               }
           }
       }
    }
}

__global__ void storePolinomialsGPU(CHelpersStepsPackGPU *cHelpersSteps, uint64_t *sharedStorage, uint64_t *exclusiveStorage) {
    uint64_t nCudaThreads = cHelpersSteps->nCudaThreads;

    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nCudaThreads) {
        return;
    }

    bool domainExtended = cHelpersSteps->domainExtended;
    uint64_t nrowsPack = cHelpersSteps->nrowsPack;
    uint64_t nBufferT = cHelpersSteps->nBufferT;

    uint64_t row = idx*nrowsPack;

    uint64_t *nColsStages = sharedStorage + cHelpersSteps->nColsStages_offset;
    uint64_t *nColsStagesAcc = sharedStorage + cHelpersSteps->nColsStagesAcc_offset;
    uint64_t *offsetsStages = sharedStorage + cHelpersSteps->offsetsStages_offset;

    gl64_t *bufferT_ = (gl64_t *)exclusiveStorage + cHelpersSteps->bufferT_offset + idx * nBufferT;
    gl64_t *pols = (gl64_t *)exclusiveStorage + cHelpersSteps->pols_offset;

    if(domainExtended) {
        // Store either polinomial f or polinomial q
        for(uint64_t k = 0; k < nColsStages[10]; ++k) {
            gl64_t *buffT = &bufferT_[(nColsStagesAcc[10] + k)* nrowsPack];
            gl64_t::copy_pack(nrowsPack, &pols[offsetsStages[10] + k + row * nColsStages[10]], nColsStages[10], buffT);
        }
    } else {
        uint64_t nStages = 3;
        for(uint64_t s = 2; s <= nStages + 1; ++s) {
            for(uint64_t k = 0; k < nColsStages[s]; ++k) {
                gl64_t *buffT = &bufferT_[(nColsStagesAcc[s] + k)* nrowsPack];
                gl64_t::copy_pack(nrowsPack, &pols[offsetsStages[s] + k + row * nColsStages[s]], nColsStages[s], buffT);
            }
        }
    }
}

__global__ void pack_kernel(CHelpersStepsPackGPU *cHelpersSteps, uint64_t *sharedStorage, uint64_t *exclusiveStorage)
{
    uint64_t nCudaThreads = cHelpersSteps->nCudaThreads;

    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nCudaThreads) {
        return;
    }

    uint64_t nrowsPack = cHelpersSteps->nrowsPack;
    uint64_t nOps = cHelpersSteps->nOps;
    uint64_t nArgs = cHelpersSteps->nArgs;
    uint64_t nBufferT = cHelpersSteps->nBufferT;
    uint64_t nTemp1 = cHelpersSteps->nTemp1;
    uint64_t nTemp3 = cHelpersSteps->nTemp3;

    uint64_t *nColsStagesAcc = sharedStorage + cHelpersSteps->nColsStagesAcc_offset;
    uint64_t *ops = sharedStorage + cHelpersSteps->ops_offset;
    uint64_t *args = sharedStorage + cHelpersSteps->args_offset;
    gl64_t *challenges = (gl64_t *)sharedStorage + cHelpersSteps->challenges_offset;
    gl64_t *challenges_ops = (gl64_t *)sharedStorage + cHelpersSteps->challenges_ops_offset;
    gl64_t *numbers_ = (gl64_t *)sharedStorage + cHelpersSteps->numbers_offset;
    gl64_t *publics = (gl64_t *)sharedStorage + cHelpersSteps->publics_offset;
    gl64_t *evals = (gl64_t *)sharedStorage + cHelpersSteps->evals_offset;

    gl64_t *bufferT_ = (gl64_t *)exclusiveStorage + cHelpersSteps->bufferT_offset + idx * nBufferT;
    gl64_t *tmp1 = (gl64_t *)exclusiveStorage + cHelpersSteps->tmp1_offset + nTemp1*idx;
    gl64_t *tmp3 = (gl64_t *)exclusiveStorage + cHelpersSteps->tmp3_offset + nTemp3*idx;

    uint64_t i_args = 0;

    for (uint64_t kk = 0; kk < nOps; ++kk) {
        switch (ops[kk]) {
            case 0: {
                // COPY commit1 to commit1
                gl64_t::copy_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args[i_args]] + args[i_args + 1]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack]);
                i_args += 4;
                break;
            }
            case 1: {
                // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: commit1
                gl64_t::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 5]] + args[i_args + 6]) * nrowsPack]);
                i_args += 7;
                break;
            }
            case 2: {
                // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: tmp1
                gl64_t::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &tmp1[args[i_args + 5] * nrowsPack]);
                i_args += 6;
                break;
            }
            case 3: {
                // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: public
                gl64_t::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &publics[args[i_args + 5] * nrowsPack]);
                i_args += 6;
                break;
            }
            case 4: {
                // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: number
                gl64_t::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &numbers_[args[i_args + 5]*nrowsPack]);
                i_args += 6;
                break;
            }
            case 5: {
                // COPY tmp1 to commit1
                gl64_t::copy_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args[i_args]] + args[i_args + 1]) * nrowsPack], &tmp1[args[i_args + 2] * nrowsPack]);
                i_args += 3;
                break;
            }
            case 6: {
                // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: tmp1
                gl64_t::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &tmp1[args[i_args + 3] * nrowsPack], &tmp1[args[i_args + 4] * nrowsPack]);
                i_args += 5;
                break;
            }
            case 7: {
                // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: public
                gl64_t::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &tmp1[args[i_args + 3] * nrowsPack], &publics[args[i_args + 4] * nrowsPack]);
                i_args += 5;
                break;
            }
            case 8: {
                // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: number
                gl64_t::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &tmp1[args[i_args + 3] * nrowsPack], &numbers_[args[i_args + 4]*nrowsPack]);
                i_args += 5;
                break;
            }
            case 9: {
                // COPY public to commit1
                gl64_t::copy_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args[i_args]] + args[i_args + 1]) * nrowsPack], &publics[args[i_args + 2] * nrowsPack]);
                i_args += 3;
                break;
            }
            case 10: {
                // OPERATION WITH DEST: commit1 - SRC0: public - SRC1: public
                gl64_t::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &publics[args[i_args + 3] * nrowsPack], &publics[args[i_args + 4] * nrowsPack]);
                i_args += 5;
                break;
            }
            case 11: {
                // OPERATION WITH DEST: commit1 - SRC0: public - SRC1: number
                gl64_t::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &publics[args[i_args + 3] * nrowsPack], &numbers_[args[i_args + 4]*nrowsPack]);
                i_args += 5;
                break;
            }
            case 12: {
                // COPY number to commit1
                gl64_t::copy_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args[i_args]] + args[i_args + 1]) * nrowsPack], &numbers_[args[i_args + 2]*nrowsPack]);
                i_args += 3;
                break;
            }
            case 13: {
                // OPERATION WITH DEST: commit1 - SRC0: number - SRC1: number
                gl64_t::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &numbers_[args[i_args + 3]*nrowsPack], &numbers_[args[i_args + 4]*nrowsPack]);
                i_args += 5;
                break;
            }
            case 14: {
                // COPY commit1 to tmp1
                gl64_t::copy_pack(nrowsPack, &tmp1[args[i_args] * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack]);
                i_args += 3;
                break;
            }
            case 15: {
                // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: commit1
                gl64_t::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]) * nrowsPack]);
                i_args += 6;
                break;
            }
            case 16: {
                // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: tmp1
                gl64_t::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &tmp1[args[i_args + 4] * nrowsPack]);
                i_args += 5;
                break;
            }
            case 17: {
                // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: public
                gl64_t::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &publics[args[i_args + 4] * nrowsPack]);
                i_args += 5;
                break;
            }
            case 18: {
                // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: number
                gl64_t::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &numbers_[args[i_args + 4]*nrowsPack]);
                i_args += 5;
                break;
            }
            case 19: {
                // COPY tmp1 to tmp1
                gl64_t::copy_pack(nrowsPack, &tmp1[args[i_args] * nrowsPack], &tmp1[args[i_args + 1] * nrowsPack]);
                i_args += 2;
                break;
            }
            case 20: {
                // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: tmp1
                gl64_t::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &tmp1[args[i_args + 2] * nrowsPack], &tmp1[args[i_args + 3] * nrowsPack]);
                i_args += 4;
                break;
            }
            case 21: {
                // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: public
                gl64_t::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &tmp1[args[i_args + 2] * nrowsPack], &publics[args[i_args + 3] * nrowsPack]);
                i_args += 4;
                break;
            }
            case 22: {
                // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: number
                gl64_t::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &tmp1[args[i_args + 2] * nrowsPack], &numbers_[args[i_args + 3]*nrowsPack]);
                i_args += 4;
                break;
            }
            case 23: {
                // COPY public to tmp1
                gl64_t::copy_pack(nrowsPack, &tmp1[args[i_args] * nrowsPack], &publics[args[i_args + 1] * nrowsPack]);
                i_args += 2;
                break;
            }
            case 24: {
                // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: public
                gl64_t::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &publics[args[i_args + 2] * nrowsPack], &publics[args[i_args + 3] * nrowsPack]);
                i_args += 4;
                break;
            }
            case 25: {
                // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: number
                gl64_t::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &publics[args[i_args + 2] * nrowsPack], &numbers_[args[i_args + 3]*nrowsPack]);
                i_args += 4;
                break;
            }
            case 26: {
                // COPY number to tmp1
                gl64_t::copy_pack(nrowsPack, &tmp1[args[i_args] * nrowsPack], &numbers_[args[i_args + 1]*nrowsPack]);
                i_args += 2;
                break;
            }
            case 27: {
                // OPERATION WITH DEST: tmp1 - SRC0: number - SRC1: number
                gl64_t::op_pack(nrowsPack, args[i_args], &tmp1[args[i_args + 1] * nrowsPack], &numbers_[args[i_args + 2]*nrowsPack], &numbers_[args[i_args + 3]*nrowsPack]);
                i_args += 4;
                break;
            }
            case 28: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: commit1
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 5]] + args[i_args + 6]) * nrowsPack]);
                i_args += 7;
                break;
            }
            case 29: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: tmp1
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &tmp1[args[i_args + 5] * nrowsPack]);
                i_args += 6;
                break;
            }
            case 30: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: public
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &publics[args[i_args + 5] * nrowsPack]);
                i_args += 6;
                break;
            }
            case 31: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: number
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &numbers_[args[i_args + 5]*nrowsPack]);
                i_args += 6;
                break;
            }
            case 32: {
                // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: commit1
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &tmp3[args[i_args + 3] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]) * nrowsPack]);
                i_args += 6;
                break;
            }
            case 33: {
                // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: tmp1
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &tmp3[args[i_args + 3] * nrowsPack * FIELD_EXTENSION], &tmp1[args[i_args + 4] * nrowsPack]);
                i_args += 5;
                break;
            }
            case 34: {
                // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: public
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &tmp3[args[i_args + 3] * nrowsPack * FIELD_EXTENSION], &publics[args[i_args + 4] * nrowsPack]);
                i_args += 5;
                break;
            }
            case 35: {
                // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: number
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &tmp3[args[i_args + 3] * nrowsPack * FIELD_EXTENSION], &numbers_[args[i_args + 4]*nrowsPack]);
                i_args += 5;
                break;
            }
            case 36: {
                // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: commit1
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]) * nrowsPack]);
                i_args += 6;
                break;
            }
            case 37: {
                // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: tmp1
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack], &tmp1[args[i_args + 4] * nrowsPack]);
                i_args += 5;
                break;
            }
            case 38: {
                // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: public
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack], &publics[args[i_args + 4] * nrowsPack]);
                i_args += 5;
                break;
            }
            case 39: {
                // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: number
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack], &numbers_[args[i_args + 4]*nrowsPack]);
                i_args += 5;
                break;
            }
            case 40: {
                // COPY commit3 to commit3
                Goldilocks3GPU::copy_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args[i_args]] + args[i_args + 1]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack]);
                i_args += 4;
                break;
            }
            case 41: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: commit3
                Goldilocks3GPU::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 5]] + args[i_args + 6]) * nrowsPack]);
                i_args += 7;
                break;
            }
            case 42: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: tmp3
                Goldilocks3GPU::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &tmp3[args[i_args + 5] * nrowsPack * FIELD_EXTENSION]);
                i_args += 6;
                break;
            }
            case 43: {
                // MULTIPLICATION WITH DEST: commit3 - SRC0: commit3 - SRC1: challenge
                Goldilocks3GPU::mul_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &challenges[args[i_args + 5]*FIELD_EXTENSION*nrowsPack], &challenges_ops[args[i_args + 5]*FIELD_EXTENSION*nrowsPack]);
                i_args += 6;
                break;
            }
            case 44: {
                // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: challenge
                Goldilocks3GPU::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack], &challenges[args[i_args + 5]*FIELD_EXTENSION*nrowsPack]);
                i_args += 6;
                break;
            }
            case 45: {
                // COPY tmp3 to commit3
                Goldilocks3GPU::copy_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args[i_args]] + args[i_args + 1]) * nrowsPack], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION]);
                i_args += 3;
                break;
            }
            case 46: {
                // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: tmp3
                Goldilocks3GPU::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &tmp3[args[i_args + 3] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 4] * nrowsPack * FIELD_EXTENSION]);
                i_args += 5;
                break;
            }
            case 47: {
                // MULTIPLICATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: challenge
                Goldilocks3GPU::mul_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &tmp3[args[i_args + 3] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 4]*FIELD_EXTENSION*nrowsPack], &challenges_ops[args[i_args + 4]*FIELD_EXTENSION*nrowsPack]);
                i_args += 5;
                break;
            }
            case 48: {
                // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: challenge
                Goldilocks3GPU::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &tmp3[args[i_args + 3] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 4]*FIELD_EXTENSION*nrowsPack]);
                i_args += 5;
                break;
            }
            case 49: {
                // MULTIPLICATION WITH DEST: commit3 - SRC0: challenge - SRC1: challenge
                Goldilocks3GPU::mul_pack(nrowsPack, &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack], &challenges[args[i_args + 4]*FIELD_EXTENSION*nrowsPack], &challenges_ops[args[i_args + 4]*FIELD_EXTENSION*nrowsPack]);
                i_args += 5;
                break;
            }
            case 50: {
                // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: challenge
                Goldilocks3GPU::op_pack(nrowsPack, args[i_args], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack], &challenges[args[i_args + 4]*FIELD_EXTENSION*nrowsPack]);
                i_args += 5;
                break;
            }
            case 51: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit1
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]) * nrowsPack]);
                i_args += 6;
                break;
            }
            case 52: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp1
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &tmp1[args[i_args + 4] * nrowsPack]);
                i_args += 5;
                break;
            }
            case 53: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: public
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &publics[args[i_args + 4] * nrowsPack]);
                i_args += 5;
                break;
            }
            case 54: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: number
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &numbers_[args[i_args + 4]*nrowsPack]);
                i_args += 5;
                break;
            }
            case 55: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: commit1
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack]);
                i_args += 5;
                break;
            }
            case 56: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp1
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], &tmp1[args[i_args + 3] * nrowsPack]);
                i_args += 4;
                break;
            }
            case 57: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: public
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], &publics[args[i_args + 3] * nrowsPack]);
                i_args += 4;
                break;
            }
            case 58: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: number
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], &numbers_[args[i_args + 3]*nrowsPack]);
                i_args += 4;
                break;
            }
            case 59: {
                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: commit1
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack]);
                i_args += 5;
                break;
            }
            case 60: {
                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: tmp1
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &tmp1[args[i_args + 3] * nrowsPack]);
                i_args += 4;
                break;
            }
            case 61: {
                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: public
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &publics[args[i_args + 3] * nrowsPack]);
                i_args += 4;
                break;
            }
            case 62: {
                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: number
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &numbers_[args[i_args + 3]*nrowsPack]);
                i_args += 4;
                break;
            }
            case 63: {
                // COPY commit3 to tmp3
                Goldilocks3GPU::copy_pack(nrowsPack, &tmp3[args[i_args] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 1]] + args[i_args + 2]) * nrowsPack]);
                i_args += 3;
                break;
            }
            case 64: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit3
                Goldilocks3GPU::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 4]] + args[i_args + 5]) * nrowsPack]);
                i_args += 6;
                break;
            }
            case 65: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp3
                Goldilocks3GPU::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &tmp3[args[i_args + 4] * nrowsPack * FIELD_EXTENSION]);
                i_args += 5;
                break;
            }
            case 66: {
                // MULTIPLICATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: challenge
                Goldilocks3GPU::mul_pack(nrowsPack, &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &challenges[args[i_args + 4]*FIELD_EXTENSION*nrowsPack], &challenges_ops[args[i_args + 4]*FIELD_EXTENSION*nrowsPack]);
                i_args += 5;
                break;
            }
            case 67: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: challenge
                Goldilocks3GPU::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &challenges[args[i_args + 4]*FIELD_EXTENSION*nrowsPack]);
                i_args += 5;
                break;
            }
            case 68: {
                // COPY tmp3 to tmp3
                Goldilocks3GPU::copy_pack(nrowsPack, &tmp3[args[i_args] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION]);
                i_args += 2;
                break;
            }
            case 69: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp3
                Goldilocks3GPU::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 3] * nrowsPack * FIELD_EXTENSION]);
                i_args += 4;
                break;
            }
            case 70: {
                // MULTIPLICATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: challenge
                Goldilocks3GPU::mul_pack(nrowsPack, &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack], &challenges_ops[args[i_args + 3]*FIELD_EXTENSION*nrowsPack]);
                i_args += 4;
                break;
            }
            case 71: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: challenge
                Goldilocks3GPU::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack]);
                i_args += 4;
                break;
            }
            case 72: {
                // MULTIPLICATION WITH DEST: tmp3 - SRC0: challenge - SRC1: challenge
                Goldilocks3GPU::mul_pack(nrowsPack, &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack], &challenges_ops[args[i_args + 3]*FIELD_EXTENSION*nrowsPack]);
                i_args += 4;
                break;
            }
            case 73: {
                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: challenge
                Goldilocks3GPU::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack]);
                i_args += 4;
                break;
            }
            case 74: {
                // COPY eval to tmp3
                Goldilocks3GPU::copy_pack(nrowsPack, &tmp3[args[i_args] * nrowsPack * FIELD_EXTENSION], &evals[args[i_args + 1]*FIELD_EXTENSION*nrowsPack]);
                i_args += 2;
                break;
            }
            case 75: {
                // MULTIPLICATION WITH DEST: tmp3 - SRC0: eval - SRC1: challenge
                Goldilocks3GPU::mul_pack(nrowsPack, &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &evals[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &challenges[args[i_args + 3]*FIELD_EXTENSION*nrowsPack], &challenges_ops[args[i_args + 3]*FIELD_EXTENSION*nrowsPack]);
                i_args += 4;
                break;
            }
            case 76: {
                // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: eval
                Goldilocks3GPU::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &challenges[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &evals[args[i_args + 3]*FIELD_EXTENSION*nrowsPack]);
                i_args += 4;
                break;
            }
            case 77: {
                // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: eval
                Goldilocks3GPU::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &tmp3[args[i_args + 2] * nrowsPack * FIELD_EXTENSION], &evals[args[i_args + 3]*FIELD_EXTENSION*nrowsPack]);
                i_args += 4;
                break;
            }
            case 78: {
                // OPERATION WITH DEST: tmp3 - SRC0: eval - SRC1: commit1
                Goldilocks3GPU::op_31_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &evals[args[i_args + 2]*FIELD_EXTENSION*nrowsPack], &bufferT_[(nColsStagesAcc[args[i_args + 3]] + args[i_args + 4]) * nrowsPack]);
                i_args += 5;
                break;
            }
            case 79: {
                // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: eval
                Goldilocks3GPU::op_pack(nrowsPack, args[i_args], &tmp3[args[i_args + 1] * nrowsPack * FIELD_EXTENSION], &bufferT_[(nColsStagesAcc[args[i_args + 2]] + args[i_args + 3]) * nrowsPack], &evals[args[i_args + 4]*FIELD_EXTENSION*nrowsPack]);
                i_args += 5;
                break;
            }
            default: {
                assert(false);
            }
        }
    }
    assert(i_args == nArgs);
}
#endif
