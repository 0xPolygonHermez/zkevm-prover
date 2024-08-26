#ifndef CHELPERS_STEPS_GPU_CUH
#define CHELPERS_STEPS_GPU_CUH

#if defined(__USE_CUDA__) && defined(ENABLE_EXPERIMENTAL_CODE)
#include "chelpers_steps_pack.hpp"
#include <cuda_runtime.h>
const int nStreams = 2; // streams per device
const int MAX_GPUS = 8;
class gl64_t;
class CHelpersStepsPackGPU: public CHelpersStepsPack {
public:

    int nDevices;
    int32_t nCudaThreads;

    bool domainExtended;
    uint64_t domainSize;
    uint64_t subDomainSize;
    uint32_t nextStride;
    uint32_t nOps;
    uint32_t nArgs;
    uint32_t nBufferT;
    uint32_t nTemp1;
    uint32_t nTemp3;

    vector<uint64_t> offsetsStagesGPU;

    uint32_t sharedStorageSize = 0;
    uint32_t ops_offset;
    uint32_t args_offset;
    uint32_t offsetsStages_offset;
    uint32_t nColsStages_offset;
    uint32_t nColsStagesAcc_offset;
    uint32_t challenges_offset;
    uint32_t challenges_ops_offset;
    uint32_t numbers_offset;
    uint32_t publics_offset;
    uint32_t evals_offset;


    uint32_t exclusiveStorageSize = 0;
    uint32_t constPols_offset;
    uint32_t x_offset;
    uint32_t zi_offset;
    uint32_t pols_offset;
    uint32_t xDivXSubXi_offset;
    uint32_t bufferT_offset;
    uint32_t tmp1_offset;
    uint32_t tmp3_offset;

    void calculateExpressions(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams);
    void calculateExpressionsRowsGPU(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams, uint64_t rowIni, uint64_t rowEnd);
    void prepareGPU(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams);
    void compare(StepsParams &params, uint64_t row);
    void cleanupGPU();

    void loadData(StarkInfo &starkInfo, StepsParams &params, uint64_t row, uint32_t streamIdx);
    void storeData(StarkInfo &starkInfo, StepsParams &params, uint64_t row, uint32_t streamIdx);
};

__global__ void loadPolinomialsGPU(CHelpersStepsPackGPU *cHelpersSteps, uint64_t *sharedStorage, uint64_t *exclusiveStorage, uint64_t nConstants, uint64_t stage);
__global__ void storePolinomialsGPU(CHelpersStepsPackGPU *cHelpersSteps, uint64_t *sharedStorage, uint64_t *exclusiveStorage);
__global__ void pack_kernel(CHelpersStepsPackGPU *cHelpersSteps, uint64_t *sharedStorage, uint64_t *exclusiveStorage);

#endif
#endif
