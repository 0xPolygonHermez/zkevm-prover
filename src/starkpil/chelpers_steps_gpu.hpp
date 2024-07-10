
#ifndef CHELPERS_STEPS_GPU_HPP
#define CHELPERS_STEPS_GPU_HPP
#include "definitions.hpp"
#include "chelpers.hpp"
#include "chelpers_steps.hpp"
#include "steps.hpp"
#ifdef __USE_CUDA__
#include <cuda_runtime.h>
#endif
#ifdef ENABLE_EXPERIMENTAL_CODE
class gl64_t;
struct StepsPointers
{
    uint64_t domainSize;
    uint64_t nConstants;
    uint64_t nextStride;
    uint64_t *nColsStages_d;
    uint64_t *nColsStagesAcc_d;

    uint32_t *ops_d;
    uint32_t *args_d;
    gl64_t *numbers_d;
    gl64_t *challenges_d;
    gl64_t *challenges_ops_d;
    gl64_t *publics_d;
    gl64_t *evals_d;
    gl64_t *x_n_d;
    gl64_t *x_2ns_d;
    gl64_t *zi_d;
    gl64_t *xDivXSubXi_d;

    gl64_t *bufferT_d;
    gl64_t *bufferPols_d;
    gl64_t *bufferConsts_d;
    gl64_t *tmp1_d;
    gl64_t *tmp3_d;

    uint32_t dimBufferT;
    uint32_t dimBufferPols;
    uint32_t dimBufferConsts;
    uint32_t dimTmp1;
    uint32_t dimTmp3;
};
class CHelpersStepsGPU : public CHelpersSteps
{

public:
    uint32_t nstreams;
    StepsPointers *stepPointers_d;
    StepsPointers stepPointers_h;

    void dataSetup(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams);
    void loadData(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams, uint64_t row);
    void storeData(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams, uint64_t row);
    void calculateExpressions(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams);
    void freePointers();

};
#ifdef __USE_CUDA__
__global__ void _transposeToBuffer(StepsPointers *stepPointers_d, uint64_t row, uint32_t stage, bool domainExtended, uint32_t stream);
__global__ void _transposeFromBuffer(StepsPointers *stepPointers_d, uint64_t row, uint32_t stage, bool domainExtended, uint32_t stream);
__global__ void _packComputation(StepsPointers *stepPointers_d, uint32_t N, uint32_t nOps, uint32_t nArgs, uint32_t stream);

inline void checkCudaError(cudaError_t err, const char *operation){
    if (err != cudaSuccess)
    {
        printf("%s failed: %s\n", operation, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
#endif
#endif
#endif