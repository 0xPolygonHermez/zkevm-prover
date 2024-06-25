
#ifndef CHELPERS_STEPS_GPU_HPP
#define CHELPERS_STEPS_GPU_HPP
#include "chelpers.hpp"
#include "chelpers_steps.hpp"
#include "steps.hpp"
#ifdef __USE_CUDA__
#include <cuda_runtime.h>
#endif

class gl64_t;
struct StepsPointers
{

    uint64_t *nColsStages_d;
    uint64_t *nColsStagesAcc_d;
    uint64_t *offsetsStages_d;

    uint32_t *ops_d;
    uint32_t *args_d;
    gl64_t *numbers_d;
    gl64_t *challenges_d;
    gl64_t *challenges_ops_d;
    gl64_t *publics_d;
    gl64_t *evals_d;

    gl64_t *bufferT_d;
    gl64_t *tmp1_d;
    gl64_t *tmp3_d;

    Goldilocks::Element *bufferT_h;

    uint32_t dimBufferT;
    uint32_t dimTmp1;
    uint32_t dimTmp3;
};
class CHelpersStepsGPU : public CHelpersSteps
{

public:
    uint64_t nCols;
    uint32_t nRowsPack;
    uint32_t nStreams;

    StepsPointers *stepPointers_d;
    StepsPointers stepPointers_h;

    void dataSetup(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams);
    void loadPolinomials_(StarkInfo &starkInfo, StepsParams &params, uint64_t row, uint64_t stage, uint64_t nrowsPack, uint64_t domainExtended);
    void storePolinomials_(StarkInfo &starkInfo, StepsParams &params, uint8_t *storePol, uint64_t row, uint64_t nrowsPack, uint64_t domainExtended);
    void calculateExpressions(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams);
    void freePointers();
#ifdef __USE_CUDA__
    inline void checkCudaError(cudaError_t err, const char *operation)
    {
        if (err != cudaSuccess)
        {
            printf("%s failed: %s\n", operation, cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }
#endif
};
#ifdef __USE_CUDA__
__global__ void _rowComputation(StepsPointers *stepPointers_d, uint32_t N, uint32_t nOps, uint32_t nArgs, uint32_t stream);
#endif
#endif