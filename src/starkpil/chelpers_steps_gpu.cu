#include "chelpers_steps_gpu.hpp"
#include "chelpers_steps_pack.hpp"
#ifdef __USE_CUDA__
#include "gl64_t.cuh"
#include "goldilocks_cubic_extension.cuh"
#include <inttypes.h>
#endif


#ifdef ENABLE_EXPERIMENTAL_CODE

void CHelpersStepsGPU::dataSetup(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams)
{
    bool domainExtended = parserParams.stage <= 3 ? false : true;
    uint64_t domainSize = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
    uint64_t nextStride = domainExtended ? 1 << (starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits) : 1;

    /*
        Metadata
    */
    nColsStagesAcc.resize(10 + 2);
    nColsStages.resize(10 + 2);
    offsetsStages.resize(10 + 2);

    nColsStages[0] = starkInfo.nConstants + 2;
    offsetsStages[0] = 0;

    for (uint64_t s = 1; s <= 3; ++s)
    {
        nColsStages[s] = starkInfo.mapSectionsN.section[string2section("cm" + to_string(s) + "_n")];
        if (domainExtended)
        {
            offsetsStages[s] = starkInfo.mapOffsets.section[string2section("cm" + to_string(s) + "_2ns")];
        }
        else
        {
            offsetsStages[s] = starkInfo.mapOffsets.section[string2section("cm" + to_string(s) + "_n")];
        }
    }
    if (domainExtended)
    {
        nColsStages[4] = starkInfo.mapSectionsN.section[eSection::cm4_2ns];
        offsetsStages[4] = starkInfo.mapOffsets.section[eSection::cm4_2ns];
    }
    else
    {
        nColsStages[4] = starkInfo.mapSectionsN.section[eSection::tmpExp_n];
        offsetsStages[4] = starkInfo.mapOffsets.section[eSection::tmpExp_n];
    }
    for (uint64_t o = 0; o < 2; ++o)
    {
        for (uint64_t s = 0; s < 5; ++s)
        {
            if (s == 0)
            {
                if (o == 0)
                {
                    nColsStagesAcc[0] = 0;
                }
                else
                {
                    nColsStagesAcc[5 * o] = nColsStagesAcc[5 * o - 1] + nColsStages[4];
                }
            }
            else
            {
                nColsStagesAcc[5 * o + s] = nColsStagesAcc[5 * o + (s - 1)] + nColsStages[(s - 1)];
            }
        }
    }
    nColsStagesAcc[10] = nColsStagesAcc[9] + nColsStages[4]; // Polinomials f & q
    if (parserParams.stage == 4)
    {
        offsetsStages[10] = starkInfo.mapOffsets.section[eSection::q_2ns];
        nColsStages[10] = starkInfo.qDim;
    }
    else if (parserParams.stage == 5)
    {
        offsetsStages[10] = starkInfo.mapOffsets.section[eSection::f_2ns];
        nColsStages[10] = 3;
    }
    nColsStagesAcc[11] = nColsStagesAcc[10] + nColsStages[10]; // xDivXSubXi
    nCols = nColsStagesAcc[11] + 6; // 3 for xDivXSubXi and 3 for xDivXSubWxi
    
    stepPointers_h.domainSize = domainSize;
    stepPointers_h.nConstants = starkInfo.nConstants;
    stepPointers_h.nextStride = nextStride;
    
    cudaMalloc((void **)&(stepPointers_h.nColsStages_d), nColsStages.size() * sizeof(uint64_t));
    cudaMemcpy(stepPointers_h.nColsStages_d, nColsStages.data(), nColsStages.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&(stepPointers_h.nColsStagesAcc_d), nColsStagesAcc.size() * sizeof(uint64_t));
    cudaMemcpy(stepPointers_h.nColsStagesAcc_d, nColsStagesAcc.data(), nColsStagesAcc.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);

    /*
        non-buffered data
    */
    uint8_t *ops = &parserArgs.ops[parserParams.opsOffset];
    uint32_t *ops_aux = new uint32_t[parserParams.nOps];
    for (uint64_t i = 0; i < parserParams.nOps; ++i)
        ops_aux[i] = uint32_t(ops[i]);
    cudaMalloc((void **)&(stepPointers_h.ops_d), parserParams.nOps * sizeof(uint32_t));
    cudaMemcpy(stepPointers_h.ops_d, ops_aux, parserParams.nOps * sizeof(uint32_t), cudaMemcpyHostToDevice);
    delete[] ops_aux;

    uint16_t *args = &parserArgs.args[parserParams.argsOffset];
    uint32_t *args_aux = new uint32_t[parserParams.nArgs];
    for (uint64_t i = 0; i < parserParams.nArgs; ++i)
        args_aux[i] = uint32_t(args[i]);
    cudaMalloc((void **)&(stepPointers_h.args_d), parserParams.nArgs * sizeof(uint32_t));
    cudaMemcpy(stepPointers_h.args_d, args_aux, parserParams.nArgs * sizeof(uint32_t), cudaMemcpyHostToDevice);
    delete[] args_aux;

    uint64_t *numbers = &parserArgs.numbers[parserParams.numbersOffset];
    Goldilocks::Element *numbers_aux = new Goldilocks::Element[parserParams.nNumbers * nrowsPack];
    //this expansion could be done in the GPU...
    for (uint64_t i = 0; i < parserParams.nNumbers; ++i)
    {
        for (uint64_t j = 0; j < nrowsPack; ++j)
        {
            numbers_aux[i * nrowsPack + j] = Goldilocks::fromU64(numbers[i]);
        }
    }
    cudaMalloc((void **)&(stepPointers_h.numbers_d), parserParams.nNumbers * nrowsPack * sizeof(gl64_t));
    cudaMemcpy(stepPointers_h.numbers_d, numbers_aux, parserParams.nNumbers * nrowsPack * sizeof(gl64_t), cudaMemcpyHostToDevice);
    delete[] numbers_aux;

    Goldilocks::Element *challenges_aux = new Goldilocks::Element[params.challenges.degree() * FIELD_EXTENSION * nrowsPack];
    Goldilocks::Element *challenges_ops_aux = new Goldilocks::Element[params.challenges.degree() * FIELD_EXTENSION * nrowsPack];
    //this expansion could be done in the GPU...
    for (uint64_t i = 0; i < params.challenges.degree(); ++i)
    {
        for (uint64_t j = 0; j < nrowsPack; ++j)
        {
            challenges_aux[(i * FIELD_EXTENSION) * nrowsPack + j] = params.challenges[i][0];
            challenges_aux[(i * FIELD_EXTENSION + 1) * nrowsPack + j] = params.challenges[i][1];
            challenges_aux[(i * FIELD_EXTENSION + 2) * nrowsPack + j] = params.challenges[i][2];
            challenges_ops_aux[(i * FIELD_EXTENSION) * nrowsPack + j] = params.challenges[i][0] + params.challenges[i][1];
            challenges_ops_aux[(i * FIELD_EXTENSION + 1) * nrowsPack + j] = params.challenges[i][0] + params.challenges[i][2];
            challenges_ops_aux[(i * FIELD_EXTENSION + 2) * nrowsPack + j] = params.challenges[i][1] + params.challenges[i][2];
        }
    }
    

    cudaMalloc((void **)&(stepPointers_h.challenges_d), params.challenges.degree() * FIELD_EXTENSION * nrowsPack * sizeof(gl64_t));
    cudaMemcpy(stepPointers_h.challenges_d, challenges_aux, params.challenges.degree() * FIELD_EXTENSION * nrowsPack * sizeof(gl64_t), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&(stepPointers_h.challenges_ops_d), params.challenges.degree() * FIELD_EXTENSION * nrowsPack * sizeof(gl64_t));
    cudaMemcpy(stepPointers_h.challenges_ops_d, challenges_ops_aux, params.challenges.degree() * FIELD_EXTENSION * nrowsPack * sizeof(gl64_t), cudaMemcpyHostToDevice);

    delete[] challenges_aux;
    delete[] challenges_ops_aux;

    Goldilocks::Element *publics_aux = new Goldilocks::Element[starkInfo.nPublics * nrowsPack];
    for (uint64_t i = 0; i < starkInfo.nPublics; ++i)
    {
        for (uint64_t j = 0; j < nrowsPack; ++j)
        {
            publics_aux[i * nrowsPack + j] = params.publicInputs[i];
        }
    }
    cudaMalloc((void **)&(stepPointers_h.publics_d), starkInfo.nPublics * nrowsPack * sizeof(gl64_t));
    cudaMemcpy(stepPointers_h.publics_d, publics_aux, starkInfo.nPublics * nrowsPack * sizeof(gl64_t), cudaMemcpyHostToDevice);
    delete[] publics_aux;

    Goldilocks::Element *evals_aux = new Goldilocks::Element[params.evals.degree() * FIELD_EXTENSION * nrowsPack];
    for (uint64_t i = 0; i < params.evals.degree(); ++i)
    {
        for (uint64_t j = 0; j < nrowsPack; ++j)
        {
            evals_aux[(i * FIELD_EXTENSION) * nrowsPack + j] = params.evals[i][0];
            evals_aux[(i * FIELD_EXTENSION + 1) * nrowsPack + j] = params.evals[i][1];
            evals_aux[(i * FIELD_EXTENSION + 2) * nrowsPack + j] = params.evals[i][2];
        }
    }
    cudaMalloc((void **)&(stepPointers_h.evals_d), params.evals.degree() * FIELD_EXTENSION * nrowsPack * sizeof(gl64_t));
    cudaMemcpy(stepPointers_h.evals_d, evals_aux, params.evals.degree() * FIELD_EXTENSION * nrowsPack * sizeof(gl64_t), cudaMemcpyHostToDevice);
    delete[] evals_aux;

    cudaMalloc((void**)&(stepPointers_h.x_n_d), params.x_n.degree()*sizeof(gl64_t));
    cudaMemcpy(stepPointers_h.x_n_d, params.x_n.address(), params.x_n.degree()*sizeof(gl64_t), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&(stepPointers_h.x_2ns_d), params.x_2ns.degree()*sizeof(gl64_t));
    cudaMemcpy(stepPointers_h.x_2ns_d, params.x_2ns.address(), params.x_2ns.degree()*sizeof(gl64_t), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&(stepPointers_h.zi_d), params.zi.degree()*sizeof(gl64_t));
    cudaMemcpy(stepPointers_h.zi_d, params.zi.address(), params.zi.degree()*sizeof(gl64_t), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&(stepPointers_h.xDivXSubXi_d), params.xDivXSubXi.degree()*params.xDivXSubXi.dim()*sizeof(gl64_t));
    cudaMemcpy(stepPointers_h.xDivXSubXi_d, params.xDivXSubXi.address(), params.xDivXSubXi.degree()*params.xDivXSubXi.dim()*sizeof(gl64_t), cudaMemcpyHostToDevice);

    /*
        temporal buffers
    */    

    stepPointers_h.dimBufferT = 2 * nCols * nrowsPack;
    cudaMalloc((void **)&(stepPointers_h.bufferT_d), stepPointers_h.dimBufferT * nstreams * sizeof(gl64_t));
    
    stepPointers_h.dimBufferPols = 0;
    uint64_t nStages = 3;
    for (uint64_t s = 1; s <= nStages; ++s){
        stepPointers_h.dimBufferPols += nColsStages[s];
    }
    if(parserParams.stage==5){
        stepPointers_h.dimBufferPols += nColsStages[nStages + 1];
    }
    stepPointers_h.dimBufferPols = stepPointers_h.dimBufferPols * nColsStages[10]; //for the store
    stepPointers_h.dimBufferPols = stepPointers_h.dimBufferPols * (nrowsPack+nextStride);
    stepPointers_h.dimBufferPols = 1000000;
    cudaMalloc((void **)&(stepPointers_h.bufferPols_d), stepPointers_h.dimBufferPols * nstreams * sizeof(gl64_t));

    stepPointers_h.dimBufferConsts = starkInfo.nConstants * (nrowsPack+nextStride);
    cudaMalloc((void **)&(stepPointers_h.bufferConsts_d), stepPointers_h.dimBufferConsts * nstreams * sizeof(gl64_t));
    
    stepPointers_h.dimTmp1 = parserParams.nTemp1 * nrowsPack;
    cudaMalloc((void **)&(stepPointers_h.tmp1_d), stepPointers_h.dimTmp1 * nstreams * sizeof(gl64_t));
    
    stepPointers_h.dimTmp3 = parserParams.nTemp3 * nrowsPack * FIELD_EXTENSION;
    cudaMalloc((void **)&(stepPointers_h.tmp3_d), stepPointers_h.dimTmp3 * nstreams * sizeof(gl64_t));

    /*
        copy to device
    */
    cudaMalloc((void **)&(stepPointers_d), sizeof(StepsPointers));
    cudaMemcpy(stepPointers_d, &stepPointers_h, sizeof(StepsPointers), cudaMemcpyHostToDevice);
}

void CHelpersStepsGPU::loadData(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams, uint64_t row){

    cudaError_t err;
    bool domainExtended = parserParams.stage > 3 ? true : false;
    uint32_t iStream = (row / nrowsPack) % nstreams; 
    gl64_t *bufferConsts_d = &stepPointers_h.bufferConsts_d[stepPointers_h.dimBufferConsts * iStream];
    gl64_t *bufferPols_d = &stepPointers_h.bufferPols_d[stepPointers_h.dimBufferPols * iStream];
    ConstantPolsStarks *constPols = domainExtended ? params.pConstPols2ns : params.pConstPols;


    err = cudaMemcpy(bufferConsts_d, &(((Goldilocks::Element *)constPols->address())[row * starkInfo.nConstants]), stepPointers_h.dimBufferConsts*sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);
    checkCudaError(err, "CHelpersStepsGPU::loadData: copy constants to device");
    
    uint64_t nStages=3; // do I relly need to copy all
    uint64_t offset_pols_d=0;
    for (uint64_t s = 1; s <= nStages; ++s) 
    {
        
        uint64_t offset_pols_h = offsetsStages[s] + row * nColsStages[s];
        uint64_t size_copy = nColsStages[s]*(nrowsPack+stepPointers_h.nextStride);
        err = cudaMemcpy(&(bufferPols_d[offset_pols_d]), &(params.pols[offset_pols_h]), size_copy*sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);
        checkCudaError(err, "CHelpersStepsGPU::loadData: copy pols to device");
        offset_pols_d += size_copy;
    }
    if (parserParams.stage == 5){

        uint64_t offset_pols_h = offsetsStages[nStages + 1] + row * nColsStages[nStages + 1];
        uint64_t size_copy = nColsStages[nStages + 1]*(nrowsPack+stepPointers_h.nextStride);
        err = cudaMemcpy(&(bufferPols_d[offset_pols_d]), &(params.pols[offset_pols_h]), size_copy*sizeof(Goldilocks::Element), cudaMemcpyHostToDevice);
        checkCudaError(err, "CHelpersStepsGPU::loadData: copy pols stage 5 to device");
    }


}

void CHelpersStepsGPU::storeData(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams, uint64_t row){

    cudaError_t err;
    uint32_t iStream = (row / nrowsPack) % nstreams; 
    bool domainExtended = parserParams.stage > 3 ? true : false;
    gl64_t *bufferPols_d = &stepPointers_h.bufferPols_d[stepPointers_h.dimBufferPols * iStream];
    
    if (!domainExtended){
        uint64_t size_copy = 0;
        uint64_t nStages=3; // do I relly need to copy all
        uint64_t offset_pols_d=0;
        for (uint64_t s = 2; s <= nStages + 1; ++s) //optimize copies that can be avoided...
        {
            
            uint64_t offset_pols_h = offsetsStages[s] + row * nColsStages[s];
            uint64_t size_copy = nColsStages[s]*nrowsPack;

            err = cudaMemcpy(&(params.pols[offset_pols_h]), &(bufferPols_d[offset_pols_d]), size_copy*sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost);
            checkCudaError(err, "CHelpersStepsGPU::storeData: copy device to host");
            offset_pols_d += size_copy;
        }
    }else{
        uint64_t size_copy = nColsStages[10]*nrowsPack;
        gl64_t *bufferPols_ = &(stepPointers_h.bufferPols_d[(iStream+1) * stepPointers_h.dimBufferPols-size_copy]); //data available at the end
        err = cudaMemcpy(&(params.pols[offsetsStages[10] + row * nColsStages[10]]), bufferPols_, size_copy*sizeof(Goldilocks::Element), cudaMemcpyDeviceToHost);
        checkCudaError(err, "CHelpersStepsGPU::storeData: copy pols to device domain extended");
    }

}

void CHelpersStepsGPU::calculateExpressions(StarkInfo &starkInfo, StepsParams &params, ParserArgs &parserArgs, ParserParams &parserParams)
{

    // condition, nrowsPack should cover the opening and divide domainSize!!
    bool domainExtended = parserParams.stage > 3 ? true : false;
    uint64_t domainSize = domainExtended ? 1 << starkInfo.starkStruct.nBitsExt : 1 << starkInfo.starkStruct.nBits;
    uint64_t nextStride = domainExtended ? 1 << (starkInfo.starkStruct.nBitsExt - starkInfo.starkStruct.nBits) : 1;
    nrowsPack = 32;
    nstreams = 32;

    // For the last chunk of rows will be solved uwing the chelpers_pack
    assert(nrowsPack >= nextStride);
    CHelpersStepsPack chelpersPack;
    chelpersPack.calculateExpressionsRows(starkInfo, params, parserArgs, parserParams, domainSize-nrowsPack, domainSize);

    //Rest of packs will be copmuted in the GPU...
    dataSetup(starkInfo, params, parserArgs, parserParams);
    cudaStream_t *streams = new cudaStream_t[nstreams];
    for (int i = 0; i < nstreams; i++)
    {
        cudaStreamCreate(&streams[i]);
    }

    for (uint64_t i = 0; i < domainSize-nrowsPack; i += nrowsPack)
    {
        uint32_t iStream = (i / nrowsPack) % nstreams; 
        loadData(starkInfo, params, parserArgs, parserParams, i);
        _transposeToBuffer<<<1, nrowsPack>>>(stepPointers_d, i, parserParams.stage, domainExtended, iStream);
        _packComputation<<<1, nrowsPack>>>(stepPointers_d, domainSize, parserParams.nOps, parserParams.nArgs, iStream);
        _transposeFromBuffer<<<1, nrowsPack>>>(stepPointers_d, i, parserParams.stage, domainExtended, iStream);
        storeData(starkInfo, params, parserArgs, parserParams, i);
    
    }

    //
    // Free pointers and destroy streams
    //
    freePointers();
    for (int i = 0; i < nstreams; i++)
    {
        cudaStreamDestroy(streams[i]);
    }
    delete[] streams;
}

void CHelpersStepsGPU::freePointers()
{
    cudaFree(stepPointers_h.nColsStages_d);
    cudaFree(stepPointers_h.nColsStagesAcc_d);
    cudaFree(stepPointers_h.ops_d);
    cudaFree(stepPointers_h.args_d);
    cudaFree(stepPointers_h.numbers_d);
    cudaFree(stepPointers_h.challenges_d);
    cudaFree(stepPointers_h.challenges_ops_d);
    cudaFree(stepPointers_h.publics_d);
    cudaFree(stepPointers_h.evals_d);
    cudaFree(stepPointers_h.x_n_d);
    cudaFree(stepPointers_h.x_2ns_d);
    cudaFree(stepPointers_h.zi_d);
    cudaFree(stepPointers_h.xDivXSubXi_d);
    cudaFree(stepPointers_h.bufferT_d);
    cudaFree(stepPointers_h.bufferPols_d);
    cudaFree(stepPointers_h.bufferConsts_d);
    cudaFree(stepPointers_h.tmp1_d);
    cudaFree(stepPointers_h.tmp3_d);
    cudaFree(stepPointers_d);
}

__global__ void _packComputation(StepsPointers *stepPointers_d, uint32_t N, uint32_t nOps, uint32_t nArgs, uint32_t stream)
{

    uint64_t i_args = 0;
    gl64_t *bufferT_ = &(stepPointers_d->bufferT_d[stream * stepPointers_d->dimBufferT]);
    gl64_t *tmp1 = &(stepPointers_d->tmp1_d[stream * stepPointers_d->dimTmp1]);
    gl64_t *tmp3 = &(stepPointers_d->tmp3_d[stream * stepPointers_d->dimTmp3]);
    uint32_t *ops_ = stepPointers_d->ops_d;
    uint32_t *args_ = stepPointers_d->args_d;
    
    for (uint64_t kk = 0; kk < nOps; ++kk)
    {
        switch (ops_[kk])
        {
        case 0:
        {
            // COPY commit1 to commit1
            gl64_t::copy_gpu(&bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args]] + args_[i_args + 1]) * blockDim.x], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 2]] + args_[i_args + 3]) * blockDim.x]);
            i_args += 4;
            break;
        }
        case 1:
        {
            // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: commit1
            gl64_t::op_gpu(args_[i_args], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 1]] + args_[i_args + 2]) * blockDim.x], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 3]] + args_[i_args + 4]) * blockDim.x], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 5]] + args_[i_args + 6]) * blockDim.x]);
            i_args += 7;
            break;
        }
        case 2:
        {
            // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: tmp1
            gl64_t::op_gpu(args_[i_args], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 1]] + args_[i_args + 2]) * blockDim.x], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 3]] + args_[i_args + 4]) * blockDim.x], &tmp1[args_[i_args + 5] * blockDim.x]);
            i_args += 6;
            break;
        }
        case 3:
        {
            // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: public
            gl64_t::op_gpu(args_[i_args], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 1]] + args_[i_args + 2]) * blockDim.x], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 3]] + args_[i_args + 4]) * blockDim.x], &stepPointers_d->publics_d[args_[i_args + 5] * blockDim.x]);
            i_args += 6;
            break;
        }
        case 4:
        {
            // OPERATION WITH DEST: commit1 - SRC0: commit1 - SRC1: number
            gl64_t::op_gpu(args_[i_args], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 1]] + args_[i_args + 2]) * blockDim.x], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 3]] + args_[i_args + 4]) * blockDim.x], &stepPointers_d->numbers_d[args_[i_args + 5] * blockDim.x]);
            i_args += 6;
            break;
        }
        case 5:
        {
            // COPY tmp1 to commit1
            gl64_t::copy_gpu(&bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args]] + args_[i_args + 1]) * blockDim.x], &tmp1[args_[i_args + 2] * blockDim.x]);
            i_args += 3;
            break;
        }
        case 6:
        {
            // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: tmp1
            gl64_t::op_gpu(args_[i_args], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 1]] + args_[i_args + 2]) * blockDim.x], &tmp1[args_[i_args + 3] * blockDim.x], &tmp1[args_[i_args + 4] * blockDim.x]);
            i_args += 5;
            break;
        }
        case 7:
        {
            // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: public
            gl64_t::op_gpu(args_[i_args], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 1]] + args_[i_args + 2]) * blockDim.x], &tmp1[args_[i_args + 3] * blockDim.x], &stepPointers_d->publics_d[args_[i_args + 4] * blockDim.x]);
            i_args += 5;
            break;
        }
        case 8:
        {
            // OPERATION WITH DEST: commit1 - SRC0: tmp1 - SRC1: number
            gl64_t::op_gpu(args_[i_args], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 1]] + args_[i_args + 2]) * blockDim.x], &tmp1[args_[i_args + 3] * blockDim.x], &stepPointers_d->numbers_d[args_[i_args + 4] * blockDim.x]);
            i_args += 5;
            break;
        }
        case 9:
        {
            // COPY public to commit1
            gl64_t::copy_gpu(&bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args]] + args_[i_args + 1]) * blockDim.x], &stepPointers_d->publics_d[args_[i_args + 2] * blockDim.x]);
            i_args += 3;
            break;
        }
        case 10:
        {
            // OPERATION WITH DEST: commit1 - SRC0: public - SRC1: public
            gl64_t::op_gpu(args_[i_args], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 1]] + args_[i_args + 2]) * blockDim.x], &stepPointers_d->publics_d[args_[i_args + 3] * blockDim.x], &stepPointers_d->publics_d[args_[i_args + 4] * blockDim.x]);
            i_args += 5;
            break;
        }
        case 11:
        {
            // OPERATION WITH DEST: commit1 - SRC0: public - SRC1: number
            gl64_t::op_gpu(args_[i_args], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 1]] + args_[i_args + 2]) * blockDim.x], &stepPointers_d->publics_d[args_[i_args + 3] * blockDim.x], &stepPointers_d->numbers_d[args_[i_args + 4] * blockDim.x]);
            i_args += 5;
            break;
        }
        case 12:
        {
            // COPY number to commit1
            gl64_t::copy_gpu(&bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args]] + args_[i_args + 1]) * blockDim.x], &stepPointers_d->numbers_d[args_[i_args + 2] * blockDim.x]);
            i_args += 3;
            break;
        }
        case 13:
        {
            // OPERATION WITH DEST: commit1 - SRC0: number - SRC1: number
            gl64_t::op_gpu(args_[i_args], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 1]] + args_[i_args + 2]) * blockDim.x], &stepPointers_d->numbers_d[args_[i_args + 3] * blockDim.x], &stepPointers_d->numbers_d[args_[i_args + 4] * blockDim.x]);
            i_args += 5;
            break;
        }
        case 14:
        {
            // COPY commit1 to tmp1
            gl64_t::copy_gpu(&tmp1[args_[i_args] * blockDim.x], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 1]] + args_[i_args + 2]) * blockDim.x]);
            i_args += 3;
            break;
        }
        case 15:
        {
            // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: commit1
            gl64_t::op_gpu(args_[i_args], &tmp1[args_[i_args + 1] * blockDim.x], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 2]] + args_[i_args + 3]) * blockDim.x], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 4]] + args_[i_args + 5]) * blockDim.x]);
            i_args += 6;
            break;
        }
        case 16:
        {
            // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: tmp1
            gl64_t::op_gpu(args_[i_args], &tmp1[args_[i_args + 1] * blockDim.x], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 2]] + args_[i_args + 3]) * blockDim.x], &tmp1[args_[i_args + 4] * blockDim.x]);
            i_args += 5;
            break;
        }
        case 17:
        {
            // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: public
            gl64_t::op_gpu(args_[i_args], &tmp1[args_[i_args + 1] * blockDim.x], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 2]] + args_[i_args + 3]) * blockDim.x], &stepPointers_d->publics_d[args_[i_args + 4] * blockDim.x]);
            i_args += 5;
            break;
        }
        case 18:
        {
            // OPERATION WITH DEST: tmp1 - SRC0: commit1 - SRC1: number
            gl64_t::op_gpu(args_[i_args], &tmp1[args_[i_args + 1] * blockDim.x], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 2]] + args_[i_args + 3]) * blockDim.x], &stepPointers_d->numbers_d[args_[i_args + 4] * blockDim.x]);
            i_args += 5;
            break;
        }
        case 19:
        {
            // COPY tmp1 to tmp1
            gl64_t::copy_gpu(&tmp1[args_[i_args] * blockDim.x], &tmp1[args_[i_args + 1] * blockDim.x]);
            i_args += 2;
            break;
        }
        case 20:
        {
            // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: tmp1
            gl64_t::op_gpu(args_[i_args], &tmp1[args_[i_args + 1] * blockDim.x], &tmp1[args_[i_args + 2] * blockDim.x], &tmp1[args_[i_args + 3] * blockDim.x]);
            i_args += 4;
            break;
        }
        case 21:
        {
            // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: public
            gl64_t::op_gpu(args_[i_args], &tmp1[args_[i_args + 1] * blockDim.x], &tmp1[args_[i_args + 2] * blockDim.x], &stepPointers_d->publics_d[args_[i_args + 3] * blockDim.x]);
            i_args += 4;
            break;
        }
        case 22:
        {
            // OPERATION WITH DEST: tmp1 - SRC0: tmp1 - SRC1: number
            gl64_t::op_gpu(args_[i_args], &tmp1[args_[i_args + 1] * blockDim.x], &tmp1[args_[i_args + 2] * blockDim.x], &stepPointers_d->numbers_d[args_[i_args + 3] * blockDim.x]);
            i_args += 4;
            break;
        }
        case 23:
        {
            // COPY public to tmp1
            gl64_t::copy_gpu(&tmp1[args_[i_args] * blockDim.x], &stepPointers_d->publics_d[args_[i_args + 1] * blockDim.x]);
            i_args += 2;
            break;
        }
        case 24:
        {
            // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: public
            gl64_t::op_gpu(args_[i_args], &tmp1[args_[i_args + 1] * blockDim.x], &stepPointers_d->publics_d[args_[i_args + 2] * blockDim.x], &stepPointers_d->publics_d[args_[i_args + 3] * blockDim.x]);
            i_args += 4;
            break;
        }
        case 25:
        {
            // OPERATION WITH DEST: tmp1 - SRC0: public - SRC1: number
            gl64_t::op_gpu(args_[i_args], &tmp1[args_[i_args + 1] * blockDim.x], &stepPointers_d->publics_d[args_[i_args + 2] * blockDim.x], &stepPointers_d->numbers_d[args_[i_args + 3] * blockDim.x]);
            i_args += 4;
            break;
        }
        case 26:
        {
            // COPY number to tmp1
            gl64_t::copy_gpu(&tmp1[args_[i_args] * blockDim.x], &stepPointers_d->numbers_d[args_[i_args + 1] * blockDim.x]);
            i_args += 2;
            break;
        }
        case 27:
        {
            // OPERATION WITH DEST: tmp1 - SRC0: number - SRC1: number
            gl64_t::op_gpu(args_[i_args], &tmp1[args_[i_args + 1] * blockDim.x], &stepPointers_d->numbers_d[args_[i_args + 2] * blockDim.x], &stepPointers_d->numbers_d[args_[i_args + 3] * blockDim.x]);
            i_args += 4;
            break;
        }
        case 28:
        {
            // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: commit1
            Goldilocks3GPU::op_31_gpu(args_[i_args], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 1]] + args_[i_args + 2]) * blockDim.x], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 3]] + args_[i_args + 4]) * blockDim.x], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 5]] + args_[i_args + 6]) * blockDim.x]);
            i_args += 7;
            break;
        }
        case 29:
        {
            // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: tmp1
            Goldilocks3GPU::op_31_gpu(args_[i_args], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 1]] + args_[i_args + 2]) * blockDim.x], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 3]] + args_[i_args + 4]) * blockDim.x], &tmp1[args_[i_args + 5] * blockDim.x]);
            i_args += 6;
            break;
        }
        case 30:
        {
            // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: public
            Goldilocks3GPU::op_31_gpu(args_[i_args], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 1]] + args_[i_args + 2]) * blockDim.x], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 3]] + args_[i_args + 4]) * blockDim.x], &stepPointers_d->publics_d[args_[i_args + 5] * blockDim.x]);
            i_args += 6;
            break;
        }
        case 31:
        {
            // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: number
            Goldilocks3GPU::op_31_gpu(args_[i_args], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 1]] + args_[i_args + 2]) * blockDim.x], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 3]] + args_[i_args + 4]) * blockDim.x], &stepPointers_d->numbers_d[args_[i_args + 5] * blockDim.x]);
            i_args += 6;
            break;
        }
        case 32:
        {
            // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: commit1
            Goldilocks3GPU::op_31_gpu(args_[i_args], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 1]] + args_[i_args + 2]) * blockDim.x], &tmp3[args_[i_args + 3] * blockDim.x * FIELD_EXTENSION], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 4]] + args_[i_args + 5]) * blockDim.x]);
            i_args += 6;
            break;
        }
        case 33:
        {
            // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: tmp1
            Goldilocks3GPU::op_31_gpu(args_[i_args], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 1]] + args_[i_args + 2]) * blockDim.x], &tmp3[args_[i_args + 3] * blockDim.x * FIELD_EXTENSION], &tmp1[args_[i_args + 4] * blockDim.x]);
            i_args += 5;
            break;
        }
        case 34:
        {
            // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: public
            Goldilocks3GPU::op_31_gpu(args_[i_args], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 1]] + args_[i_args + 2]) * blockDim.x], &tmp3[args_[i_args + 3] * blockDim.x * FIELD_EXTENSION], &stepPointers_d->publics_d[args_[i_args + 4] * blockDim.x]);
            i_args += 5;
            break;
        }
        case 35:
        {
            // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: number
            Goldilocks3GPU::op_31_gpu(args_[i_args], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 1]] + args_[i_args + 2]) * blockDim.x], &tmp3[args_[i_args + 3] * blockDim.x * FIELD_EXTENSION], &stepPointers_d->numbers_d[args_[i_args + 4] * blockDim.x]);
            i_args += 5;
            break;
        }
        case 36:
        {
            // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: commit1
            Goldilocks3GPU::op_31_gpu(args_[i_args], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 1]] + args_[i_args + 2]) * blockDim.x], &stepPointers_d->challenges_d[args_[i_args + 3] * FIELD_EXTENSION * blockDim.x], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 4]] + args_[i_args + 5]) * blockDim.x]);
            i_args += 6;
            break;
        }
        case 37:
        {
            // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: tmp1
            Goldilocks3GPU::op_31_gpu(args_[i_args], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 1]] + args_[i_args + 2]) * blockDim.x], &stepPointers_d->challenges_d[args_[i_args + 3] * FIELD_EXTENSION * blockDim.x], &tmp1[args_[i_args + 4] * blockDim.x]);
            i_args += 5;
            break;
        }
        case 38:
        {
            // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: public
            Goldilocks3GPU::op_31_gpu(args_[i_args], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 1]] + args_[i_args + 2]) * blockDim.x], &stepPointers_d->challenges_d[args_[i_args + 3] * FIELD_EXTENSION * blockDim.x], &stepPointers_d->publics_d[args_[i_args + 4] * blockDim.x]);
            i_args += 5;
            break;
        }
        case 39:
        {
            // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: number
            Goldilocks3GPU::op_31_gpu(args_[i_args], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 1]] + args_[i_args + 2]) * blockDim.x], &stepPointers_d->challenges_d[args_[i_args + 3] * FIELD_EXTENSION * blockDim.x], &stepPointers_d->numbers_d[args_[i_args + 4] * blockDim.x]);
            i_args += 5;
            break;
        }
        case 40:
        {
            // COPY commit3 to commit3
            Goldilocks3GPU::copy_gpu(&bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args]] + args_[i_args + 1]) * blockDim.x], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 2]] + args_[i_args + 3]) * blockDim.x]);
            i_args += 4;
            break;
        }
        case 41:
        {
            // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: commit3
            Goldilocks3GPU::op_gpu(args_[i_args], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 1]] + args_[i_args + 2]) * blockDim.x], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 3]] + args_[i_args + 4]) * blockDim.x], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 5]] + args_[i_args + 6]) * blockDim.x]);
            i_args += 7;
            break;
        }
        case 42:
        {
            // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: tmp3
            Goldilocks3GPU::op_gpu(args_[i_args], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 1]] + args_[i_args + 2]) * blockDim.x], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 3]] + args_[i_args + 4]) * blockDim.x], &tmp3[args_[i_args + 5] * blockDim.x * FIELD_EXTENSION]);
            i_args += 6;
            break;
        }
        case 43:
        {
            // MULTIPLICATION WITH DEST: commit3 - SRC0: commit3 - SRC1: challenge
            Goldilocks3GPU::mul_gpu(&bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 1]] + args_[i_args + 2]) * blockDim.x], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 3]] + args_[i_args + 4]) * blockDim.x], &stepPointers_d->challenges_d[args_[i_args + 5] * FIELD_EXTENSION * blockDim.x], &stepPointers_d->challenges_ops_d[args_[i_args + 5] * FIELD_EXTENSION * blockDim.x]);
            i_args += 6;
            break;
        }
        case 44:
        {
            // OPERATION WITH DEST: commit3 - SRC0: commit3 - SRC1: challenge
            Goldilocks3GPU::op_gpu(args_[i_args], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 1]] + args_[i_args + 2]) * blockDim.x], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 3]] + args_[i_args + 4]) * blockDim.x], &stepPointers_d->challenges_d[args_[i_args + 5] * FIELD_EXTENSION * blockDim.x]);
            i_args += 6;
            break;
        }
        case 45:
        {
            // COPY tmp3 to commit3
            Goldilocks3GPU::copy_gpu(&bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args]] + args_[i_args + 1]) * blockDim.x], &tmp3[args_[i_args + 2] * blockDim.x * FIELD_EXTENSION]);
            i_args += 3;
            break;
        }
        case 46:
        {
            // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: tmp3
            Goldilocks3GPU::op_gpu(args_[i_args], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 1]] + args_[i_args + 2]) * blockDim.x], &tmp3[args_[i_args + 3] * blockDim.x * FIELD_EXTENSION], &tmp3[args_[i_args + 4] * blockDim.x * FIELD_EXTENSION]);
            i_args += 5;
            break;
        }
        case 47:
        {
            // MULTIPLICATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: challenge
            Goldilocks3GPU::mul_gpu(&bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 1]] + args_[i_args + 2]) * blockDim.x], &tmp3[args_[i_args + 3] * blockDim.x * FIELD_EXTENSION], &stepPointers_d->challenges_d[args_[i_args + 4] * FIELD_EXTENSION * blockDim.x], &stepPointers_d->challenges_ops_d[args_[i_args + 4] * FIELD_EXTENSION * blockDim.x]);
            i_args += 5;
            break;
        }
        case 48:
        {
            // OPERATION WITH DEST: commit3 - SRC0: tmp3 - SRC1: challenge
            Goldilocks3GPU::op_gpu(args_[i_args], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 1]] + args_[i_args + 2]) * blockDim.x], &tmp3[args_[i_args + 3] * blockDim.x * FIELD_EXTENSION], &stepPointers_d->challenges_d[args_[i_args + 4] * FIELD_EXTENSION * blockDim.x]);
            i_args += 5;
            break;
        }
        case 49:
        {
            // MULTIPLICATION WITH DEST: commit3 - SRC0: challenge - SRC1: challenge
            Goldilocks3GPU::mul_gpu(&bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 1]] + args_[i_args + 2]) * blockDim.x], &stepPointers_d->challenges_d[args_[i_args + 3] * FIELD_EXTENSION * blockDim.x], &stepPointers_d->challenges_d[args_[i_args + 4] * FIELD_EXTENSION * blockDim.x], &stepPointers_d->challenges_ops_d[args_[i_args + 4] * FIELD_EXTENSION * blockDim.x]);
            i_args += 5;
            break;
        }
        case 50:
        {
            // OPERATION WITH DEST: commit3 - SRC0: challenge - SRC1: challenge
            Goldilocks3GPU::op_gpu(args_[i_args], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 1]] + args_[i_args + 2]) * blockDim.x], &stepPointers_d->challenges_d[args_[i_args + 3] * FIELD_EXTENSION * blockDim.x], &stepPointers_d->challenges_d[args_[i_args + 4] * FIELD_EXTENSION * blockDim.x]);
            i_args += 5;
            break;
        }
        case 51:
        {
            // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit1
            Goldilocks3GPU::op_31_gpu(args_[i_args], &tmp3[args_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 2]] + args_[i_args + 3]) * blockDim.x], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 4]] + args_[i_args + 5]) * blockDim.x]);
            i_args += 6;
            break;
        }
        case 52:
        {
            // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp1
            Goldilocks3GPU::op_31_gpu(args_[i_args], &tmp3[args_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 2]] + args_[i_args + 3]) * blockDim.x], &tmp1[args_[i_args + 4] * blockDim.x]);
            i_args += 5;
            break;
        }
        case 53:
        {
            // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: public
            Goldilocks3GPU::op_31_gpu(args_[i_args], &tmp3[args_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 2]] + args_[i_args + 3]) * blockDim.x], &stepPointers_d->publics_d[args_[i_args + 4] * blockDim.x]);
            i_args += 5;
            break;
        }
        case 54:
        {
            // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: number
            Goldilocks3GPU::op_31_gpu(args_[i_args], &tmp3[args_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 2]] + args_[i_args + 3]) * blockDim.x], &stepPointers_d->numbers_d[args_[i_args + 4] * blockDim.x]);
            i_args += 5;
            break;
        }
        case 55:
        {
            // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: commit1
            Goldilocks3GPU::op_31_gpu(args_[i_args], &tmp3[args_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &tmp3[args_[i_args + 2] * blockDim.x * FIELD_EXTENSION], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 3]] + args_[i_args + 4]) * blockDim.x]);
            i_args += 5;
            break;
        }
        case 56:
        {
            // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp1
            Goldilocks3GPU::op_31_gpu(args_[i_args], &tmp3[args_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &tmp3[args_[i_args + 2] * blockDim.x * FIELD_EXTENSION], &tmp1[args_[i_args + 3] * blockDim.x]);
            i_args += 4;
            break;
        }
        case 57:
        {
            // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: public
            Goldilocks3GPU::op_31_gpu(args_[i_args], &tmp3[args_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &tmp3[args_[i_args + 2] * blockDim.x * FIELD_EXTENSION], &stepPointers_d->publics_d[args_[i_args + 3] * blockDim.x]);
            i_args += 4;
            break;
        }
        case 58:
        {
            // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: number
            Goldilocks3GPU::op_31_gpu(args_[i_args], &tmp3[args_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &tmp3[args_[i_args + 2] * blockDim.x * FIELD_EXTENSION], &stepPointers_d->numbers_d[args_[i_args + 3] * blockDim.x]);
            i_args += 4;
            break;
        }
        case 59:
        {
            // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: commit1
            Goldilocks3GPU::op_31_gpu(args_[i_args], &tmp3[args_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &stepPointers_d->challenges_d[args_[i_args + 2] * FIELD_EXTENSION * blockDim.x], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 3]] + args_[i_args + 4]) * blockDim.x]);
            i_args += 5;
            break;
        }
        case 60:
        {
            // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: tmp1
            Goldilocks3GPU::op_31_gpu(args_[i_args], &tmp3[args_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &stepPointers_d->challenges_d[args_[i_args + 2] * FIELD_EXTENSION * blockDim.x], &tmp1[args_[i_args + 3] * blockDim.x]);
            i_args += 4;
            break;
        }
        case 61:
        {
            // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: public
            Goldilocks3GPU::op_31_gpu(args_[i_args], &tmp3[args_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &stepPointers_d->challenges_d[args_[i_args + 2] * FIELD_EXTENSION * blockDim.x], &stepPointers_d->publics_d[args_[i_args + 3] * blockDim.x]);
            i_args += 4;
            break;
        }
        case 62:
        {
            // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: number
            Goldilocks3GPU::op_31_gpu(args_[i_args], &tmp3[args_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &stepPointers_d->challenges_d[args_[i_args + 2] * FIELD_EXTENSION * blockDim.x], &stepPointers_d->numbers_d[args_[i_args + 3] * blockDim.x]);
            i_args += 4;
            break;
        }
        case 63:
        {
            // COPY commit3 to tmp3
            Goldilocks3GPU::copy_gpu(&tmp3[args_[i_args] * blockDim.x * FIELD_EXTENSION], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 1]] + args_[i_args + 2]) * blockDim.x]);
            i_args += 3;
            break;
        }
        case 64:
        {
            // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: commit3
            Goldilocks3GPU::op_gpu(args_[i_args], &tmp3[args_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 2]] + args_[i_args + 3]) * blockDim.x], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 4]] + args_[i_args + 5]) * blockDim.x]);
            i_args += 6;
            break;
        }
        case 65:
        {
            // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: tmp3
            Goldilocks3GPU::op_gpu(args_[i_args], &tmp3[args_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 2]] + args_[i_args + 3]) * blockDim.x], &tmp3[args_[i_args + 4] * blockDim.x * FIELD_EXTENSION]);
            i_args += 5;
            break;
        }
        case 66:
        {
            // MULTIPLICATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: challenge
            Goldilocks3GPU::mul_gpu(&tmp3[args_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 2]] + args_[i_args + 3]) * blockDim.x], &stepPointers_d->challenges_d[args_[i_args + 4] * FIELD_EXTENSION * blockDim.x], &stepPointers_d->challenges_ops_d[args_[i_args + 4] * FIELD_EXTENSION * blockDim.x]);
            i_args += 5;
            break;
        }
        case 67:
        {
            // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: challenge
            Goldilocks3GPU::op_gpu(args_[i_args], &tmp3[args_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 2]] + args_[i_args + 3]) * blockDim.x], &stepPointers_d->challenges_d[args_[i_args + 4] * FIELD_EXTENSION * blockDim.x]);
            i_args += 5;
            break;
        }
        case 68:
        {
            // COPY tmp3 to tmp3
            Goldilocks3GPU::copy_gpu(&tmp3[args_[i_args] * blockDim.x * FIELD_EXTENSION], &tmp3[args_[i_args + 1] * blockDim.x * FIELD_EXTENSION]);
            i_args += 2;
            break;
        }
        case 69:
        {
            // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: tmp3
            Goldilocks3GPU::op_gpu(args_[i_args], &tmp3[args_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &tmp3[args_[i_args + 2] * blockDim.x * FIELD_EXTENSION], &tmp3[args_[i_args + 3] * blockDim.x * FIELD_EXTENSION]);
            i_args += 4;
            break;
        }
        case 70:
        {
            // MULTIPLICATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: challenge
            Goldilocks3GPU::mul_gpu(&tmp3[args_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &tmp3[args_[i_args + 2] * blockDim.x * FIELD_EXTENSION], &stepPointers_d->challenges_d[args_[i_args + 3] * FIELD_EXTENSION * blockDim.x], &stepPointers_d->challenges_ops_d[args_[i_args + 3] * FIELD_EXTENSION * blockDim.x]);
            i_args += 4;
            break;
        }
        case 71:
        {
            // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: challenge
            Goldilocks3GPU::op_gpu(args_[i_args], &tmp3[args_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &tmp3[args_[i_args + 2] * blockDim.x * FIELD_EXTENSION], &stepPointers_d->challenges_d[args_[i_args + 3] * FIELD_EXTENSION * blockDim.x]);
            i_args += 4;
            break;
        }
        case 72:
        {
            // MULTIPLICATION WITH DEST: tmp3 - SRC0: challenge - SRC1: challenge
            Goldilocks3GPU::mul_gpu(&tmp3[args_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &stepPointers_d->challenges_d[args_[i_args + 2] * FIELD_EXTENSION * blockDim.x], &stepPointers_d->challenges_d[args_[i_args + 3] * FIELD_EXTENSION * blockDim.x], &stepPointers_d->challenges_ops_d[args_[i_args + 3] * FIELD_EXTENSION * blockDim.x]);
            i_args += 4;
            break;
        }
        case 73:
        {
            // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: challenge
            Goldilocks3GPU::op_gpu(args_[i_args], &tmp3[args_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &stepPointers_d->challenges_d[args_[i_args + 2] * FIELD_EXTENSION * blockDim.x], &stepPointers_d->challenges_d[args_[i_args + 3] * FIELD_EXTENSION * blockDim.x]);
            i_args += 4;
            break;
        }
        case 74:
        {
            // COPY eval to tmp3
            Goldilocks3GPU::copy_gpu(&tmp3[args_[i_args] * blockDim.x * FIELD_EXTENSION], &stepPointers_d->evals_d[args_[i_args + 1] * FIELD_EXTENSION * blockDim.x]);
            i_args += 2;
            break;
        }
        case 75:
        {
            // MULTIPLICATION WITH DEST: tmp3 - SRC0: eval - SRC1: challenge
            Goldilocks3GPU::mul_gpu(&tmp3[args_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &stepPointers_d->evals_d[args_[i_args + 2] * FIELD_EXTENSION * blockDim.x], &stepPointers_d->challenges_d[args_[i_args + 3] * FIELD_EXTENSION * blockDim.x], &stepPointers_d->challenges_ops_d[args_[i_args + 3] * FIELD_EXTENSION * blockDim.x]);
            i_args += 4;
            break;
        }
        case 76:
        {
            // OPERATION WITH DEST: tmp3 - SRC0: challenge - SRC1: eval
            Goldilocks3GPU::op_gpu(args_[i_args], &tmp3[args_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &stepPointers_d->challenges_d[args_[i_args + 2] * FIELD_EXTENSION * blockDim.x], &stepPointers_d->evals_d[args_[i_args + 3] * FIELD_EXTENSION * blockDim.x]);
            i_args += 4;
            break;
        }
        case 77:
        {
            // OPERATION WITH DEST: tmp3 - SRC0: tmp3 - SRC1: eval
            Goldilocks3GPU::op_gpu(args_[i_args], &tmp3[args_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &tmp3[args_[i_args + 2] * blockDim.x * FIELD_EXTENSION], &stepPointers_d->evals_d[args_[i_args + 3] * FIELD_EXTENSION * blockDim.x]);
            i_args += 4;
            break;
        }
        case 78:
        {
            // OPERATION WITH DEST: tmp3 - SRC0: eval - SRC1: commit1
            Goldilocks3GPU::op_31_gpu(args_[i_args], &tmp3[args_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &stepPointers_d->evals_d[args_[i_args + 2] * FIELD_EXTENSION * blockDim.x], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 3]] + args_[i_args + 4]) * blockDim.x]);
            i_args += 5;
            break;
        }
        case 79:
        {
            // OPERATION WITH DEST: tmp3 - SRC0: commit3 - SRC1: eval
            Goldilocks3GPU::op_gpu(args_[i_args], &tmp3[args_[i_args + 1] * blockDim.x * FIELD_EXTENSION], &bufferT_[(stepPointers_d->nColsStagesAcc_d[args_[i_args + 2]] + args_[i_args + 3]) * blockDim.x], &stepPointers_d->evals_d[args_[i_args + 4] * FIELD_EXTENSION * blockDim.x]);
            i_args += 5;
            break;
        }
        default:
        {
            return;
        }
        }
    }
}

__global__ void _transposeToBuffer(StepsPointers *stepPointers_d, uint64_t row, uint32_t stage, bool domainExtended, uint32_t istream){

    gl64_t *bufferT_ = &(stepPointers_d->bufferT_d[istream * stepPointers_d->dimBufferT]);
    gl64_t *bufferConsts_ = &(stepPointers_d->bufferConsts_d[istream * stepPointers_d->dimBufferConsts]);
    gl64_t *bufferPols_ = &(stepPointers_d->bufferPols_d[istream * stepPointers_d->dimBufferPols]);
    gl64_t * x = domainExtended ? stepPointers_d->x_2ns_d : stepPointers_d->x_n_d;

    uint64_t nextStrides[2] = {0, stepPointers_d->nextStride};
    for (uint64_t o = 0; o < 2; ++o)
    {
        for (uint64_t k = 0; k < stepPointers_d->nConstants; ++k)
        {
        
            bufferT_[(stepPointers_d->nColsStagesAcc_d[5 * o] + k) * blockDim.x + threadIdx.x] = bufferConsts_[(threadIdx.x+nextStrides[o])* stepPointers_d->nConstants + k]; 
        }
    }

    bufferT_[stepPointers_d->nConstants * blockDim.x + threadIdx.x] = x[row + threadIdx.x];
    bufferT_[(stepPointers_d->nConstants + 1) * blockDim.x + threadIdx.x] = stepPointers_d->zi_d[row + threadIdx.x];

    uint32_t offset_pols = 0;
    uint64_t nStages = 3;
    for (uint64_t s = 1; s <= nStages; ++s) 
    {
        for (uint64_t o = 0; o < 2; ++o) 
        {
            for (uint64_t k = 0; k < stepPointers_d->nColsStages_d[s]; ++k)
            {    
                uint64_t l = threadIdx.x + nextStrides[o];
                bufferT_[(stepPointers_d->nColsStagesAcc_d[5 * o + s] + k) * blockDim.x + threadIdx.x] = bufferPols_[offset_pols + l * stepPointers_d->nColsStages_d[s] + k];
            }
        }
        offset_pols += stepPointers_d->nColsStages_d[s] * (blockDim.x+stepPointers_d->nextStride);
    }
    if (stage == 5)
    {
        
        for (uint64_t o = 0; o < 2; ++o)
        {
            for (uint64_t k = 0; k < stepPointers_d->nColsStages_d[nStages + 1]; ++k)
            {
                uint64_t l = threadIdx.x + nextStrides[o];
                bufferT_[(stepPointers_d->nColsStagesAcc_d[5 * o + nStages + 1] + k) * blockDim.x + threadIdx.x] = bufferPols_[offset_pols + l * stepPointers_d->nColsStages_d[nStages + 1] + k];
            }
        }

        for (uint64_t d = 0; d < 2; ++d)
        {
            for (uint64_t i = 0; i < FIELD_EXTENSION; ++i)
            {
                bufferT_[(stepPointers_d->nColsStagesAcc_d[11] + FIELD_EXTENSION * d + i) * blockDim.x + threadIdx.x] = stepPointers_d->xDivXSubXi_d[(d * stepPointers_d->domainSize + row + threadIdx.x)*FIELD_EXTENSION+i];
            }
        }
    }
     
}

__global__ void _transposeFromBuffer(StepsPointers *stepPointers_d, uint64_t row, uint32_t stage, bool domainExtended, uint32_t istream){

    gl64_t *bufferT_ = &(stepPointers_d->bufferT_d[istream * stepPointers_d->dimBufferT]);

    if (domainExtended)
    {
        gl64_t *bufferPols_ = &(stepPointers_d->bufferPols_d[(istream+1) * stepPointers_d->dimBufferPols-stepPointers_d->nColsStages_d[10]*blockDim.x]);
        // Store either polinomial f or polinomial q
        for (uint64_t k = 0; k < stepPointers_d->nColsStages_d[10]; ++k)
        {
            bufferPols_[threadIdx.x*stepPointers_d->nColsStages_d[10]+k] = bufferT_[(stepPointers_d->nColsStagesAcc_d[10] + k) * blockDim.x + threadIdx.x];
        }
    }else{
        gl64_t *bufferPols_ = &(stepPointers_d->bufferPols_d[(istream) * stepPointers_d->dimBufferPols]);
        uint64_t nStages = 3;
        uint64_t offset_pols_d=0;
        for (uint64_t s = 2; s <= nStages + 1; ++s)
        {
            gl64_t *buffT = &bufferT_[stepPointers_d->nColsStagesAcc_d[s]*blockDim.x];
            for (uint64_t k = 0; k < stepPointers_d->nColsStages_d[s]; ++k)
            {
                bufferPols_[offset_pols_d +  threadIdx.x *stepPointers_d->nColsStages_d[s]+k] = buffT[k*blockDim.x + threadIdx.x];
            }
            offset_pols_d += stepPointers_d->nColsStages_d[s]*blockDim.x;
        }
    }
}

#endif