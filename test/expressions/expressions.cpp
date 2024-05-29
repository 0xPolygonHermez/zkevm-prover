#include <iostream>
#include "config.hpp"
#include "definitions.hpp"
#include "zkassert.hpp"
#include "zklog.hpp"
#include "timer.hpp"
#include "stark_info.hpp"
#include "polinomial.hpp"
#include "constant_pols_starks.hpp"
#include "binfile_utils.hpp"
#include "steps.hpp"
#include "chelpers.hpp"
#include "chelpers_steps.hpp"
#include "chelpers_steps_pack.hpp"
#ifdef __AVX512__
#include "chelpers_steps_avx512.hpp"
#endif
#include "ZkevmSteps.hpp"


#define NUM_ITERATIONS 2

int main()
{
    CHelpers cHelpersTest;
    CHelpers cHelpersTestGeneric;

    CHelpers cHelpers;
    CHelpers cHelpersGeneric;

    string starkInfoFile = "test/expressions/zkevm.starkinfo.json";

    string cHelpersGenericFile = "config/zkevm/zkevm.chelpers_generic.bin";
    string cHelpersFile = "config/zkevm/zkevm.chelpers.bin";


    TimerStart(CHELPERS_ALLOCATION);

    std::unique_ptr<BinFileUtils::BinFile> cHelpersGenericBinFile = BinFileUtils::openExisting(cHelpersGenericFile, "chps", 1);
    cHelpersGeneric.loadCHelpers(cHelpersGenericBinFile.get());

    std::unique_ptr<BinFileUtils::BinFile> cHelpersBinFile = BinFileUtils::openExisting(cHelpersFile, "chps", 1);
    cHelpers.loadCHelpers(cHelpersBinFile.get());

    TimerStopAndLog(CHELPERS_ALLOCATION);

    StarkInfo starkInfo(starkInfoFile);

    uint64_t N = 1 << starkInfo.starkStruct.nBits;
    uint64_t NExtended = 1 << starkInfo.starkStruct.nBitsExt;

    Goldilocks::Element* mem = (Goldilocks::Element *)malloc(starkInfo.mapTotalN * sizeof(Goldilocks::Element));
    Goldilocks::Element *pConstPolsAddress = (Goldilocks::Element *)malloc(N * starkInfo.nConstants * sizeof(Goldilocks::Element));
    Goldilocks::Element *pConstPolsAddressExt = (Goldilocks::Element *)malloc(NExtended * starkInfo.nConstants * sizeof(Goldilocks::Element));

    ConstantPolsStarks *pConstPols = new ConstantPolsStarks(pConstPolsAddress, N, starkInfo.nConstants);
    ConstantPolsStarks *pConstPols2ns = new ConstantPolsStarks(pConstPolsAddressExt, NExtended, starkInfo.nConstants);
    Polinomial challenges(8, FIELD_EXTENSION);
    Polinomial x_n(N, 1);
    Polinomial x_2ns(NExtended, 1);
    Polinomial zi(NExtended, 1);
    Polinomial evals(starkInfo.evMap.size(), FIELD_EXTENSION);
    Polinomial xDivXSubXi(&mem[starkInfo.mapOffsets.section[eSection::xDivXSubXi_2ns]], 2 * NExtended, FIELD_EXTENSION, FIELD_EXTENSION);
    Goldilocks::Element publicInputs[starkInfo.nPublics];
    Goldilocks::Element *p_q_2ns = &mem[starkInfo.mapOffsets.section[eSection::q_2ns]];
    Goldilocks::Element *p_f_2ns = &mem[starkInfo.mapOffsets.section[eSection::f_2ns]];


    TimerStart(INITIALIZE_MEMORY_RANDOM);
#pragma omp parallel for
    for(uint64_t i = 0; i < starkInfo.mapTotalN; ++i) {
        mem[i] = Goldilocks::fromU64(i);
    }
    TimerStopAndLog(INITIALIZE_MEMORY_RANDOM);

    TimerStart(INITIALIZE_CONSTANTS_RANDOM);
#pragma omp parallel for
    for(uint64_t i = 0; i < starkInfo.nConstants*N; ++i) {
        pConstPolsAddress[i] = Goldilocks::fromU64(i);
    }

#pragma omp parallel for
    for(uint64_t i = 0; i < starkInfo.nConstants*NExtended; ++i) {
        pConstPolsAddressExt[i] = Goldilocks::fromU64(i);
    }
    TimerStopAndLog(INITIALIZE_CONSTANTS_RANDOM);


    TimerStart(INITIALIZE_CHALLENGES_RANDOM);
    for(uint64_t i = 0; i < challenges.length(); ++i) {
        challenges.address()[i] = Goldilocks::fromU64(rand());
    }
    TimerStopAndLog(INITIALIZE_CHALLENGES_RANDOM);

    
    TimerStart(COMPUTE_X_N_AND_X_2_NS);
    Goldilocks::Element xx = Goldilocks::one();
    for (uint64_t i = 0; i < N; i++)
    {
        *x_n[i] = xx;
        Goldilocks::mul(xx, xx, Goldilocks::w(starkInfo.starkStruct.nBits));
    }
    xx = Goldilocks::shift();
    for (uint64_t i = 0; i < NExtended; i++)
    {
        *x_2ns[i] = xx;
        Goldilocks::mul(xx, xx, Goldilocks::w(starkInfo.starkStruct.nBitsExt));
    }
    TimerStopAndLog(COMPUTE_X_N_AND_X_2_NS);

    TimerStart(COMPUTE_ZHINV);
    Polinomial::buildZHInv(zi, starkInfo.starkStruct.nBits, starkInfo.starkStruct.nBitsExt);
    TimerStopAndLog(COMPUTE_ZHINV);

    TimerStart(INITIALIZE_EVALS_RANDOM);
    for(uint64_t i = 0; i < evals.length(); ++i) {
        evals.address()[i] = Goldilocks::fromU64(rand());
    }
    TimerStopAndLog(INITIALIZE_EVALS_RANDOM);


    TimerStart(INITIALIZE_XDIVXSUBXI_RANDOM);
    for(uint64_t i = 0; i < xDivXSubXi.length(); ++i) {
        xDivXSubXi.address()[i] = Goldilocks::fromU64(rand());
    }
    TimerStopAndLog(INITIALIZE_XDIVXSUBXI_RANDOM);

    
    TimerStart(INITIALIZE_PUBLICS_RANDOM);
    for(uint64_t i = 0; i < starkInfo.nPublics; ++i) {
        publicInputs[i] = Goldilocks::fromU64(rand());
    }
    TimerStopAndLog(INITIALIZE_PUBLICS_RANDOM);

    StepsParams params = {
        pols : mem,
        pConstPols : pConstPols,
        pConstPols2ns : pConstPols2ns,
        challenges : challenges,
        x_n : x_n,
        x_2ns : x_2ns,
        zi : zi,
        evals : evals,
        xDivXSubXi : xDivXSubXi,
        publicInputs : publicInputs,
        q_2ns : p_q_2ns,
        f_2ns : p_f_2ns
    };


    CHelpersSteps cHelpersStepsGeneric;
    CHelpersStepsPack cHelpersStepsPackGeneric;
#ifdef __AVX512__
    CHelpersStepsAvx512 cHelpersStepsAvx512Generic;
#endif

    ZkevmSteps zkevmSteps;

   for(uint64_t i = 0; i < NUM_ITERATIONS; ++i) {
        TimerStart(CALCULATING_EXPRESSIONS_ZKEVM);
        zkevmSteps.calculateExpressions(starkInfo, params, cHelpers.cHelpersArgs, cHelpers.stagesInfo["step4"]);
        TimerStopAndLog(CALCULATING_EXPRESSIONS_ZKEVM);
    }

    for(uint64_t i = 0; i < NUM_ITERATIONS; ++i) {
        TimerStart(CALCULATING_EXPRESSIONS_ZKEVM_GENERIC);
        cHelpersStepsGeneric.calculateExpressions(starkInfo, params, cHelpersGeneric.cHelpersArgs, cHelpersGeneric.stagesInfo["step4"]);
        TimerStopAndLog(CALCULATING_EXPRESSIONS_ZKEVM_GENERIC);
    }

#ifdef __AVX512__
    for(uint64_t i = 0; i < NUM_ITERATIONS; ++i) {
        TimerStart(CALCULATING_EXPRESSIONS_ZKEVM_AVX512_GENERIC);
        cHelpersStepsAvx512Generic.calculateExpressions(starkInfo, params, cHelpers.cHelpersArgs, cHelpers.stagesInfo["step4"]);
        TimerStopAndLog(CALCULATING_EXPRESSIONS_ZKEVM_AVX512_GENERIC);
    }
#endif

    for(uint64_t i = 0; i < NUM_ITERATIONS; ++i) {
        TimerStart(CALCULATING_EXPRESSIONS_ZKEVM_PACK_GENERIC);
        cHelpersStepsPackGeneric.calculateExpressions(starkInfo, params, cHelpersGeneric.cHelpersArgs, cHelpersGeneric.stagesInfo["step4"]);
        TimerStopAndLog(CALCULATING_EXPRESSIONS_ZKEVM_PACK_GENERIC);
    }

    free(mem);
    free(pConstPolsAddress);
    free(pConstPolsAddressExt);

    delete pConstPols;
    delete pConstPols2ns;
}