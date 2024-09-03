#include <stdio.h>
#include "starks.hpp"
#include "proof2zkinStark.hpp"
#include "chelpers_steps.hpp"
#ifdef __AVX512__
    #include "chelpers_steps_avx512.hpp"
#endif
#include "chelpers_steps_pack.hpp"
#include "chelpers_steps_gpu.cuh"
#include "AllSteps.hpp"
#include "zklog.hpp"
#include "exit_process.hpp"
#include "cuda_utils.hpp"

int main()
{

    Config config;
    config.runFileGenBatchProof = true; // So that starkInfo is created
    config.mapConstPolsFile = false;
    config.mapConstantsTreeFile = false;
    
    string constPols = "test/examples/all/all.const";
    string constTree = "test/examples/all/all.consttree";
    string starkInfoFile = "test/examples/all/all.starkinfo.json";
    string commitPols = "test/examples/all/all.commit";
    string verkey = "test/examples/all/all.verkey.json";

    string cHelpersFile;

    if(USE_GENERIC_PARSER) {
        cHelpersFile = "test/examples/all/all.chelpers/all.chelpers_generic.bin";
    } else {
        cHelpersFile = "test/examples/all/all.chelpers/all.chelpers.bin";
    }

    bool reduceMemory = true;

    StarkInfo starkInfo(starkInfoFile, reduceMemory);

#if defined(__USE_CUDA__) && defined(ENABLE_EXPERIMENTAL_CODE)
    alloc_pinned_mem_per_device((1 << starkInfo.starkStruct.nBitsExt) * 32);
#endif

    uint64_t polBits = starkInfo.starkStruct.steps[starkInfo.starkStruct.steps.size() - 1].nBits;
    FRIProof fproof((1 << polBits), FIELD_EXTENSION, starkInfo.starkStruct.steps.size(), starkInfo.evMap.size(), starkInfo.nPublics);

    void *pCommit = loadFileParallel(commitPols, starkInfo.mapSectionsN.section[cm1_n] * sizeof(Goldilocks::Element) * (1 << starkInfo.starkStruct.nBits));
    void *pAddress = (void *)malloc(starkInfo.mapTotalN * sizeof(Goldilocks::Element));
    if (pAddress == NULL)
    {
        zklog.error("Memory allocation failed for pAddress");
        exitProcess();
    }

    uint64_t N = (1 << starkInfo.starkStruct.nBits);
    #pragma omp parallel for
    for (uint64_t i = 0; i < N; i += 1)
    {
        std::memcpy((uint8_t*)pAddress + starkInfo.mapOffsets.section[cm1_n]*sizeof(Goldilocks::Element) + i*starkInfo.mapSectionsN.section[cm1_n]*sizeof(Goldilocks::Element), 
            (uint8_t*)pCommit + i*starkInfo.mapSectionsN.section[cm1_n]*sizeof(Goldilocks::Element), 
            starkInfo.mapSectionsN.section[cm1_n]*sizeof(Goldilocks::Element));
    }

    Starks starks(config, {constPols, config.mapConstPolsFile, constTree, starkInfoFile, cHelpersFile}, reduceMemory, pAddress);

    Goldilocks::Element publicInputs[3] = {
        Goldilocks::fromU64(1),
        Goldilocks::fromU64(2),
        Goldilocks::fromU64(74469561660084004),
    };

    json publicStarkJson;
    for (int i = 0; i < 3; i++)
    {
        publicStarkJson[i] = Goldilocks::toString(publicInputs[i]);
    }

    json allVerkeyJson;
    file2json(verkey, allVerkeyJson);
    Goldilocks::Element allVerkey[4];
    allVerkey[0] = Goldilocks::fromU64(allVerkeyJson["constRoot"][0]);
    allVerkey[1] = Goldilocks::fromU64(allVerkeyJson["constRoot"][1]);
    allVerkey[2] = Goldilocks::fromU64(allVerkeyJson["constRoot"][2]);
    allVerkey[3] = Goldilocks::fromU64(allVerkeyJson["constRoot"][3]);

    if(USE_GENERIC_PARSER) {

#if defined(__USE_CUDA__) && defined(ENABLE_EXPERIMENTAL_CODE)
        CHelpersStepsGPU cHelpersSteps;
#elif defined(__AVX512__)
        CHelpersStepsAvx512 cHelpersSteps;
#elif defined(__PACK__) 
        CHelpersStepsPack cHelpersSteps;
        cHelpersSteps.nrowsPack = NROWS_PACK;
#else
         CHelpersSteps cHelpersSteps;
#endif
        starks.genProof(fproof, &publicInputs[0], allVerkey, &cHelpersSteps); 
    } else {
        AllSteps allSteps;
        starks.genProof(fproof, &publicInputs[0], allVerkey, &allSteps);
    }

    nlohmann::ordered_json jProof = fproof.proofs.proof2json();
    nlohmann::json zkin = proof2zkinStark(jProof);
    // Generate publics
    jProof["publics"] = publicStarkJson;
    zkin["publics"] = publicStarkJson;

    json2file(publicStarkJson, "runtime/output/publics.json");
    json2file(zkin, "runtime/output/zkin.json");
    json2file(jProof, "runtime/output/jProof.json");

    return 0;
}