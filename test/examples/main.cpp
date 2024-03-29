#include <stdio.h>
#include "starks.hpp"
#include "proof2zkinStark.hpp"
#include "AllPil2Steps.hpp"
#include "AllC18Pil2Steps.hpp"


int main(int argc, char **argv)
{
    Config config;
    config.mapConstPolsFile = false;
    config.mapConstantsTreeFile = false;
    
    string constPols;
    string constTree;
    string starkInfoFile;
    string commitPols;
    string cHelpersFile;
    string verkey;
    string publicsFile;

    // Check arguments list
    if (argc != 2)
    {
        cout << "Error: expected 1 arguments but got " << argc - 1 << " Usage: zkProverTest <testname>" << endl;
        return -1;
    }

    string testName = argv[1];

    if(testName != "all" && testName != "compressor") {
        cout << "Error: unknown test name " << testName << endl;
        return -1;
    }

    if(testName == "all") {
        constPols = "test/examples/all/all.const";
        constTree = "test/examples/all/all.consttree";
        starkInfoFile = "test/examples/all/all.starkinfo.json";
        commitPols = "test/examples/all/all.commit";
        verkey = "test/examples/all/all.verkey.json";
        publicsFile = "test/examples/all/all.publics.json";

        if(USE_GENERIC_PARSER) {
            cHelpersFile = "test/examples/all/all.chelpers/all.chelpers_generic.bin";
        } else {
            cHelpersFile = "test/examples/all/all.chelpers/all.chelpers.bin";
        }
    } else if(testName == "compressor") {
        constPols = "test/examples/compressor/all.c18.const";
        constTree = "test/examples/compressor/all.c18.consttree";
        starkInfoFile = "test/examples/compressor/all.c18.starkinfo.json";
        commitPols = "test/examples/compressor/all.c18.commit";
        verkey = "test/examples/compressor/all.c18.verkey.json";
        publicsFile = "test/examples/compressor/all.c18.publics.json";
        if(USE_GENERIC_PARSER) {
            cHelpersFile = "test/examples/compressor/all.c18.chelpers/all.c18.chelpers_generic.bin";
        } else {
            cHelpersFile = "test/examples/compressor/all.c18.chelpers/all.c18.chelpers.bin";
        }
    }
   
    StarkInfo starkInfo(starkInfoFile);

    void *pCommit = copyFile(commitPols, starkInfo.nCm1 * sizeof(Goldilocks::Element) * (1 << starkInfo.starkStruct.nBits));
    void *pAddress = (void *)malloc((starkInfo.mapTotalN + starkInfo.mapSectionsN.section[eSection::cm3_n] * (1 << starkInfo.starkStruct.nBitsExt)) * sizeof(Goldilocks::Element));

    uint64_t N = (1 << starkInfo.starkStruct.nBits);
    #pragma omp parallel for
    for (uint64_t i = 0; i < N; i += 1)
    {
        std::memcpy((uint8_t*)pAddress + i*starkInfo.nCm1*sizeof(Goldilocks::Element), (uint8_t*)pCommit + i*starkInfo.nCm1*sizeof(Goldilocks::Element), starkInfo.nCm1*sizeof(Goldilocks::Element));
    }


    json publics;
    file2json(publicsFile, publics);

    Goldilocks::Element publicInputs[starkInfo.nPublics];

    for(uint64_t i = 0; i < starkInfo.nPublics; i++) {
        publicInputs[i] = Goldilocks::fromU64(publics[i]);
    }
    
    json publicStarkJson;
    for (uint64_t i = 0; i < starkInfo.nPublics; i++)
    {
        publicStarkJson[i] = Goldilocks::toString(publicInputs[i]);
    }

    nlohmann::ordered_json jProof;

    if(testName == "all") {
        FRIProof<Goldilocks::Element> fproof(starkInfo, 4);
        Starks<Goldilocks::Element> starks(config, {constPols, config.mapConstPolsFile, constTree, starkInfoFile, cHelpersFile}, pAddress, false);
        if(USE_GENERIC_PARSER) {
            CHelpersSteps cHelpersSteps;
            starks.genProof(fproof, &publicInputs[0], &cHelpersSteps); 
        } else {
            AllPil2Steps allPil2Steps;
            starks.genProof(fproof, &publicInputs[0], &allPil2Steps);
        }
        jProof = fproof.proofs.proof2json();
    } else if(testName == "compressor") {
        
        FRIProof<RawFr::Element> fproof(starkInfo, 1);
        Starks<RawFr::Element> starks(config, {constPols, config.mapConstPolsFile, constTree, starkInfoFile, cHelpersFile}, pAddress, false);
        if(USE_GENERIC_PARSER) {
            CHelpersSteps cHelpersSteps;
            starks.genProof(fproof, &publicInputs[0], &cHelpersSteps); 
        } else {
            AllC18Pil2Steps allC18Pil2Steps;
            starks.genProof(fproof, &publicInputs[0], &allC18Pil2Steps);
        }
        jProof = fproof.proofs.proof2json();
    }

    nlohmann::json zkin = proof2zkinStark(jProof, starkInfo);

    // Generate publics
    jProof["publics"] = publics;
    zkin["publics"] = publics;

    json2file(publicStarkJson, "runtime/output/publics.json");
    json2file(zkin, "runtime/output/zkin.json");
    json2file(jProof, "runtime/output/jProof.json");

    return 0;
}