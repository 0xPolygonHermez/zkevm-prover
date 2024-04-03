#include <stdio.h>
#include "starks.hpp"
#include "proof2zkinStark.hpp"
#include "AllSteps.hpp"
#include "AllC18Steps.hpp"


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
    string proofFile;
    string zkinProofFile;

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
        proofFile = "runtime/output/all_proof.json";
        zkinProofFile = "runtime/output/all_proof.zkin.json";

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
        proofFile = "runtime/output/compressor_proof.json";
        zkinProofFile = "runtime/output/compressor_proof.zkin.json";

        if(USE_GENERIC_PARSER) {
            cHelpersFile = "test/examples/compressor/all.c18.chelpers/all.c18.chelpers_generic.bin";
        } else {
            cHelpersFile = "test/examples/compressor/all.c18.chelpers/all.c18.chelpers.bin";
        }
    }
   
    StarkInfo starkInfo(starkInfoFile);

    void *pCommit = copyFile(commitPols, starkInfo.mapSectionsN["cm1"] * sizeof(Goldilocks::Element) * (1 << starkInfo.starkStruct.nBits));
    void *pAddress = (void *)malloc((starkInfo.mapTotalN + starkInfo.mapSectionsN["cm3"] * (1 << starkInfo.starkStruct.nBitsExt)) * sizeof(Goldilocks::Element));

    uint64_t N = (1 << starkInfo.starkStruct.nBits);
    #pragma omp parallel for
    for (uint64_t i = 0; i < N; i += 1)
    {
        std::memcpy((uint8_t*)pAddress + i*starkInfo.mapSectionsN["cm1"]*sizeof(Goldilocks::Element), (uint8_t*)pCommit + i*starkInfo.mapSectionsN["cm1"]*sizeof(Goldilocks::Element), starkInfo.mapSectionsN["cm1"]*sizeof(Goldilocks::Element));
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
            AllSteps allSteps;
            starks.genProof(fproof, &publicInputs[0], &allSteps);
        }
        jProof = fproof.proofs.proof2json();
    } else if(testName == "compressor") {
        
        FRIProof<RawFr::Element> fproof(starkInfo, 1);
        Starks<RawFr::Element> starks(config, {constPols, config.mapConstPolsFile, constTree, starkInfoFile, cHelpersFile}, pAddress, false);
        if(USE_GENERIC_PARSER) {
            CHelpersSteps cHelpersSteps;
            starks.genProof(fproof, &publicInputs[0], &cHelpersSteps); 
        } else {
            AllC18Steps allC18Steps;
            starks.genProof(fproof, &publicInputs[0], &allC18Steps);
        }
        jProof = fproof.proofs.proof2json();
    }

    nlohmann::json zkin = proof2zkinStark(jProof, starkInfo);

    // Generate publics
    jProof["publics"] = publics;
    zkin["publics"] = publics;

    json2file(zkin, zkinProofFile);
    json2file(jProof, proofFile);

    return 0;
}