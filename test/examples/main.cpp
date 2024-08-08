#include <stdio.h>
#include "starks.hpp"
#include "proof2zkinStark.hpp"
#include "AllSteps.hpp"
#include "AllC18Steps.hpp"
#include "FibonacciPil2Steps.hpp"
#include "hint_handler.hpp"
#include "hint_handler_builder.hpp"

int main(int argc, char **argv)
{
    Config config;
    
    string constPolsFile;
    string constTreeFile;
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

    if(testName != "all" && testName != "compressor" && testName != "fibonacci_pil2") {
        cout << "Error: unknown test name " << testName << endl;
        return -1;
    }

    HintHandlerBuilder::registerBuilder(H1H2HintHandler::getName(), std::make_unique<H1H2HintHandlerBuilder>());
    HintHandlerBuilder::registerBuilder(GProdHintHandler::getName(), std::make_unique<GProdHintHandlerBuilder>());
    HintHandlerBuilder::registerBuilder(GSumHintHandler::getName(), std::make_unique<GSumHintHandlerBuilder>());

    if(testName == "all") {
        constPolsFile = "test/examples/all/all.const";
        constTreeFile = "test/examples/all/all.consttree";
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
        constPolsFile = "test/examples/compressor/all.c18.const";
        constTreeFile = "test/examples/compressor/all.c18.consttree";
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
    } else if(testName == "fibonacci_pil2") {
        constPolsFile = "test/examples/fibonacci.pil2/fibonacci.pil2.const";
        constTreeFile = "test/examples/fibonacci.pil2/fibonacci.pil2.consttree";
        starkInfoFile = "test/examples/fibonacci.pil2/fibonacci.pil2.starkinfo.json";
        commitPols = "test/examples/fibonacci.pil2/fibonacci.pil2.commit";
        verkey = "test/examples/fibonacci.pil2/fibonacci.pil2.verkey.json";
        publicsFile = "test/examples/fibonacci.pil2/fibonacci.pil2.publics.json";
        proofFile = "runtime/output/fibonacci_pil2_proof.json";
        zkinProofFile = "runtime/output/fibonacci_pil2_proof.zkin.json";

        if(USE_GENERIC_PARSER) {
            cHelpersFile = "test/examples/fibonacci.pil2/fibonacci.pil2.chelpers/fibonacci.pil2.chelpers_generic.bin";
        } else {
            cHelpersFile = "test/examples/fibonacci.pil2/fibonacci.pil2.chelpers/fibonacci.pil2.chelpers.bin";
        }
    }
   
    StarkInfo starkInfo(starkInfoFile);
    CHelpers cHelpers(cHelpersFile);

    void *pCommit = copyFile(commitPols, starkInfo.mapSectionsN["cm1"] * sizeof(Goldilocks::Element) * (1 << starkInfo.starkStruct.nBits));
    void *pAddress = (void *)malloc((starkInfo.mapTotalN) * sizeof(Goldilocks::Element));

    uint64_t N = (1 << starkInfo.starkStruct.nBits);
    #pragma omp parallel for
    for (uint64_t i = 0; i < N; i += 1)
    {
        std::memcpy((uint8_t*)pAddress + starkInfo.mapOffsets[std::make_pair("cm1", false)]*sizeof(Goldilocks::Element) + i*starkInfo.mapSectionsN["cm1"]*sizeof(Goldilocks::Element), 
            (uint8_t*)pCommit + i*starkInfo.mapSectionsN["cm1"]*sizeof(Goldilocks::Element), 
            starkInfo.mapSectionsN["cm1"]*sizeof(Goldilocks::Element));
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
        FRIProof<Goldilocks::Element> fproof(starkInfo);
        ConstPols<Goldilocks::Element> constPols(starkInfo, constPolsFile);
        Starks<Goldilocks::Element> starks(config, pAddress, starkInfo, cHelpers, constPols, false);
        if(USE_GENERIC_PARSER) {
            CHelpersSteps cHelpersSteps;
            starks.genProof(fproof, &publicInputs[0], &cHelpersSteps); 
        } else {
            AllSteps allSteps;
            starks.genProof(fproof, &publicInputs[0], &allSteps);
        }
        jProof = fproof.proofs.proof2json();
    } else if(testName == "compressor") {
        FRIProof<RawFr::Element> fproof(starkInfo);
        ConstPols<RawFr::Element> constPols(starkInfo, constPolsFile);
        Starks<RawFr::Element> starks(config, pAddress, starkInfo, cHelpers, constPols, false);
        if(USE_GENERIC_PARSER) {
            CHelpersSteps cHelpersSteps;
            starks.genProof(fproof, &publicInputs[0], &cHelpersSteps); 
        } else {
            AllC18Steps allC18Steps;
            starks.genProof(fproof, &publicInputs[0], &allC18Steps);
        }
        jProof = fproof.proofs.proof2json();
    } else if(testName == "fibonacci_pil2") {
        FRIProof<Goldilocks::Element> fproof(starkInfo);
        ConstPols<Goldilocks::Element> constPols(starkInfo, constPolsFile);
        Starks<Goldilocks::Element> starks(config, pAddress, starkInfo, cHelpers, constPols, false);
        if(USE_GENERIC_PARSER) {
            CHelpersSteps cHelpersSteps;
            starks.genProof(fproof, &publicInputs[0], &cHelpersSteps); 
        } else {
            FibonacciPil2Steps fibonacciPil2Steps;
            starks.genProof(fproof, &publicInputs[0], &fibonacciPil2Steps);
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