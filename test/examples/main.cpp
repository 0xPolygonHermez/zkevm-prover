#include <stdio.h>
#include "starks.hpp"
#include "proof2zkinStark.hpp"
#include "hint_handler.hpp"
#include "hint_handler_builder.hpp"

int main(int argc, char **argv)
{
    
    string constPolsFile;
    string constTreeFile;
    string starkInfoFile;
    string commitPols;
    string expressionsBinFile;
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
        expressionsBinFile = "test/examples/all/all.chelpers/all.chelpers_generic.bin";
    } else if(testName == "compressor") {
        constPolsFile = "test/examples/compressor/all.c18.const";
        constTreeFile = "test/examples/compressor/all.c18.consttree";
        starkInfoFile = "test/examples/compressor/all.c18.starkinfo.json";
        commitPols = "test/examples/compressor/all.c18.commit";
        verkey = "test/examples/compressor/all.c18.verkey.json";
        publicsFile = "test/examples/compressor/all.c18.publics.json";
        proofFile = "runtime/output/compressor_proof.json";
        zkinProofFile = "runtime/output/compressor_proof.zkin.json";
        expressionsBinFile = "test/examples/compressor/all.c18.chelpers/all.c18.chelpers_generic.bin";
    } else if(testName == "fibonacci_pil2") {
        constPolsFile = "test/examples/fibonacci.pil2/fibonacci.pil2.const";
        constTreeFile = "test/examples/fibonacci.pil2/fibonacci.pil2.consttree";
        starkInfoFile = "test/examples/fibonacci.pil2/fibonacci.pil2.starkinfo.json";
        commitPols = "test/examples/fibonacci.pil2/fibonacci.pil2.commit";
        verkey = "test/examples/fibonacci.pil2/fibonacci.pil2.verkey.json";
        publicsFile = "test/examples/fibonacci.pil2/fibonacci.pil2.publics.json";
        proofFile = "runtime/output/fibonacci_pil2_proof.json";
        zkinProofFile = "runtime/output/fibonacci_pil2_proof.zkin.json";
        expressionsBinFile = "test/examples/fibonacci.pil2/fibonacci.pil2.chelpers/fibonacci.pil2.chelpers_generic.bin";
    }
   
    StarkInfo starkInfo(starkInfoFile);
    ConstPols constPols(starkInfo, constPolsFile);
    ExpressionsBin expressionsBin(expressionsBinFile);

    SetupCtx setupCtx(starkInfo, expressionsBin, constPols);

    void *pCommit = copyFile(commitPols, setupCtx.starkInfo.mapSectionsN["cm1"] * sizeof(Goldilocks::Element) * (1 << setupCtx.starkInfo.starkStruct.nBits));
    void *pAddress = (void *)malloc((setupCtx.starkInfo.mapTotalN) * sizeof(Goldilocks::Element));

    uint64_t N = (1 << setupCtx.starkInfo.starkStruct.nBits);
    #pragma omp parallel for
    for (uint64_t i = 0; i < N; i += 1)
    {
        std::memcpy((uint8_t*)pAddress + setupCtx.starkInfo.mapOffsets[std::make_pair("cm1", false)]*sizeof(Goldilocks::Element) + i*setupCtx.starkInfo.mapSectionsN["cm1"]*sizeof(Goldilocks::Element), 
            (uint8_t*)pCommit + i*setupCtx.starkInfo.mapSectionsN["cm1"]*sizeof(Goldilocks::Element), 
            setupCtx.starkInfo.mapSectionsN["cm1"]*sizeof(Goldilocks::Element));
    }

    json publics;
    file2json(publicsFile, publics);

    Goldilocks::Element publicInputs[setupCtx.starkInfo.nPublics];

    for(uint64_t i = 0; i < setupCtx.starkInfo.nPublics; i++) {
        publicInputs[i] = Goldilocks::fromU64(publics[i]);
    }
    
    json publicStarkJson;
    for (uint64_t i = 0; i < setupCtx.starkInfo.nPublics; i++)
    {
        publicStarkJson[i] = Goldilocks::toString(publicInputs[i]);
    }

    nlohmann::ordered_json jProof;
    
    if(testName == "all") {
        FRIProof<Goldilocks::Element> fproof(setupCtx.starkInfo);
        ExpressionsAvx expressionsAvx(setupCtx);
        Starks<Goldilocks::Element> starks(setupCtx, expressionsAvx);
        starks.genProof((Goldilocks::Element *)pAddress, fproof, &publicInputs[0], false); 
        jProof = fproof.proof.proof2json();
    } else if(testName == "compressor") {
        FRIProof<RawFr::Element> fproof(setupCtx.starkInfo);
        ExpressionsAvx expressionsAvx(setupCtx);
        Starks<RawFr::Element> starks(setupCtx, expressionsAvx);
        starks.genProof((Goldilocks::Element *)pAddress, fproof, &publicInputs[0], false); 
        jProof = fproof.proof.proof2json();
    } else if(testName == "fibonacci_pil2") {
        FRIProof<Goldilocks::Element> fproof(setupCtx.starkInfo);
        ExpressionsAvx expressionsAvx(setupCtx);
        Starks<Goldilocks::Element> starks(setupCtx, expressionsAvx);
        starks.genProof((Goldilocks::Element *)pAddress, fproof, &publicInputs[0], false); 
        jProof = fproof.proof.proof2json();
    }

    nlohmann::json zkin = proof2zkinStark(jProof, setupCtx.starkInfo);

    // Generate publics
    jProof["publics"] = publics;
    zkin["publics"] = publics;

    json2file(zkin, zkinProofFile);
    json2file(jProof, proofFile);

    return 0;
}