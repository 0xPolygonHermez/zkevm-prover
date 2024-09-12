#include <stdio.h>
#include "starks.hpp"
#include "proof2zkinStark.hpp"
#include "hint_handler.hpp"
#include "hint_handler_builder.hpp"
#include "gen_recursive_proof.hpp"

#include <dlfcn.h> // Required for dlopen, dlsym, dlclose
typedef void (*FunctionType)(void*, uint64_t, uint64_t, const std::string, const std::string,  nlohmann::json);

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

    string zkinAllFile;
    string execFile;
    string datFile;

    // Check arguments list
    if (argc != 2)
    {
        cout << "Error: expected 1 arguments but got " << argc - 1 << " Usage: zkProverTest <testname>" << endl;
        return -1;
    }

    string testName = argv[1];

    if(testName != "compressor" && testName != "fibonacci_pil2") {
        cout << "Error: unknown test name " << testName << endl;
        return -1;
    }

    HintHandlerBuilder::registerBuilder(H1H2HintHandler::getName(), std::make_unique<H1H2HintHandlerBuilder>());
    HintHandlerBuilder::registerBuilder(GProdHintHandler::getName(), std::make_unique<GProdHintHandlerBuilder>());
    HintHandlerBuilder::registerBuilder(GSumHintHandler::getName(), std::make_unique<GSumHintHandlerBuilder>());

    bool circomLib = true;

    if(testName == "compressor") {
        zkinAllFile = "test/examples/compressor/circom/all.proof.zkin.json";
        execFile = "test/examples/compressor/circom/all.c18.exec";
        datFile = "test/examples/compressor/circom/all.verifier.dat";
        constPolsFile = "test/examples/compressor/all.c18.const";
        constTreeFile = "test/examples/compressor/all.c18.consttree";
        starkInfoFile = "test/examples/compressor/all.c18.starkinfo.json";
        commitPols = "test/examples/compressor/all.c18.commit";
        verkey = "test/examples/compressor/all.c18.verkey.json";
        publicsFile = "test/examples/compressor/all.c18.publics.json";
        proofFile = "runtime/output/compressor_proof.json";
        zkinProofFile = "runtime/output/compressor_proof.zkin.json";
        expressionsBinFile = "test/examples/compressor/all.c18.chelpers.bin";
    } else if(testName == "fibonacci_pil2") {
        constPolsFile = "test/examples/fibonacci.pil2/fibonacci.pil2.const";
        constTreeFile = "test/examples/fibonacci.pil2/fibonacci.pil2.consttree";
        starkInfoFile = "test/examples/fibonacci.pil2/fibonacci.pil2.starkinfo.json";
        commitPols = "test/examples/fibonacci.pil2/fibonacci.pil2.commit";
        verkey = "test/examples/fibonacci.pil2/fibonacci.pil2.verkey.json";
        publicsFile = "test/examples/fibonacci.pil2/fibonacci.pil2.publics.json";
        proofFile = "runtime/output/fibonacci_pil2_proof.json";
        zkinProofFile = "runtime/output/fibonacci_pil2_proof.zkin.json";
        expressionsBinFile = "test/examples/fibonacci.pil2/fibonacci.pil2.chelpers.bin";
    }
   
    StarkInfo starkInfo(starkInfoFile);
    ConstPols constPols(starkInfo, constPolsFile);
    ExpressionsBin expressionsBin(expressionsBinFile);

    SetupCtx setupCtx(starkInfo, expressionsBin, constPols);

    void *pAddress = (void *)malloc((setupCtx.starkInfo.mapTotalN) * sizeof(Goldilocks::Element));
    uint64_t N = (1 << setupCtx.starkInfo.starkStruct.nBits);
    uint64_t nCm1 = setupCtx.starkInfo.mapSectionsN["cm1"];

    if(testName == "fibonacci_pil2" || !circomLib) {
        void *pCommit = copyFile(commitPols, nCm1 * sizeof(Goldilocks::Element) * (1 << setupCtx.starkInfo.starkStruct.nBits));

        #pragma omp parallel for
        for (uint64_t i = 0; i < N; i += 1)
        {
            std::memcpy((uint8_t*)pAddress + setupCtx.starkInfo.mapOffsets[std::make_pair("cm1", false)]*sizeof(Goldilocks::Element) + i*nCm1*sizeof(Goldilocks::Element), 
                (uint8_t*)pCommit + i*nCm1*sizeof(Goldilocks::Element), 
                nCm1*sizeof(Goldilocks::Element));
        }
    } else {
        json zkin;
        file2json(zkinAllFile, zkin);
        
        void* pAddressCm1 = (uint8_t *)pAddress + setupCtx.starkInfo.mapOffsets[std::make_pair("cm1", false)] * sizeof(Goldilocks::Element);

        // Load the dynamic library
        void* handle = dlopen("./witness_lib/witness.so", RTLD_LAZY);
        if (!handle) {
            cout << "Cannot load library: " << dlerror() << std::endl;
            return 1;
        }
        
        // Clear any existing errors
        dlerror();
        
        // Load the symbol (function)
        FunctionType function = (FunctionType)dlsym(handle, "getCommitedPols");
        const char* dlsym_error = dlerror();
        if (dlsym_error) {
            cout << "Something went wrong" << endl;
            dlclose(handle);
            return 1;
        }
        // Use the function
        function(pAddressCm1, N, setupCtx.starkInfo.mapSectionsN["cm1"], datFile, execFile, zkin);
        
        // Close the library
        dlclose(handle);
    }

    cout << "FINISH" << endl;
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
    
    if(testName == "compressor") {
        FRIProof<Goldilocks::Element> fproof(setupCtx.starkInfo);
        genRecursiveProof(setupCtx, (Goldilocks::Element *)pAddress, fproof, &publicInputs[0]);
        jProof = fproof.proof.proof2json();
    } else if(testName == "fibonacci_pil2") {
        FRIProof<Goldilocks::Element> fproof(setupCtx.starkInfo);
        Starks<Goldilocks::Element> starks(setupCtx);
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