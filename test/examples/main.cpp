#include <stdio.h>
#include "starks.hpp"
#include "proof2zkinStark.hpp"
#include "AllSteps.hpp"

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
    string cHelpersFile = "test/examples/all/all.chelpers/all.chelpers.bin";
    string verkey = "test/examples/all/all.verkey.json";

    StarkInfo starkInfo(config, starkInfoFile);

    FRIProof<Goldilocks::Element> fproof(starkInfo, 4);

    void *pCommit = copyFile(commitPols, starkInfo.nCm1 * sizeof(Goldilocks::Element) * (1 << starkInfo.starkStruct.nBits));
    void *pAddress = (void *)calloc(starkInfo.mapTotalN + (starkInfo.mapSectionsN.section[eSection::cm1_n] * (1 << starkInfo.starkStruct.nBits) * FIELD_EXTENSION ), sizeof(uint64_t));

    Starks<Goldilocks::Element> starks(config, {constPols, config.mapConstPolsFile, constTree, starkInfoFile, cHelpersFile}, pAddress);

    uint64_t N = (1 << starkInfo.starkStruct.nBits);
    #pragma omp parallel for
    for (uint64_t i = 0; i < N; i += 1)
    {
        std::memcpy((uint8_t*)pAddress + i*starkInfo.nCm1*sizeof(Goldilocks::Element), (uint8_t*)pCommit + i*starkInfo.nCm1*sizeof(Goldilocks::Element), starkInfo.nCm1*sizeof(Goldilocks::Element));
    }

    Goldilocks::Element publicInputs[3] = {
        Goldilocks::fromU64(1),
        Goldilocks::fromU64(2),
        // Goldilocks::fromU64(74469561660084004),
        Goldilocks::fromU64(16222592280316846942ULL),
    };

    json publicStarkJson;
    for (int i = 0; i < 3; i++)
    {
        publicStarkJson[i] = Goldilocks::toString(publicInputs[i]);
    }

    AllSteps allSteps;
    starks.genProof(fproof, &publicInputs[0], &allSteps);

    nlohmann::ordered_json jProof = fproof.proofs.proof2json();

    nlohmann::json zkin = proof2zkinStark(jProof, starkInfo);

    // Generate publics
    jProof["publics"] = publicStarkJson;
    zkin["publics"] = publicStarkJson;

    json2file(publicStarkJson, "runtime/output/publics.json");
    json2file(zkin, "runtime/output/zkin.json");
    json2file(jProof, "runtime/output/jProof.json");

    return 0;
}