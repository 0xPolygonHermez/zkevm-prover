#include <stdio.h>
#include "starks.hpp"
#include "all.steps.hpp"
#include "proof2zkinStark.hpp"

int main()
{

    Config config;
    config.runFileGenBatchProof = true;
    config.zkevmConstPols = "config/tmp/all.const";
    config.mapConstPolsFile = false;
    config.zkevmConstantsTree = "config/tmp/all.consttree";
    config.zkevmStarkInfo = "config/tmp/all.starkinfo.json";

    StarkInfo starkInfo(config, config.zkevmStarkInfo);

    Starks starks(config, {config.zkevmConstPols, config.mapConstPolsFile, config.zkevmConstantsTree, config.zkevmStarkInfo});

    uint64_t polBits = starks.starkInfo.starkStruct.steps[starks.starkInfo.starkStruct.steps.size() - 1].nBits;
    FRIProof fproof((1 << polBits), FIELD_EXTENSION, starks.starkInfo.starkStruct.steps.size(), starks.starkInfo.evMap.size(), starks.starkInfo.nPublics);

    void *pCommit = copyFile("config/tmp/all.commit", starkInfo.nCm1 * sizeof(Goldilocks::Element) * (1 << starkInfo.starkStruct.nBits));
    void *pAddress = (void *)calloc(starkInfo.mapTotalN, sizeof(uint64_t));
    std::memcpy(pAddress, pCommit, starkInfo.nCm1 * sizeof(Goldilocks::Element) * (1 << starkInfo.starkStruct.nBits));

    Goldilocks::Element publicInputs[3] = {Goldilocks::fromU64(1), Goldilocks::fromU64(2), Goldilocks::fromU64(74469561660084004)};
    json publicStarkJson;
    publicStarkJson[0] = Goldilocks::toString(publicInputs[0]);
    publicStarkJson[1] = Goldilocks::toString(publicInputs[1]);
    publicStarkJson[2] = Goldilocks::toString(publicInputs[2]);
    AllSteps allteps;
    starks.genProof(pAddress, fproof, &publicInputs[0], &allteps);

    nlohmann::ordered_json jProof = fproof.proofs.proof2json();
    nlohmann::json zkin = proof2zkinStark(jProof);
    // Generate publics
    jProof["publics"] = publicStarkJson;
    zkin["publics"] = publicStarkJson;

    json2file(zkin, "zkin.json");
    json2file(jProof, "jProof.json");

    return 0;
}