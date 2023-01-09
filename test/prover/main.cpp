#include <stdio.h>
#include "starks.hpp"
#include "proof2zkinStark.hpp"
#include "zkevmSteps.hpp"

int main()
{

    Config config;
    config.runFileGenBatchProof = true;
    config.zkevmConstPols = "config/zkevm/zkevm.const";
    config.mapConstPolsFile = false;
    config.zkevmConstantsTree = "config/zkevm/zkevm.consttree";
    config.zkevmStarkInfo = "config/zkevm/zkevm.starkinfo.json";

    StarkInfo starkInfo(config, config.zkevmStarkInfo);


    uint64_t polBits = starkInfo.starkStruct.steps[starkInfo.starkStruct.steps.size() - 1].nBits;
    FRIProof fproof((1 << polBits), FIELD_EXTENSION, starkInfo.starkStruct.steps.size(), starkInfo.evMap.size(), starkInfo.nPublics);

    void *pCommit = copyFile("config/zkevm/zkevm.commit", starkInfo.nCm1 * sizeof(Goldilocks::Element) * (1 << starkInfo.starkStruct.nBits));
    void *pAddress = (void *)calloc(starkInfo.mapTotalN + (starkInfo.mapSectionsN.section[eSection::cm1_n] * (1 << starkInfo.starkStruct.nBits) * FIELD_EXTENSION ), sizeof(uint64_t));

        Starks starks(config, {config.zkevmConstPols, config.mapConstPolsFile, config.zkevmConstantsTree, config.zkevmStarkInfo},pAddress);


    std::memcpy(pAddress, pCommit, starkInfo.nCm1 * sizeof(Goldilocks::Element) * (1 << starkInfo.starkStruct.nBits));

    Goldilocks::Element publicInputs[47] = {
        Goldilocks::fromU64(3248459814),
        Goldilocks::fromU64(1620587195),
        Goldilocks::fromU64(3678822139),
        Goldilocks::fromU64(1824295850),
        Goldilocks::fromU64(366027599),
        Goldilocks::fromU64(1355324045),
        Goldilocks::fromU64(1531026716),
        Goldilocks::fromU64(1017354875),
        Goldilocks::fromU64(0),
        Goldilocks::fromU64(0),
        Goldilocks::fromU64(0),
        Goldilocks::fromU64(0),
        Goldilocks::fromU64(0),
        Goldilocks::fromU64(0),
        Goldilocks::fromU64(0),
        Goldilocks::fromU64(0),
        Goldilocks::fromU64(0),
        Goldilocks::fromU64(1000),
        Goldilocks::fromU64(2046823526),
        Goldilocks::fromU64(324844084),
        Goldilocks::fromU64(2227533143),
        Goldilocks::fromU64(4019066851),
        Goldilocks::fromU64(2813851449),
        Goldilocks::fromU64(3704192333),
        Goldilocks::fromU64(4283076590),
        Goldilocks::fromU64(1639128234),
        Goldilocks::fromU64(4132056474),
        Goldilocks::fromU64(3588173260),
        Goldilocks::fromU64(2226075649),
        Goldilocks::fromU64(3791783573),
        Goldilocks::fromU64(459514060),
        Goldilocks::fromU64(4265611735),
        Goldilocks::fromU64(3195494985),
        Goldilocks::fromU64(118230042),
        Goldilocks::fromU64(0),
        Goldilocks::fromU64(0),
        Goldilocks::fromU64(0),
        Goldilocks::fromU64(0),
        Goldilocks::fromU64(0),
        Goldilocks::fromU64(0),
        Goldilocks::fromU64(0),
        Goldilocks::fromU64(0),
        Goldilocks::fromU64(1),
        Goldilocks::fromU64(17096957522471052170ULL),
        Goldilocks::fromU64(3393099452376483046ULL),
        Goldilocks::fromU64(13454683351222426301ULL),
        Goldilocks::fromU64(7169267199863900071ULL)};

    json publicStarkJson;
    for (int i = 0; i < 47; i++)
    {
        publicStarkJson[i] = Goldilocks::toString(publicInputs[i]);
    }
    ZkevmSteps zkevmSteps;
    starks.genProof(fproof, &publicInputs[0], &zkevmSteps);

    nlohmann::ordered_json jProof = fproof.proofs.proof2json();
    nlohmann::json zkin = proof2zkinStark(jProof);
    // Generate publics
    jProof["publics"] = publicStarkJson;
    zkin["publics"] = publicStarkJson;

    json2file(zkin, "zkin.json");
    json2file(jProof, "jProof.json");

    return 0;
}